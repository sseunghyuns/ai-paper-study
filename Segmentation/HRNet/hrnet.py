import torch
import torch.nn as nn
import torch.nn.functional as F

# 1
class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x) # [1, 3, 512, 512] -> [1, 64, 128, 128]    

# 2
class Stage01Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )
        
        if in_channels==64:
            self.identity_block = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256)
            )

        self.relu = nn.ReLU()
        self.in_channels = in_channels
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        
        if self.in_channels == 64:
            identity = self.identity_block(identity)
        
        out += identity # skip connection
        
        return self.relu(out) # [1, 64, 128, 128] -> [1, 256, 128, 128]

# 3
class Stage01StreamGenerateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_res_block = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        self. medium_res_block = nn.Sequential(
            nn.Conv2d(256, 96, kernel_size=3, stride=2, padding=1, bias=False), # strided conv로 해상도 1/2배
            nn.BatchNorm2d(96),
            nn.ReLU() 
        )
    def forward(self, x):
        out_high = self.high_res_block(x)
        out_medium = self.medium_res_block(x)
        return out_high, out_medium

# 4
class StageBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity
        return self.relu(out)

# 5
class Stage02(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 4회 반복
        high_res_blocks = [StageBlock(48) for _ in range(4)]
        medium_res_blocks = [StageBlock(96) for _ in range(4)]

        self.high_res_blocks = nn.Sequential(*high_res_blocks)
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks)

    def forward(self, x_high, x_medium):
        out_high = self.high_res_blocks(x_high)
        out_medium = self.medium_res_blocks(x_medium)
        return out_high, out_medium

# 6
class Stage02Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96)
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x_high, x_medium):
        high_size = (x_high.shape[-1], x_high.shape[-2])
        
        med2high = self.medium_to_high(x_medium) # "bilinear upsampling followed by a 1x1 conv" 
        med2high = F.interpolate(med2high, size=high_size, mode='bilinear', align_corners=True)
        
        high2med = self.high_to_medium(x_high)
        
        out_high = x_high + med2high # concat X, sum O
        out_med = x_medium + high2med
        
        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        
        return out_high, out_med

# 7
# Fuse 후, 하위 stream 생성
class StreamGenerateBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            # 채널 2배, 해상도 1/2배
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

# 8
class Stage03(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 4회 반복
        high_res_blocks = [StageBlock(48) for _ in range(4)]
        medium_res_blocks = [StageBlock(96) for _ in range(4)]
        low_res_blocks = [StageBlock(192) for _ in range(4)]
        
        self.high_res_blocks = nn.Sequential(*high_res_blocks)
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks)
        self.low_res_blocks = nn.Sequential(*low_res_blocks)

    def forward(self, x_high, x_medium, x_low):
        out_high = self.high_res_blocks(x_high)
        out_medium = self.medium_res_blocks(x_medium)
        out_low = self.low_res_blocks(x_low)
        return out_high, out_medium, out_low

# 9
class Stage03Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96)
        )
        
        self.high_to_low = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192)
        )
        
        self.medium_to_low = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192)
        )
        
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(48)
        )        
        
        self.low_to_high = nn.Sequential(
            nn.Conv2d(192, 48, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(48)
        ) 
        
        self.low_to_medium = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(96)
        )          
        
        self.relu = nn.ReLU()
        
    def forward(self, x_high, x_medium, x_low):
        high_size = (x_high.shape[-1], x_high.shape[-2])
        medium_size = (x_medium.shape[-1], x_medium.shape[-2])
    
        low2high = F.interpolate(x_low, size=high_size, mode="bilinear", align_corners=True)
        low2high = self.low_to_high(low2high)
        
        low2med = F.interpolate(x_low, size=medium_size, mode="bilinear", align_corners=True)
        low2med = self.low_to_medium(low2med)
        
        med2high = F.interpolate(x_medium, size=high_size, mode="bilinear", align_corners=True)
        med2high = self.medium_to_high(med2high)
        
        med2low = self.medium_to_low(x_medium)
        
        high2med = self.high_to_medium(x_high)
        high2low = self.high_to_low(x_high)
        
        out_high = x_high + med2high + low2high 
        out_medium = x_medium + high2med + low2med
        out_low = x_low + high2low + med2low
        
        out_high = self.relu(out_high)
        out_medium = self.relu(out_medium)
        out_low = self.relu(out_low)

        return out_high, out_medium, out_low

# 10
class Stage04(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 4회 반복
        high_res_blocks = [StageBlock(48) for _ in range(4)]
        medium_res_blocks = [StageBlock(96) for _ in range(4)]
        low_res_blocks = [StageBlock(192) for _ in range(4)]
        vlow_res_blocks = [StageBlock(384) for _ in range(4)]

        self.high_res_blocks = nn.Sequential(*high_res_blocks)
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks)
        self.low_res_blocks = nn.Sequential(*low_res_blocks)
        self.vlow_res_blocks = nn.Sequential(*vlow_res_blocks)

    def forward(self, x_high, x_medium, x_low, x_vlow):
        out_high = self.high_res_blocks(x_high)
        out_medium = self.medium_res_blocks(x_medium)
        out_low = self.low_res_blocks(x_low)
        out_vlow = self.vlow_res_blocks(x_vlow)
        return out_high, out_medium, out_low, out_vlow

# 11
class Stage04Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96)
        )
        
        # Stream을 두 단계 낮추는 경우. 첫 번째 strided conv는 채널 수를 그대로 유지하며 두 번째 strided conv에서 해당 stream과 채널 수를 동일하게 맞춘다
        self.high_to_low = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192)
        )
        
        self.high_to_vlow = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384)
        )
                

        self.medium_to_low = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192)
        )
        
        self.medium_to_vlow = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384)
        )        
        
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(48)
        )        
        
        self.low_to_high = nn.Sequential(
            nn.Conv2d(192, 48, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(48)
        ) 
        
        self.low_to_medium = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(96)
        )  
        
        self.low_to_vlow = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384)
        ) 
        
        self.vlow_to_low = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(192)        
        )
        
        
        self.vlow_to_medium = nn.Sequential(
            nn.Conv2d(384, 96, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(96)        
        )
        
        self.vlow_to_high = nn.Sequential(
            nn.Conv2d(384, 48, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(48)        
        )
            
        self.relu = nn.ReLU()
        
    def forward(self, x_high, x_medium, x_low, x_vlow):
        high_size = (x_high.shape[-1], x_high.shape[-2])
        medium_size = (x_medium.shape[-1], x_medium.shape[-2])
        low_size = (x_low.shape[-1], x_low.shape[-2])
        
        vlow2low = F.interpolate(x_vlow, size=low_size, mode="bilinear", align_corners=True)
        vlow2low = self.vlow_to_low(vlow2low)
        
        vlow2med = F.interpolate(x_vlow, size=medium_size, mode="bilinear", align_corners=True)
        vlow2med = self.vlow_to_medium(vlow2med)
        
        vlow2high = F.interpolate(x_vlow, size=high_size, mode="bilinear", align_corners=True)
        vlow2high = self.vlow_to_high(vlow2high)
        
        low2high = F.interpolate(x_low, size=high_size, mode="bilinear", align_corners=True)
        low2high = self.low_to_high(low2high)
        
        low2med = F.interpolate(x_low, size=medium_size, mode="bilinear", align_corners=True)
        low2med = self.low_to_medium(low2med)
        low2vlow = self.low_to_vlow(x_low)
        
        med2high = F.interpolate(x_medium, size=high_size, mode="bilinear", align_corners=True)
        med2high = self.medium_to_high(med2high)
        med2low = self.medium_to_low(x_medium)
        med2vlow = self.medium_to_vlow(x_medium)

        high2med = self.high_to_medium(x_high)
        high2low = self.high_to_low(x_high)
        high2vlow = self.high_to_vlow(x_high)

        out_high = x_high + med2high + low2high + vlow2high
        out_medium = x_medium + high2med + low2med + vlow2med
        out_low = x_low + high2low + med2low + vlow2low
        out_vlow = x_vlow + high2vlow + med2vlow + low2vlow
        
        out_high = self.relu(out_high)
        out_medium = self.relu(out_medium)
        out_low = self.relu(out_low)
        out_vlow = self.relu(out_vlow)

        return out_high, out_medium, out_low, out_vlow

class Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        total_channels = 48 + 96 + 192 + 384
        self.block = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(),
            nn.Conv2d(total_channels, num_classes, kernel_size=1, bias=False)
        )
        
    def forward(self, x_high, x_medium, x_low, x_vlow):
        high_size = (x_high.shape[-1], x_high.shape[-2])
        original_size = (high_size[0]*4, high_size[1]*4)
        
        med2high = F.interpolate(x_medium, size=high_size, mode="bilinear", align_corners=True)
        low2high = F.interpolate(x_low, size=high_size, mode="bilinear", align_corners=True)
        vlow2high = F.interpolate(x_vlow, size=high_size, mode="bilinear", align_corners=True)
        
        out = torch.cat([x_high, med2high, low2high, vlow2high], dim=1)
        out = self.block(out)
        out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=True)
        return out

class HRNetV2(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        
        # Stage 0
        self.input_stemblock = StemBlock() 
        
        # Stage 1
        stage1_layers = []
        for i in range(4):
            if i==0: # channel=64
                stage1_layers.append(Stage01Block(64))
            else:
                stage1_layers.append(Stage01Block(256))
        self.stage1 = nn.Sequential(*stage1_layers)
        self.stage1_stream_generate = Stage01StreamGenerateBlock()
        
        # Stage 2
        self.stage2 = Stage02()
        self.stage2_fuse = Stage02Fuse()
        self.stage2_stream_generate = StreamGenerateBlock(96)
        
        # Stage 3
        self.stage3_1 = Stage03()
        self.stage3_1_fuse = Stage03Fuse()
        self.stage3_2 = Stage03()
        self.stage3_2_fuse = Stage03Fuse()        
        self.stage3_3 = Stage03()
        self.stage3_3_fuse = Stage03Fuse() 
        self.stage3_4 = Stage03()
        self.stage3_4_fuse = Stage03Fuse()
        self.stage3_stream_generate = StreamGenerateBlock(192)
        
        # Stage 4
        self.stage4_1 = Stage04()
        self.stage4_1_fuse = Stage04Fuse()
        self.stage4_2 = Stage04()
        self.stage4_2_fuse = Stage04Fuse()        
        self.stage4_3 = Stage04()
        self.stage4_3_fuse = Stage04Fuse() 
        
        # Head
        self.head = Head(num_classes=11)
        
    def forward(self, x):
        # Stage 1
        out = self.input_stemblock(x) # [1, 64, 128, 128]    
        out = self.stage1(out) # [1, 256, 128, 128]
        out_high, out_med = self.stage1_stream_generate(out) # high: [1, 48, 128, 128], medium: [1, 96, 64, 64]
        
        # Stage 2
        out_high, out_med = self.stage2(out_high, out_med) # high: [1, 48, 128, 128], medium: [1, 96, 64, 64]
        out_high, out_med = self.stage2_fuse(out_high, out_med) # high: [1, 48, 128, 128], medium: [1, 96, 64, 64] / Fusion 후 새로운 stream 생성
        out_low = self.stage2_stream_generate(out_med) # [1, 192, 32, 32]
        
        # Stage 3
        out_high, out_med, out_low = self.stage3_1(out_high, out_med, out_low)
        out_high, out_med, out_low = self.stage3_1_fuse(out_high, out_med, out_low)
        out_high, out_med, out_low = self.stage3_2(out_high, out_med, out_low)
        out_high, out_med, out_low = self.stage3_2_fuse(out_high, out_med, out_low)
        out_high, out_med, out_low = self.stage3_3(out_high, out_med, out_low)
        out_high, out_med, out_low = self.stage3_3_fuse(out_high, out_med, out_low)
        out_high, out_med, out_low = self.stage3_4(out_high, out_med, out_low)
        out_high, out_med, out_low = self.stage3_4_fuse(out_high, out_med, out_low)
        out_vlow = self.stage3_stream_generate(out_low) # [1, 384, 16, 16]
        
        # Stage 4
        out_high, out_med, out_low, out_vlow = self.stage4_1(out_high, out_med, out_low, out_vlow)
        out_high, out_med, out_low, out_vlow = self.stage4_1_fuse(out_high, out_med, out_low, out_vlow)
        out_high, out_med, out_low, out_vlow = self.stage4_2(out_high, out_med, out_low, out_vlow)
        out_high, out_med, out_low, out_vlow = self.stage4_2_fuse(out_high, out_med, out_low, out_vlow)
        out_high, out_med, out_low, out_vlow = self.stage4_3(out_high, out_med, out_low, out_vlow)
        out_high, out_med, out_low, out_vlow = self.stage4_3_fuse(out_high, out_med, out_low, out_vlow)
 
        # Representation Head
        out = self.head(out_high, out_med, out_low, out_vlow)
        
        return out
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    input_tensor = torch.rand((1,3,512,512))
    model = HRNetV2()
    print("Number of params: ", count_parameters(model))
    out = model(input_tensor)
    print("Output shape: ", out.shape)