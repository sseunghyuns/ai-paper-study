import torch
import torch.nn as nn

# nn.ConvTranspose2d, nn.Upsample
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.contract_block1 = DoubleConv(in_channels=in_channels, out_channels=64)
        self.contract_block2 = DoubleConv(in_channels=64, out_channels=128)
        self.contract_block3 = DoubleConv(in_channels=128, out_channels=256)
        self.contract_block4 = DoubleConv(in_channels=256, out_channels=512)

        self.bottleneck = DoubleConv(in_channels=512, out_channels=1024)

        self.extract_block4 = DoubleConv(in_channels=1024, out_channels=512)
        self.extract_block3 = DoubleConv(in_channels=512, out_channels=256)
        self.extract_block2 = DoubleConv(in_channels=256, out_channels=128)
        self.extract_block1 = DoubleConv(in_channels=128, out_channels=64)
        
        # up / down sample
        self.downsample = nn.MaxPool2d(2,2)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        # output layer
        self.out_layer = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):                          # [B, 1, 128, 128]
       
        cont1 = self.contract_block1(inputs)            # [B, 64, 128, 128]
        cont1_down = self.downsample(cont1)             # [B, 64, 64, 64]

        cont2 = self.contract_block2(cont1_down)        # [B, 128, 64, 64]    
        cont2_down = self.downsample(cont2)             # [B, 128, 32, 32]    

        cont3 = self.contract_block3(cont2_down)        # [B, 256, 32, 32]
        cont3_down = self.downsample(cont3)             # [B, 256, 16, 16]
        
        cont4 = self.contract_block4(cont3_down)        # [B, 512, 16, 16]  
        cont4_down = self.downsample(cont4)             # [B, 512, 8, 8]
        
        bot = self.bottleneck(cont4_down)               # [B, 1024, 8, 8]
        bot_upsample = self.up_conv4(bot)               # [B, 512, 16, 16]

        extr4 = torch.cat((cont4, bot_upsample), dim=1) # [B, 1024, 16, 16]
        extr4 = self.extract_block4(extr4)              # [B, 512, 16, 16]   

        extr3 = self.up_conv3(extr4)                    # [B, 256, 32, 32]
        extr3 = torch.cat((cont3, extr3), dim=1)        # [B, 512, 32, 32]
        extr3 = self.extract_block3(extr3)              # [B, 512, 32, 32])

        extr2 = self.up_conv2(extr3)                    # [B, 128, 64, 64]
        extr2 = torch.cat((cont2, extr2), dim=1)        # [B, 256, 64, 64])
        extr2 = self.extract_block2(extr2)              # [B, 128, 64, 64]

        extr1 = self.up_conv1(extr2)                    # [B, 64, 128, 128]
        extr1 = torch.cat((cont1, extr1), dim=1)        # [B, 128, 128, 128]
        extr1 = self.extract_block1(extr1)              # [B, 64, 128, 128]

        out = self.out_layer(extr1)                     # [B, 1, 128, 128]
        out = self.sigmoid(out)

        return out