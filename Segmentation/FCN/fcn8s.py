# 모델 참고 코드 
# https://github.com/wkentaro/pytorch-fcn/
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, add_conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2,2,ceil_mode=True)
        
        self.add_conv = add_conv
        if self.add_conv:
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.add_conv:
            x = self.relu(self.conv3(x)) 
        x = self.maxpool(x)
        return x
        
        
class FCN8s(nn.Module):
    def __init__(self, num_classes=21, input_channels=3):
        super(FCN8s, self).__init__()
        self.convb1 = ConvBlock(input_channels, 64)
        self.convb2 = ConvBlock(64, 128)
        self.convb3 = ConvBlock(128, 256, add_conv=True)
        self.convb4 = ConvBlock(256, 512, add_conv=True)
        self.convb5 = ConvBlock(512, 512, add_conv=True)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6=nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout2d()
        
        # Score
        self.score = nn.Conv2d(4096, num_classes, 1) # [N, 11, 7, 7]
        
        # Up Score1
        self.up_score1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        
        # Up Score2
        self.up_score2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        # Point wise for conv4 output
        self.conv4_1 = nn.Conv2d(512, num_classes, 1)
        
        # Point wise for conv3 output
        self.conv3_1 = nn.Conv2d(256, num_classes, 1)
        
        # Deconv for Final Prediction
        self.up_score_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)
        
    def forward(self, x):
  
        x1 = self.convb1(x) # [1, 64, 112, 112]
        x2 = self.convb2(x1) # [1, 128, 56, 56]
        x3 = self.convb3(x2) # [1, 256, 28, 28]
        x4 = self.convb4(x3) # [1, 512, 14, 14] -> up_score1과 summation
        x5 = self.convb5(x4) # [1, 512, 7, 7]
        
        x6 = self.dropout6(self.relu6(self.fc6(x5))) # [1, 4096, 7, 7]
        x7 = self.dropout7(self.relu7(self.fc7(x6))) # [1, 4096, 7, 7]
        score = self.score(x7) # [1, 11, 7, 7]
        up_score1 = self.up_score1(score) # [1, 11, 14, 14]
        
        x4_1 = self.conv4_1(x4) # [1, 11, 14, 14]
        up_score1_x4 = up_score1 + x4_1 # [1, 11, 28, 28]

        up_score2 = self.up_score2(up_score1_x4)
        x3_1 = self.conv3_1(x3) # [1, 11, 28, 28]
        up_score2_x3 = up_score2 + x3_1 # [1, 11, 28, 28]
        
        output = self.up_score_final(up_score2_x3)

        return output # [1, 11, 224, 224]