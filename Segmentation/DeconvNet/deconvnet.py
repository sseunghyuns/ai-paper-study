'''
reference 
http://cvlab.postech.ac.kr/research/deconvnet/model/DeconvNet/DeconvNet_inference_deploy.prototxt
'''
import torch
import torch.nn as nn
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, extra_conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.extra_conv = extra_conv
        if self.extra_conv:
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
            self.batchnorm3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        if self.extra_conv:
            x = self.relu(self.batchnorm3(self.conv3(x)))
            
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, extra_conv=False):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU(inplace=True)

        self.extra_conv = extra_conv
        if self.extra_conv:
            self.deconv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding)
            self.batchnorm3 = nn.BatchNorm2d(out_channels) 
        
    def forward(self, x):
        x = self.relu(self.batchnorm1(self.deconv1(x)))
        x = self.relu(self.batchnorm2(self.deconv2(x)))
        if self.extra_conv:
            x = self.relu(self.batchnorm3(self.deconv3(x)))
        return x

            
            
class DeconvNet(nn.Module):
    def __init__(self, input_channels, num_classes=21):
        super(DeconvNet, self).__init__()
        self.convb1 = ConvBlock(input_channels, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True, return_indices=True)
        
        self.convb2 = ConvBlock(64, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True, return_indices=True)

        self.convb3 = ConvBlock(128, 256, 3, 1, 1, True)
        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True, return_indices=True)

        self.convb4 = ConvBlock(256, 512, 3, 1, 1, True)
        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True, return_indices=True)

        self.convb5 = ConvBlock(512, 512, 3, 1, 1, True)
        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True, return_indices=True)
        
        self.fc6 = nn.Conv2d(512, 4096, 7, 1, 0) 
        self.drop6 = nn.Dropout2d(0.5)

        self.fc7 = nn.Conv2d(4096, 4096, 1, 1, 0)
        self.drop7 = nn.Dropout2d(0.5)

        self.deconv6 = nn.ConvTranspose2d(4096, 512, 7, 1, 0) ##
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        
        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.deconvb5 = DeconvBlock(512, 512, 3, 1, 1, True)

        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.deconvb4 = DeconvBlock(512, 256, 3, 1, 1, True)
        
        self.unpool3 = nn.MaxUnpool2d(2, 2) 
        self.deconvb3 = DeconvBlock(256, 128, 3, 1, 1, True)
        
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.deconvb2 = DeconvBlock(128, 64, 3, 1, 1, False)
        
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.deconvb1 = DeconvBlock(64, 64, 3, 1, 1, False)
        
        self.score = nn.Conv2d(64, num_classes, 1, 1, 0, 1)
        
    def forward(self, x):
        x = self.convb1(x) # [1, 64, 224, 224]
        x, pool1_indices = self.pool1(x) # [1, 64, 112, 112]

        x = self.convb2(x) # [1, 128, 112, 112]
        x, pool2_indices = self.pool2(x) # [1, 128, 56, 56]

        x = self.convb3(x) # [1, 256, 56, 56]
        x, pool3_indices = self.pool3(x) # [1, 256, 28, 28]

        x = self.convb4(x) # [1, 512, 28, 28]
        x, pool4_indices = self.pool4(x) # [1, 512, 14, 14]
        
        x = self.convb5(x) # [1, 512, 14, 14]
        x, pool5_indices = self.pool5(x) # [1, 512, 7, 7]
        
        x = self.fc6(x) # [1, 4096, 1, 1]
        x = self.drop6(x)
        
        x = self.fc7(x) # [1, 4096, 1, 1]
        x = self.drop7(x)
        
        x = self.relu(self.batchnorm6(self.deconv6(x))) # [1, 512, 7, 7]
        
        x = self.unpool5(x, pool5_indices) # [1, 512, 14, 14]
        x = self.deconvb5(x) # [1, 512, 14, 14]
        
        x = self.unpool4(x, pool4_indices) # [1, 512, 28, 28]
        x = self.deconvb4(x) # [1, 256, 28, 28]

        x = self.unpool3(x, pool3_indices) # [1, 256, 56, 56]
        x = self.deconvb3(x) # [1, 128, 56, 56]

        x = self.unpool2(x, pool2_indices) # [1, 128, 112, 112]
        x = self.deconvb2(x) # [1, 64, 112, 112]

        x = self.unpool1(x, pool1_indices) # [1, 64, 224, 224]
        x = self.deconvb1(x) # [1, 64, 224, 224]
        
        x = self.score(x) # [1, 11, 224, 224]
        return x