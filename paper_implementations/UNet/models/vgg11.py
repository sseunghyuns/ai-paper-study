import torch
import torch.nn as nn
from torchvision import models

class VGGSegmentation(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        vgg = models.vgg11(pretrained=True)

        self.backbone = vgg.features[:20]

        self.conv_out = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.conv_out(x)
        x = self.upsample(x)
        out = self.sigmoid(x)
        return out