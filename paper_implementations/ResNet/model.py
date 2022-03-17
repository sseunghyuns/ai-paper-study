import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super().__init__()
        layers = []
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.activation = activation


    def forward(self, x):
        
        ##fill it##
        x = self.conv(x)
        x = self.bn(x)
        
        if self.activation:
            x = self.relu(x)   
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True, downsample=None):
        super().__init__()
        layers = []
        
        ##fill##
        for _ in range(2):
            layers.append(ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation))
            in_channels = out_channels
            activation = False
            stride=1
        
        self.relu = nn.ReLU()
        self.resblk = nn.Sequential(*layers)
        self.downsample = downsample
        
    def forward(self, x):
        ##fill##
        identity_x = x

        x = self.resblk(x)
        if self.downsample is not None:
            identity_x = self.downsample(identity_x)
        # print(x.shape, identity_x.shape)
        x += identity_x
        x = self.relu(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, num_blocks=[3,4,6,3]):
        super().__init__()

        self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = 64

        ##fill##
        self.layer1 = self.make_layers(out_channels=64, num_block=num_blocks[0])
        self.layer2 = self.make_layers(out_channels=128, num_block=num_blocks[1], stride=2)
        self.layer3 = self.make_layers(out_channels=256, num_block=num_blocks[2], stride=2)
        self.layer4 = self.make_layers(out_channels=512, num_block=num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(nker*2*2*2, out_channels)

    def make_layers(self, out_channels, num_block, stride=1):
        downsample = None
        layers=[]

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels)
            )

        layers.append(ResBlock(self.in_channels, out_channels, kernel_size=3, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        for _ in range(num_block-1):
            layers.append(ResBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.enc(x)
        x = self.max_pool(x)
        ##fill##
        # print('1', x.shape)
        x = self.layer1(x)
        # print('2', x.shape)
        x = self.layer2(x)
        # print('3', x.shape)
        x = self.layer3(x)
        # print('4', x.shape)
        x = self.layer4(x)
        # print('5', x.shape)
        x = self.avgpool(x)
        # print('6', x.shape)
        x = x.reshape(x.shape[0], -1)
        # print('7', x.shape)
        out = self.fc(x)

        return out