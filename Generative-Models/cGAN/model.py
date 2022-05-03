import torch
import torch.nn as nn


class FCLayer(nn.Module):
    def __init__(self, in_dims, out_dims, normalize=True):
        super().__init__()
        self.fc = nn.Linear(in_dims, out_dims)
        self.bn = nn.BatchNorm1d(out_dims)
        self.relu = nn.ReLU(0.5)
        self.normalize = normalize

    def forward(self, x):
        x = self.fc(x)

        if self.normalize:
            x = self.bn(x)
        
        x = self.relu(x)
        return x

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1_1 = FCLayer(100, 256) # 100이 optimal한 크기는 아님
        self.fc1_2 = FCLayer(10, 256)
        self.fc2 = FCLayer(512, 512)
        self.fc3 = FCLayer(512, 1024)

        self.fc_out = nn.Linear(1024, 784)
        self.dropout = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
    

    def forward(self, noise, label):
        z = self.fc1_1(noise)
        y = self.fc1_2(label)

        x = torch.concat((z,y), dim=-1)

        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = self.fc_out(x)
        x = self.tanh(x) # 논문에서는 Sigmoid를 사용했는데, 이는 데이터 전처리에 따라 달라질 수 있음 
        return x

# Discriminator
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1_1 = FCLayer(784, 1024, False)
        self.fc1_2 = FCLayer(10, 1024, False)
    
        self.fc2 = FCLayer(2048, 512)
        self.fc3 = FCLayer(512, 256)
        
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, label):
        x = self.fc1_1(x)
        y = self.fc1_2(label)

        x = torch.concat((x,y), dim=-1)

        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x