
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


class T_Net(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x): # x shape: [32, 3, 2048] (첫 번째 Transformation network에서는 3x3를 학습함)
        batchsize = x.size()[0] # 32
        
        ########################
        ### MLP(64,128,1024) ###
        ########################
        x = F.relu(self.bn1(self.conv1(x))) # [32, 64, 2048]
        x = F.relu(self.bn2(self.conv2(x))) # [32, 128, 2048]
        x = F.relu(self.bn3(self.conv3(x))) # [32, 1024, 2048]
        
        ##############################
        ### Max Pool across Points ###
        ##############################
        x = torch.max(x, 2, keepdim=True)[0] # [32, 1024, 1]
        x = x.view(-1, 1024) # [32, 1024]
        
        #############################
        ### 2 FC layers(512, 256) ###
        #############################
        x = F.relu(self.bn4(self.fc1(x))) # [32, 512]
        x = F.relu(self.bn5(self.fc2(x))) # [32, 256]
        x = self.fc3(x) # [32, 9]

        ########################################
        ### Intialized as an Identity Matrix ###
        ########################################
        # np.eye(self.k) => [3, 3]
        # np.eye(self.k).flatten() => (9,)
        # np.eye(self.k).flatten().view(1, 3*3) => [1,9]
        # np.eye(self.k).flatten().view(1, 3*3).repeat(batchsize, 1) => [32, 9]
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1) # [32, 9]
        
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden # [32, 9]
        x = x.view(-1, self.k, self.k) # [32, 3, 3]
        return x # [32, 3, 3]
    
    
class PointNet(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
        channel = 3
        self.part_num = part_num
        
        self.stn1 = T_Net(k=channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.stn2 = T_Net(k=128)
        
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        
        self.conv_seg1 = torch.nn.Conv1d(3024, 256, 1)
        self.conv_seg2 = torch.nn.Conv1d(256, 256, 1)
        self.conv_seg3 = torch.nn.Conv1d(256, 128, 1)
        self.conv_seg4 = torch.nn.Conv1d(128, part_num, 1)
        self.bn_seg1 = nn.BatchNorm1d(256)
        self.bn_seg2 = nn.BatchNorm1d(256)
        self.bn_seg3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        # label shape: [32, 1, 16](one-hot vector)
        B, D, N = point_cloud.size() # [32, 3, 2048]
        
        #######################
        ### Input Transform ###
        #######################
        trans = self.stn1(point_cloud) # [32, 3, 3]
        point_cloud = point_cloud.transpose(2, 1) # [32, 2048, 3]
        point_cloud = torch.bmm(point_cloud, trans) # [32, 2048, 3]
        point_cloud = point_cloud.transpose(2, 1) # [32, 3, 2048]

        #########################
        ### MLP(64, 128, 128) ###
        #########################
        out1 = F.relu(self.bn1(self.conv1(point_cloud))) # [32, 64, 2048]
        out2 = F.relu(self.bn2(self.conv2(out1))) # [32, 128, 2048]
        out3 = F.relu(self.bn3(self.conv3(out2))) # [32, 128, 2048]

        
        #########################
        ### Feature Transform ###
        #########################
        trans_feat = self.stn2(out3) # [32, 128, 128] 
        x = out3.transpose(2, 1) # [32, 2048, 128]
        net_transformed = torch.bmm(x, trans_feat) # [32, 2048, 128]
        net_transformed = net_transformed.transpose(2, 1) # [32, 128, 2048]
        
        ######################
        ### MLP(512, 2048) ###
        ######################
        out4 = F.relu(self.bn4(self.conv4(net_transformed))) # [32, 512, 2048]
        out5 = self.bn5(self.conv5(out4)) # [32, 2048, 2048]
        
        ################
        ### Max Pool ###
        ################
        out_max = torch.max(out5, 2, keepdim=True)[0] # [32, 2048, 1] => Global Feature
        out_max = out_max.view(-1, 2048) # [32, 2048]
        
        ############################
        ### Segmentation Network ###
        ############################
        out_max = torch.cat([out_max, label.squeeze(1)],1) # [32, 2064]
        expand = out_max.view(-1, 2064, 1).repeat(1, 1, N) # [32, 2064, 2048]
        concat = torch.cat([expand, out1, out2, out3, net_transformed, out4], 1) # [32, 3024, 2048]

        #############################
        ### MLP(256, 256, 128, m) ###
        #############################
        net = F.relu(self.bn_seg1(self.conv_seg1(concat))) # [32, 256, 2048]
        net = F.relu(self.bn_seg2(self.conv_seg2(net))) # [32, 256, 2048]
        net = F.relu(self.bn_seg3(self.conv_seg3(net))) # [32, 128, 2048]
        net = self.conv_seg4(net) # [32, 50, 2048]
        
        #####################
        ### Output Scores ###
        #####################
        net = net.transpose(2, 1).contiguous() # [32, 2048, 50]
        net = F.log_softmax(net.view(-1, self.part_num), dim=-1) # [65536, 50]
        net = net.view(B, N, self.part_num) # [32, 2048, 50] = [B, N, 50]

        return net, trans_feat