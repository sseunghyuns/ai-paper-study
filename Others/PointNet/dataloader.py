import torch
import numpy as np 
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    def __init__(self, dataframe, data, label, seg, random_translate=False, random_jitter=False, random_rotate=False):
        super().__init__()
        self.df = dataframe
        self.data = data
        self.label = label
        self.seg = seg
        self.class_name = dataframe['label']
#         self.seg_num_all = dataframe['segmentation_part_num']
        self.random_translate = random_translate
        self.random_jitter = random_jitter
        self.random_rotate = random_rotate
        self.seg_num_all = 50
        self.seg_start_index = 0

    def __getitem__(self, index):
        points = self.data[index]
        label = self.label[index]
        seg = self.seg[index]
        c_name = self.class_name[index]
#         seg_num_all = self.seg_num_all[index]
        if self.random_translate:
            points = translate_pointcloud(points)
        if self.random_jitter:
            points = jitter_pointcloud(points)
        if self.random_rotate:
            points = rotate_pointcloud(points)
        
        # categorical vector
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        
        
        points = torch.from_numpy(points)
        label = torch.from_numpy(label)
        seg = torch.from_numpy(seg)
        
        return points, label, label_one_hot.squeeze(1), seg, c_name#, seg_num_all 
    
    def __len__(self):
        return self.df.shape[0]