import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class ImageToPatch:
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, image):
        '''
        unfold: https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
        x.unfold(dimension, size of each patch, stride)
        '''
        n_channels = image.size(0)
        patches = image.unfold(1, self.patch_size, self.patch_size) # H 방향으로 slicing
        patches = patches.unfold(2, self.patch_size, self.patch_size) # W 방향으로 slicing
        patches = patches.reshape(n_channels, -1, self.patch_size, self.patch_size) # C, N_Patch, Patch_size, Patch_size
        patches = patches.permute(1,0,2,3) # n_channel x P x P 크기의 패치가 총 num_patch 존재
        num_patch = patches.size(0)
        patches = patches.reshape(num_patch,-1)
        
        return patches

class PatchDataset:
    
    def __init__(self, patch_size=16, img_size=256, batch_size=64):
        self.patch_size = patch_size
        self.img_size = img_size
        self.batch_size = batch_size
    
    def load_dataset(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose([transforms.Resize(self.img_size), 
                                              transforms.RandomCrop(self.img_size, padding=2),
                                              transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                   (0.2023, 0.1994, 0.2010)),
                                              ImageToPatch(self.patch_size)]) # Tensor Image -> Patch
        
        test_transform = transforms.Compose([transforms.Resize(self.img_size), 
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                   (0.2023, 0.1994, 0.2010)),
                                             ImageToPatch(self.patch_size)])
        
        # Load CIFAR10
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        val_index = list(range(0, len(test_set), 2))
        test_index = list(range(1, len(test_set), 2))
        valid_dataset = torch.utils.data.Subset(test_set, val_index)
        test_dataset = torch.utils.data.Subset(test_set, test_index)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        
        return train_loader, valid_loader, test_loader