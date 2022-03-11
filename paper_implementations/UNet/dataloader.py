import os
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset


class NuclieData(Dataset):
        def __init__(self, path='data/pp_train_imgs', transform=None):
            self.path = path
            self.folders = os.listdir(path)
            self.transform = transform
        
        def __len__(self):
            return len(self.folders)
        
        def __getitem__(self,idx):
            image_folder = os.path.join(self.path, self.folders[idx], 'images/')
            mask_folder = os.path.join(self.path, self.folders[idx], 'masks/')
            
            image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
            mask_path = os.path.join(mask_folder, os.listdir(mask_folder)[0])
            img = io.imread(image_path)[:,:,:3].astype('float32')
            img = transform.resize(img, (128, 128))
            
            mask = np.load(mask_path).astype('float32')

            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            mask = mask.permute(2, 0, 1)
            return (img, mask) 


class NuclieTestData(Dataset):
        def __init__(self, path='stage2_test_final', transform=None):
            self.path = path
            self.folders = os.listdir(path)
            self.transform = transform

        def __len__(self):
            return len(self.folders)
        
        def __getitem__(self,idx):
            image_folder = os.path.join(self.path, self.folders[idx], 'images/')            
            image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
            img = io.imread(image_path)[:,:,:3].astype('float32')
            img = transform.resize(img, (128, 128))
            augmented = self.transform(image=img)
            img = augmented['image']
            return img 