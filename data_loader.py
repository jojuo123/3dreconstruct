import torch
import os
import pandas as pd 
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class DatasetCustom(Dataset):
    def __init__(self, rootdir, csvfile, transform=None):
        self.datalist = pd.read_csv(csvfile)
        self.rootdir = rootdir
        self.transform = transform

    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.rootdir, self.datalist.iloc[idx, 0]+'.png')
        image = io.imread(img_name)
        image = np.flip(image, 2).copy()
        landmark_file = os.path.join(self.rootdir, 'lm_bin/'+self.datalist.iloc[idx, 0]+'.bin')
        lm = np.memmap(landmark_file, dtype='float64', mode='r')
        lm = np.reshape(lm, [68, 2])
        mask_file = os.path.join(self.rootdir, 'mask/'+self.datalist.iloc[idx, 0]+'.png')
        mask = io.imread(mask_file)
        sample = {
            'image': image, 
            'landmark': lm,
            'mask': mask
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, lm, mask = sample['image'], sample['landmark'], sample['mask']
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return {
            'image': (torch.from_numpy(image)).type(torch.float32), 
            'landmark': torch.from_numpy(lm),
            'mask': (torch.from_numpy(mask)).type(torch.float32)
        }
def data_loader(rootdir, batchsize=4, num_workers=0, csvfile='data.csv'):
    transformed_data = DatasetCustom(csvfile=csvfile, 
    rootdir=rootdir,
    transform=transforms.Compose([ToTensor()])
    )
    dataloader = DataLoader(transformed_data, batch_size=batchsize, shuffle=True, num_workers=0)
    return dataloader