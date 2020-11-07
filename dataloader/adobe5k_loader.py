from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import os
import cv2

to_tensor = transforms.Compose([
    transforms.ToTensor()
])
    
    
class LoadAdobe(Dataset):
    def __init__(self, dataset_size, dataset_dir='data/Adobe5k/'):
        
        self.inp_dir = os.path.join(dataset_dir,'input')
        self.out_dir = os.path.join(dataset_dir,'output')
        
        self.dataset_size = dataset_size
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if idx==0:
            idx = 1
        str_idx = '0'*(4-len(str(idx)))+str(idx)
        inp = cv2.imread(os.path.join(self.inp_dir, str_idx+'.tif'))
        inp = np.float32(inp)/255.0
        inp = torch.from_numpy(inp.transpose((2,0,1)))
        
        out = cv2.imread(os.path.join(self.out_dir, str_idx+'.jpg'))
        out = np.float32(out)/255.0
        out = torch.from_numpy(out.transpose((2,0,1)))
        
        return inp, out