from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import os
import cv2

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

class LoadDPED(Dataset):
    def __init__(self, dataset_dir, phone, dataset_size, test=False):
        if test:
            self.inp_dir = os.path.join(dataset_dir, phone, 'test_data','patches', phone)
            self.canon_dir = os.path.join(dataset_dir, phone, 'test_data','patches', 'canon')
        else:
            self.inp_dir = os.path.join(dataset_dir, phone, 'training_data', phone)
            self.canon_dir = os.path.join(dataset_dir, phone, 'training_data', 'canon')
        
        self.dataset_size = dataset_size
        self.phone = phone
        self.test = test
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        inp = cv2.imread(os.path.join(self.inp_dir, str(idx)+'.jpg'))
        inp = np.float32(inp)/255.0
        inp = torch.from_numpy(inp.transpose((2,0,1)))
        
        out = cv2.imread(os.path.join(self.canon_dir, str(idx)+'.jpg'))
        out = np.float32(out)/255.0
        out = torch.from_numpy(out.transpose((2,0,1)))
        
        return inp, out
    
    
class LoadAdobe(Dataset):
    def __init__(self, dataset_dir, phone, dataset_size, test=False):
        
        if test:
            self.inp_dir = os.path.join(dataset_dir,'fivek_dataset/sup_batched80aug_daylight')
            self.canon_dir = os.path.join(dataset_dir, phone, 'test_data','patches', 'canon')
        else:
            self.inp_dir = os.path.join(dataset_dir, phone, 'training_data', phone)
            self.canon_dir = os.path.join(dataset_dir, phone, 'training_data', 'canon')
        
        self.dataset_size = dataset_size
        self.phone = phone
        self.test = test
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        inp = cv2.imread(os.path.join(self.inp_dir, str(idx)+'.jpg'))
        inp = np.float32(inp)/255.0
        inp = torch.from_numpy(inp.transpose((2,0,1)))
        
        out = cv2.imread(os.path.join(self.canon_dir, str(idx)+'.jpg'))
        out = np.float32(out)/255.0
        out = torch.from_numpy(out.transpose((2,0,1)))
        
        return inp, out