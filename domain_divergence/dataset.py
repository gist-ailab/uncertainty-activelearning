import os,sys
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Subset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

# Label idx, Unlabel idx를 받아서 label loader와 unlabel loader를  return
class DATALOADERS():
    def __init__(self, lbl_idx, ulbl_idx, batch_size, train_transform, test_transform, dataset='cifar10', datapath = ''):
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(datapath, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(datapath, download=True, transform=test_transform, train=False)
        lbl_subset = SubsetRandomSampler(lbl_idx)
        ulbl_subset = SubsetRandomSampler(ulbl_idx)
        
        self.lbl_train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=lbl_subset,
                                      drop_last=True, shuffle=False)
        self.ulbl_train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=ulbl_subset,
                                       drop_last=True, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    def get_loaders(self):
        return self.lbl_train_loader, self.ulbl_train_loader, self.test_loader

# Label idx, Unlabel idx를 받아서 Binary class data loader를 return
class BINARYSET(Dataset):
    def __init__(self, lbl_idx, ulbl_idx, transform, dataset='cifar10', datapath=''):
        if dataset == 'cifar10':
            self.train_dataset = datasets.CIFAR10(datapath, download=True, transform=transform)
        self.label_info = [0 for i in range(len(lbl_idx)+len(ulbl_idx))]
        for idx in lbl_idx:
            self.label_info[idx] = 1
        for idx in ulbl_idx:
            self.label_info[idx] = 0
        
    def __len__(self):
        return len(self.label_info)
        
    def __getitem__(self, idx):
        data, lbl = self.train_dataset[idx], self.label_info[idx]
        return data,lbl

def BINARYLOADER(lbl_idx, ulbl_idx, batch_size, transform, dataset='cifar10', datapath=''):
    binaryset = BINARYSET(lbl_idx, ulbl_idx, transform, dataset, datapath)
    binary_loader = DataLoader(binaryset, batch_size, shuffle=True)
    return binary_loader