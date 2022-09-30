import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import pickle
from torchvision import datasets
import torchvision.models as models
from utils import *
from torch.utils.data.sampler import SubsetRandomSampler
import random

seed = 6
id = 0
random.seed(seed)
torch.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 1. a,b,c로 나누기
train_dataset = datasets.CIFAR10('/ailab_mat/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         transform=get_rand_augment('cifar10'))
test_dataset = datasets.CIFAR10('/ailab_mat/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         train = False,
                                         transform=get_test_augment('cifar10'))

total_idx = [i for i in range(50000)]
random.shuffle(total_idx)

with open(f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/init_final2/subsets/cifar100_1/seed{id}/subset_a.pkl', 'rb') as f:
    subset_a_idx = pickle.load(f)
with open(f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/init_final2/subsets/cifar100_1/seed{id}/subset_b.pkl', 'rb') as f:
    subset_b_idx = pickle.load(f)
with open(f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/init_final2/subsets/cifar100_1/seed{id}/subset_c.pkl', 'rb') as f:
    subset_c_idx = pickle.load(f)
subset_a_sampler = SubsetRandomSampler(subset_a_idx)
subset_b_sampler = SubsetRandomSampler(subset_b_idx)
subset_c_sampler = SubsetRandomSampler(subset_c_idx)

subset_ac_sampler = SubsetRandomSampler(subset_a_idx+subset_c_idx)
subset_bc_sampler = SubsetRandomSampler(subset_b_idx+subset_c_idx)

subset_a_loader = DataLoader(train_dataset, batch_size=64, sampler=subset_a_sampler,
                                  drop_last=True, shuffle=False)
subset_b_loader = DataLoader(train_dataset, batch_size=64, sampler=subset_b_sampler,
                                  drop_last=True, shuffle=False)
subset_c_loader = DataLoader(train_dataset, batch_size=64, sampler=subset_c_sampler,
                                  drop_last=True, shuffle=False)
subset_ac_loader = DataLoader(train_dataset, batch_size=64, sampler=subset_ac_sampler,
                                  drop_last=True, shuffle=False)
subset_bc_loader = DataLoader(train_dataset, batch_size=64, sampler=subset_bc_sampler,
                                  drop_last=True, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model_a = models.resnet18()
model_a = model_a.to(device)
model_b = models.resnet18()
model_b = model_b.to(device)

model_ac = models.resnet18()
model_ac = model_ac.to(device)
model_bc = models.resnet18()
model_bc = model_bc.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3, weight_decay=5e-4)
optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3, weight_decay=5e-4)
optimizer_ac = torch.optim.Adam(model_ac.parameters(), lr=1e-3, weight_decay=5e-4)
optimizer_bc = torch.optim.Adam(model_bc.parameters(), lr=1e-3, weight_decay=5e-4)

if __name__ == "__main__":
    best_acc_a = 0
    best_acc_b = 0
    best_acc_ac = 0
    best_acc_bc = 0
    save_path = f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/init_final2/cifar10_500/cifar10_500_{id}/seed{seed}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    with open(save_path+'/subset_a.pkl', 'wb') as f:
        pickle.dump(subset_a_idx, f)
    with open(save_path+'/subset_b.pkl', 'wb') as f:
        pickle.dump(subset_b_idx, f)
    with open(save_path+'/subset_c.pkl', 'wb') as f:
        pickle.dump(subset_c_idx, f)
        
    for i in range(200):
        print('a------------------------------------------------')
        train(i, model_a, subset_a_loader, criterion, optimizer_a, device)
        if i > 100:
            best_acc_a = test(i, model_a, test_loader, criterion, save_path, 'a', best_acc_a, device)
        
        print('b------------------------------------------------')
        train(i, model_b, subset_b_loader, criterion, optimizer_b, device)
        if i > 100:
            best_acc_b = test(i, model_b, test_loader, criterion, save_path, 'b', best_acc_b, device)

        print('ac------------------------------------------------')
        train(i, model_ac, subset_ac_loader, criterion, optimizer_ac, device)
        if i > 100:
            best_acc_ac = test(i, model_ac, test_loader, criterion, save_path, 'ac', best_acc_ac, device)
        
        print('bc------------------------------------------------')
        train(i, model_bc, subset_bc_loader, criterion, optimizer_bc, device)
        if i > 100:
            best_acc_bc = test(i, model_bc, test_loader, criterion, save_path, 'bc', best_acc_bc, device)
    
    with open(save_path+f'/result.txt', 'a') as f:
        f.write(f"seed : {seed}, best_a : {best_acc_a}, best_b : {best_acc_b}, best_ac : {best_acc_ac}, best_bc : {best_acc_bc}\n")