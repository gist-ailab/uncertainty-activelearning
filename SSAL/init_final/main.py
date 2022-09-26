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

seed = 4
random.seed(seed)
torch.random.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# independent한 subset a,b,c에 있어서 a,b의 성능과 a+c, b+c의 성능 경향 비교 실험
# 1. a,b,c 나누기
# 2. a,b로 학습하기
# 3. a,b의 정확도 비교하기
# 4. a+c, b+c로 학습하기
# 5. a+c, b+c의 정확도 비교하기
# 6. 결과 분석

# 1. a,b,c로 나누기
train_dataset = datasets.CIFAR10('/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         transform=get_rand_augment('cifar10'))
test_dataset = datasets.CIFAR10('/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         train = False,
                                         transform=get_test_augment('cifar10'))

total_idx = [i for i in range(50000)]
random.shuffle(total_idx)

subset_a_idx = total_idx[:1000]
subset_b_idx = total_idx[1000:2000]
subset_c_idx = total_idx[2000:3000]
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
    save_path = f'/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/init_final/seed{seed}'
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
        train(i, model_a, subset_a_loader, criterion, optimizer_a)
        if i > 100:
            best_acc_a = test(i, model_a, test_loader, criterion, save_path, 'a', best_acc_a)
        
        print('b------------------------------------------------')
        train(i, model_b, subset_b_loader, criterion, optimizer_b)
        if i > 100:
            best_acc_b = test(i, model_b, test_loader, criterion, save_path, 'b', best_acc_b)

        print('ac------------------------------------------------')
        train(i, model_ac, subset_ac_loader, criterion, optimizer_ac)
        if i > 100:
            best_acc_ac = test(i, model_ac, test_loader, criterion, save_path, 'ac', best_acc_ac)
        
        print('bc------------------------------------------------')
        train(i, model_bc, subset_bc_loader, criterion, optimizer_bc)
        if i > 100:
            best_acc_bc = test(i, model_bc, test_loader, criterion, save_path, 'bc', best_acc_bc)
    
    with open(save_path+f'/result.txt', 'a') as f:
        f.write(f"seed : {seed}, best_a : {best_acc_a}, best_b : {best_acc_b}, best_ac : {best_acc_ac}, best_bc : {best_acc_bc}\n")