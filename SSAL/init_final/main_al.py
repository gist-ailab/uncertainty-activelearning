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
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import random

seed = 2
random.seed(seed)
torch.random.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
episode = 10

# random과 least conf와 most conf의 episode에 따른 성능 비교 실험
# 1. init 뽑기
# 2. init으로 학습하기
# 3. 각각의 query 알고리즘에 따라 학습 진행하기
# 4. query 알고리즘 별 episode 마다의 성능 비교하기

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
subset_a_idx = total_idx[:1000]
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if __name__ == "__main__":
    best_acc_random = 0
    best_acc_lconf = 0
    best_acc_mconf = 0
    save_path = f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/QA_comparison/seed{seed}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    random_lbl_set = subset_a_idx
    lconf_lbl_set = subset_a_idx
    mconf_lbl_set = subset_a_idx
    
    for epi in range(episode):
        print(f'episode : {epi}------------------------------------------------')
        curr_path = os.path.join(save_path, f'episode{epi}')
        if not os.path.isdir(curr_path):
            os.mkdir(curr_path)
        with open(curr_path+'/random_lbl_set.pkl', 'wb') as f:
            pickle.dump(random_lbl_set, f)
        with open(curr_path+'/lconf_lbl_set.pkl', 'wb') as f:
            pickle.dump(lconf_lbl_set, f)
        with open(curr_path+'/mconf_lbl_set.pkl', 'wb') as f:
            pickle.dump(mconf_lbl_set, f)
        
        random_sampler = SubsetRandomSampler(random_lbl_set)
        random_loader = DataLoader(train_dataset, batch_size=64, sampler=random_sampler,
                                    drop_last=True, shuffle=False)
        lconf_sampler = SubsetRandomSampler(lconf_lbl_set)
        lconf_loader = DataLoader(train_dataset, batch_size=64, sampler=lconf_sampler,
                                    drop_last=True, shuffle=False)
        mconf_sampler = SubsetRandomSampler(mconf_lbl_set)
        mconf_loader = DataLoader(train_dataset, batch_size=64, sampler=mconf_sampler,
                                    drop_last=True, shuffle=False)
        
        model_random = models.resnet18()
        model_random = model_random.to(device)

        model_lconf = models.resnet18()
        model_lconf = model_lconf.to(device)

        model_mconf = models.resnet18()
        model_mconf = model_mconf.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer_random = torch.optim.Adam(model_random.parameters(), lr=1e-3, weight_decay=5e-4)
        optimizer_lconf = torch.optim.Adam(model_lconf.parameters(), lr=1e-3, weight_decay=5e-4)
        optimizer_mconf = torch.optim.Adam(model_mconf.parameters(), lr=1e-3, weight_decay=5e-4)
        
        for i in range(200):
            print('random------------------------------------------------')
            train(i, model_random, random_loader, criterion, optimizer_random)
            if i > 100:
                best_acc_random = test(i, model_random, test_loader, criterion, curr_path, 'random', best_acc_random)
            print('lconf------------------------------------------------')
            train(i, model_lconf, lconf_loader, criterion, optimizer_lconf)
            if i > 100:
                best_acc_lconf = test(i, model_lconf, test_loader, criterion, curr_path, 'lconf', best_acc_lconf)
            print('mconf------------------------------------------------')
            train(i, model_mconf, mconf_loader, criterion, optimizer_mconf)
            if i > 100:
                best_acc_mconf = test(i, model_mconf, test_loader, criterion, curr_path, 'mconf', best_acc_mconf)
        
        random_ulbl_set = [i for i in range(50000) if i not in random_lbl_set]
        lconf_ulbl_set = [i for i in range(50000) if i not in lconf_lbl_set]
        mconf_ulbl_set = [i for i in range(50000) if i not in mconf_lbl_set]

        random_model_para = torch.load(os.path.join(curr_path, 'random', 'model.pt'))
        model_random.load_state_dict(random_model_para)
        lconf_model_para = torch.load(os.path.join(curr_path, 'lconf', 'model.pt'))
        model_lconf.load_state_dict(lconf_model_para)
        mconf_model_para = torch.load(os.path.join(curr_path, 'mconf', 'model.pt'))
        model_mconf.load_state_dict(mconf_model_para)
        
        #random
        random.shuffle(random_ulbl_set)
        random_lbl_set = random_lbl_set + random_ulbl_set[:1000]
        #lconf
        lconf_ulbl_subset = Subset(train_dataset, lconf_ulbl_set)
        lconf_ulbl_sampler = SequentialSampler(lconf_ulbl_subset)
        lconf_ulbl_loader = DataLoader(train_dataset, batch_size=1, sampler=lconf_ulbl_sampler, shuffle=False)
        lconf_lbl_set = lconf_lbl_set + query_algorithm(model_lconf, lconf_ulbl_loader, lconf_ulbl_set, 'lconf')
        #mconf
        mconf_ulbl_subset = Subset(train_dataset, mconf_ulbl_set)
        mconf_ulbl_sampler = SequentialSampler(mconf_ulbl_subset)
        mconf_ulbl_loader = DataLoader(train_dataset, batch_size=1, sampler=mconf_ulbl_sampler, shuffle=False)
        mconf_lbl_set = mconf_lbl_set + query_algorithm(model_mconf, mconf_ulbl_loader, mconf_ulbl_set, 'mconf')
    
        with open(curr_path+f'/result.txt', 'a') as f:
            f.write(f"seed : {seed}, best_random : {best_acc_random}, best_lconf : {best_acc_lconf}, best_mconf : {best_acc_mconf}\n")