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
from torchvision import models

def std_convert(state_dict):
    state_dict2 = dict()
    for key1 in state_dict.keys():
        key1_list = key1.split('.')
        if(key1_list[0]=='features'):
            if len(key1_list) == 3:
                if key1_list[1] == '0':
                    key2 = 'conv1.weight'
                if key1_list[1] == '1':
                    key2 = 'bn1.'+key1_list[-1]
            elif len(key1_list) >= 5:
                key2 = 'layer'+f'{int(key1_list[1])-3}.'+'.'.join(key1_list[2:])
        state_dict2[key2] = state_dict[key1]
    return state_dict2

def get_rand_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 96
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size, padding=int(size*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform

def get_test_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform

def get_high_sim(loss_list, k=1000):
    return [data[1] for data in loss_list][:k]

def get_low_sim(loss_list, k=1000):
    return [data[1] for data in loss_list][:k]

def get_balanced_sim(loss_list, k=1000):
    n = len(loss_list)//k
    return [loss_list[i][1] for i in range(0,len(loss_list),n)]

def get_step_inc(loss_list, k=1000):
    step = 10
    sampled_list = []
    n = k//step
    for i in range(step//2):
        sampled_list += loss_list[i*n:(i+1)*n][:(n+10*step//2-i*step)]
        sampled_list += loss_list[(step-i)*n:(step-1-i)*n:-1]
    return sampled_list

def get_step_dec(loss_list, k=1000):
    step = 10
    sampled_list = []
    n = k//step
    for i in range(step//2):
        sampled_list += loss_list[i*n:(i+1)*n][:(n-10*step//2+i*step)]
        sampled_list += loss_list[(step-i)*n:(step-1-i)*n:-1][:(n+10*step//2-i*step)]
    return sampled_list

def get_random(loss_list, k=1000, seed=1):
    import copy, random
    random.seed(seed)
    loss_list_copy = copy.deepcopy(loss_list)
    random.shuffle(loss_list_copy)
    return loss_list_copy[:k]

# def get_imbalanced():