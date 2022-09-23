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
from utils import *
from torch.utils.data.sampler import SubsetRandomSampler

# Initial accuracy를 올릴 수 있는 init sampling query에 대한 연구

torch.random.manual_seed(1)

loss_list_path = ''
init_path = None
save_path = '/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/SSAL/batches'

with open(loss_list_path, 'rb') as f:
    loss_list = pickle.load(f)

train_idx = get_high_sim(loss_list)

train_dataset = datasets.CIFAR10('/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         transform=get_rand_augment('cifar10'))
train_sampler = SubsetRandomSampler(train_idx)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler,
                                  drop_last=True, shuffle=True)

test_dataset = datasets.CIFAR10('/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         train = False,
                                         transform=get_test_augment('cifar10'))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = models.models.resnet18()
if init_path != None:
    checkpoint = torch.load(init_path)
    checkpoint = std_convert(checkpoint)
    model.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

