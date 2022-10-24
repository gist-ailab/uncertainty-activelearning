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

id = 0
seed = 4
data_amount = 5000
random.seed(seed)
torch.random.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/init_final2/cifar10_{data_amount}/cifar10_{data_amount}_{id}/seed{seed}'

train_dataset = datasets.CIFAR10('/ailab_mat/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         transform=get_rand_augment('cifar10'))


with open(os.path.join(save_path, 'subset_c.pkl'), 'rb') as f:
    subset_c_idx = pickle.load(f)
subset_c_sampler = SubsetRandomSampler(subset_c_idx)
subset_c_loader = DataLoader(train_dataset, batch_size=64, sampler=subset_c_sampler,
                                  drop_last=True, shuffle=False)

model_a = models.resnet18()
model_a = model_a.to(device)
model_b = models.resnet18()
model_b = model_b.to(device)

criterion = nn.CrossEntropyLoss()
state_dict_a = torch.load(os.path.join(save_path, 'a', 'model.pt'))
model_a.load_state_dict(state_dict_a)
state_dict_b = torch.load(os.path.join(save_path, 'b', 'model.pt'))
model_b.load_state_dict(state_dict_b)

if __name__ == "__main__":
    best_acc_a = test_eval(0, model_a, subset_c_loader, criterion, save_path, 'a', 0, device)
    best_acc_b = test_eval(0, model_b, subset_c_loader, criterion, save_path, 'b', 0, device)
    print(best_acc_a,best_acc_b)