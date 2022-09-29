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

seed = 1
random.seed(seed)
torch.random.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/QA_comparison/seed{seed}'

train_dataset = datasets.CIFAR10('/home/bengio/ailab_mat/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         transform=get_rand_augment('cifar10'))

sign = 'random'
episode = 4
with open(os.path.join(save_path, f'episode{episode-1}', f'{sign}_lbl_set.pkl'), 'rb') as f:
    pre_subset_idx = pickle.load(f)
    
with open(os.path.join(save_path, f'episode{episode}', f'{sign}_lbl_set.pkl'), 'rb') as f:
    subset_idx = pickle.load(f)

selected_data_idx = [data for data in subset_idx if not data in pre_subset_idx]
subset_sampler = SubsetRandomSampler(selected_data_idx)

subset_loader = DataLoader(train_dataset, batch_size=64, sampler=subset_sampler,
                                  drop_last=True, shuffle=False)

model_a = models.resnet18()
model_a = model_a.to(device)

criterion = nn.CrossEntropyLoss()
state_dict_a = torch.load(os.path.join(save_path, f'episode{episode-1}', sign, 'model.pt'))
model_a.load_state_dict(state_dict_a)

if __name__ == "__main__":
    best_acc_a = test_eval(0, model_a, subset_loader, criterion, save_path, sign, 0)
    print(best_acc_a)