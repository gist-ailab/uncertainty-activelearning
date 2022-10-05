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

seed = 0
random.seed(seed)
torch.random.manual_seed(seed)

total_idx = [i for i in range(50000)]
random.shuffle(total_idx)

subset_a_idx = total_idx[:5000]
subset_b_idx = total_idx[5000:10000]
subset_c_idx = total_idx[10000:15000]
subset_a_sampler = SubsetRandomSampler(subset_a_idx)
subset_b_sampler = SubsetRandomSampler(subset_b_idx)
subset_c_sampler = SubsetRandomSampler(subset_c_idx)

save_path = f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/init_final2/subsets/cifar100_10/'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
    
save_path = f'/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/init_final2/subsets/cifar100_10/seed{seed}'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(save_path+'/subset_a.pkl', 'wb') as f:
    pickle.dump(subset_a_idx, f)
with open(save_path+'/subset_b.pkl', 'wb') as f:
    pickle.dump(subset_b_idx, f)
with open(save_path+'/subset_c.pkl', 'wb') as f:
    pickle.dump(subset_c_idx, f)