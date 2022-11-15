import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from resnet import *
import argparse
import pickle
import random
import dataset
import utils
import csv
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
from ECE import ece_score, oce_score
import matplotlib.pyplot as plt
from torch.utils.data import Subset, Dataset
import random

# label smoothing을 한 것과 안한 것에서 data 간의 rank가 바뀌는지 확인
# label smoothing 안 한 것과 label smoothing = 0.10에서 각각의 confidence의 rank를 출력해 plot해서 관계가 있는지 확인

parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/data')
parser.add_argument('--data_path', type=str, default='/SSDb/Workspaces/yunjae.heo/cifar10')
parser.add_argument('--save_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/domian_divergence/ls20')
parser.add_argument('--gpu', type=str, default='6')
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
# parser.add_argument('--query_algorithm', type=str, choices=['high_unseen', 'low_conf', 'high_entropy', 'random'], default='high_entropy')
parser.add_argument('--query_algorithm', type=str, choices=['high_unseen', 'low_conf', 'high_entropy', 'random'], default='low_conf')
parser.add_argument('--batch_size', type=int, default=32)

args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

curr_path = os.path.join(args.save_path, f'seed{args.seed}', args.query_algorithm, f'episode{1}')

if args.dataset == 'cifar10':
    test_dataset = datasets.CIFAR10(args.data_path, download=True, transform=utils.get_test_augment(args.dataset), train=False)
    test_subset = Subset(test_dataset, [random.randint(0,10000) for i in range(1000)])
test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

non_smoothing_model = ResNet18()
non_smoothing_model = non_smoothing_model.to(device)
non_smoothing_para = torch.load(os.path.join(curr_path, args.dataset,'model.pt'))
non_smoothing_model.load_state_dict(non_smoothing_para)

test_loader = tqdm(test_loader)
non_smoothing_list = torch.tensor([], dtype=float).to(device)
for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs, _ = inputs.to(device), targets.to(device)
    ns_output = non_smoothing_model(inputs)
    ns_max_conf = torch.max(F.softmax(ns_output, dim=1, dtype=float),dim=1)
    non_smoothing_list = torch.cat((non_smoothing_list,ns_max_conf.values),0)
non_smoothing_list = non_smoothing_list.detach().cpu().numpy()

smoothing_model = ResNet18()
moothing_model = smoothing_model.to(device)
smoothing_para = torch.load(os.path.join(curr_path, args.dataset,'query_model.pt'))
smoothing_model.load_state_dict(smoothing_para)

test_loader = tqdm(test_loader)
smoothing_list = torch.tensor([], dtype=float).to(device)
for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs, _ = inputs.to(device), targets.to(device)
    st_output = smoothing_model(inputs)
    st_max_conf = torch.max(F.softmax(st_output, dim=1, dtype=float),dim=1)
    smoothing_list = torch.cat((smoothing_list,st_max_conf.values),0)
smoothing_list = smoothing_list.detach().cpu().numpy()

# non_smoothing_list = np.array(non_smoothing_list)
# non_smoothing_list = np.argsort(non_smoothing_list)

# smoothing_list = np.array(smoothing_list)
# smoothing_list = np.argsort(smoothing_list)

plt.scatter(non_smoothing_list, smoothing_list)
plt.savefig(args.save_path+'/compare_rank.jpg')

smoothing_dict = dict()
non_smoothing_dict = dict()
for i in range(0,10):
    smoothing_dict[i] = 0
    non_smoothing_dict[i] = 0

for value in smoothing_list:
    smoothing_dict[int(10*value)] += 1
print(smoothing_dict)

for value in non_smoothing_list:
    non_smoothing_dict[int(10*value)] += 1
print(non_smoothing_dict)

