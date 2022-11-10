#calibration plot
#input :  txt file (img idx, label, prediction, max_confidence)
#output : calibration plot img file

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
from ECE import ece_score

parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/data')
parser.add_argument('--data_path', type=str, default='/SSDb/Workspaces/yunjae.heo/cifar10')
parser.add_argument('--save_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/domian_divergence/ls20')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
parser.add_argument('--query_algorithm', type=str, choices=['high_unseen', 'low_conf', 'high_entropy', 'random'], default='high_entropy')
# parser.add_argument('--query_algorithm', type=str, choices=['high_unseen', 'low_conf', 'high_entropy', 'random'], default='low_conf')
parser.add_argument('--batch_size', type=int, default=256)

args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

curr_path = os.path.join(args.save_path, f'seed{args.seed}', args.query_algorithm, f'episode{7}')

if args.dataset == 'cifar10':
    test_dataset = datasets.CIFAR10(args.data_path, download=True, transform=utils.get_test_augment(args.dataset), train=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

main_model = ResNet18()
main_model = main_model.to(device)
query_para = torch.load(os.path.join(curr_path, args.dataset,'query_model.pt'))
main_model.load_state_dict(query_para)

test_loader = tqdm(test_loader)
ece = 0
for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs, targets = inputs.to(device), targets.numpy()
    outputs = main_model(inputs)
    outputs = outputs.cpu().detach().numpy()
    ece += ece_score(outputs, targets, n_bins=10)
print(ece)