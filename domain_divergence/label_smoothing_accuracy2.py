import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
# import torchvision.models as models
from tqdm import tqdm
from resnet import *
import argparse
import pickle
import random
import dataset
import utils
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from ECE import *

parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/data')
parser.add_argument('--data_path', type=str, default='/SSDb/Workspaces/yunjae.heo/cifar10')
parser.add_argument('--save_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/domian_divergence/callibration_test/')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--epoch2', type=int, default=200)
parser.add_argument('--episode', type=int, default=9)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
parser.add_argument('--query_algorithm', type=str, choices=['high_unseen', 'low_conf', 'high_entropy', 'random'], default='high_entropy')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=256)

args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
episode = args.episode
if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)
if not args.seed==None:
    save_path = os.path.join(args.save_path, f'seed{args.seed}')
else:
    save_path = os.path.join(args.save_path, 'current')
if not os.path.isdir(save_path):
    os.mkdir(save_path)
save_path = os.path.join(save_path,args.query_algorithm)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

train_dataset = datasets.CIFAR10(args.data_path, download=True, transform=utils.get_rand_augment('cifar10'))
subset_idx = [random.randint(0, 50000) for i in range(10000)]
train_subset = Subset(train_dataset, subset_idx)
train_loader = DataLoader(train_subset, batch_size=args.batch_size, drop_last=True, shuffle=True)

test_dataset = datasets.CIFAR10(args.data_path, download=True, transform=utils.get_test_augment('cifar10'), train=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

ns_model = ResNet18()
ns_model = ns_model.to(device)
ns_criterion = nn.CrossEntropyLoss()
ns_optimizer = torch.optim.SGD(ns_model.parameters(), lr=1e-2, weight_decay=5e-4)

ls_model = ResNet18()
ls_model = ls_model.to(device)
ls_criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
ls_optimizer = torch.optim.SGD(ls_model.parameters(), lr=1e-2, weight_decay=5e-4)

def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)
    print(f'epoch : {epoch} _________________________________________________')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({'loss':train_loss/len(train_loader), 'acc':100*correct/total})
        
def test(epoch, model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'loss':test_loss/len(test_loader), 'acc':100*correct/total})
        acc = 100*correct/total
    return acc

ns_best_acc = 0
for i in range(200):
    train(i, ns_model, train_loader, ns_criterion, ns_optimizer, device)
    if i > 160:
        acc = test(i, ns_model, test_loader, ns_criterion, device)
        if ns_best_acc < acc:
            model_para = ns_model.state_dict()
            ns_best_acc = acc

ls_best_acc = 0
for i in range(200):
    train(i, ls_model, train_loader, ls_criterion, ls_optimizer, device)
    if i > 160:
        acc = test(i, ls_model, test_loader, ls_criterion, device)
        if ls_best_acc < acc:
            model_para = ls_model.state_dict()
            ls_best_acc = acc

print(f'ns best acc : {ns_best_acc}')
print(f'ls best acc : {ls_best_acc}')