import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle
from torchvision import datasets
import torchvision.models as models
from utils import *
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import random
import argparse

parser = argparse.ArgumentParser(description='Active Learning')
parser.add_argument('--data_path', type=str, default='./datas')
parser.add_argument('--save_path', type=str, default='./saves')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
parser.add_argument('--query_algorithm', type=str, choices=['high_conf', 'low_conf', 'balance', 'random', 'high_entropy'], default='low_conf')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
episode = args.episode
if not os.path.isdir(args.data_path):
    os.mkdir(args.data_path)
if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)

if args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10(args.data_path, 
                                download=True,
                                transform=get_rand_augment('cifar10'))
    test_dataset = datasets.CIFAR10(args.data_path, 
                                    download=True,
                                    train = False,
                                    transform=get_test_augment('cifar10'))
    num_class = 10
if args.dataset == 'stl10':
    train_dataset = datasets.stl10(args.data_path, 
                                download=True,
                                transform=get_rand_augment('stl10'))
    test_dataset = datasets.stl10(args.data_path, 
                                    download=True,
                                    train = False,
                                    transform=get_test_augment('stl10'))
    num_class = 10

data_length = len(train_dataset)
total_idx = [i for i in range(data_length)]
random.shuffle(total_idx)
subset_idx = total_idx[:args.addendum]
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if __name__ == "__main__":
    best_acc = 0
    save_path = os.path.join(args.save_path, f'seed{args.seed}')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    lbl_set = subset_idx
    
    for epi in range(episode):
        print(f'episode : {epi}------------------------------------------------')
        curr_path = os.path.join(save_path, f'episode{epi}')
        if not os.path.isdir(curr_path):
            os.mkdir(curr_path)
        with open(curr_path+'/lbl_set.pkl', 'wb') as f:
            pickle.dump(lbl_set, f)
        
        train_sampler = SubsetRandomSampler(lbl_set)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                    drop_last=True, shuffle=False)
        
        model = models.resnet18()
        model.fc = nn.Linear(512, num_class)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        
        for i in range(args.epoch):
            train(i, model, train_loader, criterion, optimizer, device)
            acc = test(i, model, test_loader, criterion, curr_path, args.query_algorithm, device)
        
        ulbl_set = [i for i in range(data_length) if i not in lbl_set]

        model_para = torch.load(os.path.join(curr_path, args.query_algorithm, 'model.pt'))
        model.load_state_dict(model_para)
        
        ulbl_subset = Subset(train_dataset, ulbl_set)
        ulbl_sampler = SequentialSampler(ulbl_subset)
        ulbl_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=ulbl_sampler, shuffle=False)
        lbl_set = lbl_set + uldl_idx[query_algorithm(model, ulbl_loader, ulbl_set, args.query_algorithm, device, args.addendum)]
    
        with open(curr_path+f'/result.txt', 'a') as f:
            f.write(f"seed : {args.seed}, {args.query_algorithm} : {acc}\n")