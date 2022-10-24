import os,sys
from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from resnet import *
import argparse
import pickle
import random
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./datas')
parser.add_argument('--save_path', type=str, default='./saves')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--epoch2', type=int, default=50)
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
parser.add_argument('--query_algorithm', type=str, choices=['high_gap', 'random', 'high_entropy'], default='high_gap')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()

#1. 1000개 랜덤하게 뽑아서 학습을 진행
#2. 학습된 모델을 이용하여 train에 속한지 아닌지를 확인하는 binary classification을 진행
#3. binary classification의 결과를 바탕으로 데이터를 선별(confidence? entropy?)
#4. 선별된 데이터를 바탕으로 다시 모델을 학습하고 #2로 이동

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

if __name__ == "__main__":
    for i in range(args.episode):
        if i==0:
            #1. 1000개 랜덤하게 뽑아서 학습을 진행
            if args.dataset == 'cifar10':
                idx = [i for i in range(50000)]
                random.shuffle(idx)
                lbl_idx = idx[:1000]
                ulbl_idx = idx[1000:]

            train_transform = utils.get_rand_augment(args.dataset)
            test_transform = utils.get_test_augment(args.dataset)
            loaders = dataset.DATALOADERS(lbl_idx, ulbl_idx, args.batch_size, train_transform, test_transform, args.dataset, args.data_path)
            lbl_loader, ulbl_loader, test_loader = loaders.get_loaders()
            binary_loader = dataset.BINARYLOADER(lbl_idx, ulbl_idx, args.batch_size, train_transform, args.dataset, args.data_path)
            
            base_model = ResNet18()
            main_fc = nn.Linear(512,10)
            binary_fc = nn.Linear(512,2)

            main_model = nn.Sequential(base_model, main_fc)
            binary_model = nn.Sequential(base_model, binary_fc)
            
            criterion = nn.CrossEntropyLoss()
            lbl_optimizer = torch.optim.Adam(main_model.parameters(), lr=1e-3, weight_decay=5e-4)
            lbl_scheduler = MultiStepLR(lbl_optimizer, milestones=[160])
            
            bn_optimzier = torch.optim.Adam(main_model.parameters(), lr=1e-3, weight_decay=5e-4)
            bn_scheduler = MultiStepLR(bn_optimzier, milestones=[40])
        else:
            #4. 선별된 데이터를 바탕으로 다시 모델을 학습하고 #2로 이동
            train_transform = utils.get_rand_augment(args.dataset)
            test_transform = utils.get_test_augment(args.dataset)
            loaders = dataset.DATALOADERS(lbl_idx, ulbl_idx, args.batch_size, train_transform, test_transform, args.dataset, args.data_path)
            lbl_loader, ulbl_loader, test_loader = loaders.get_loaders()
            binary_loader = dataset.BINARYLOADER(lbl_idx, ulbl_idx, args.batch_size, train_transform, args.dataset, args.data_path)
            
            base_model = ResNet18()
            main_fc = nn.Linear(512,10)
            binary_fc = nn.Linear(512,2)

            main_model = nn.Sequential(base_model, main_fc)
            binary_model = nn.Sequential(base_model, binary_fc)

        for j in range(args.epoch):
            utils.train(j, main_model, lbl_loader, criterion, lbl_optimizer, device)
            acc = utils.test(j, main_model, test_loader, criterion, args.save_path, args.dataset, device)
            print(f'epoch {j}. test accuracy : {acc}')
        
        utils.model_freeze(base_model)
        #2. 학습된 모델을 이용하여 train에 속한지 아닌지를 확인하는 binary classification을 진행
        for j in range(args.epoch2):
            binary_model(j, binary_model, binary_loader, criterion, bn_optimzier, device)    
        selected_idx = utils.domain_gap_train(binary_model, ulbl_loader, args.dataset, device, args.addendum)
        utils.model_unfreeze(base_model)
        
        
        