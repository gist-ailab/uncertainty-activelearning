import os, sys
from sched import scheduler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.ops import MLP
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.ops import MLP
from tqdm import tqdm
from resnet import *
import argparse
import pickle
import random
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/data')
parser.add_argument('--save_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/domian_divergence')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--epoch2', type=int, default=120)
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
parser.add_argument('--query_algorithm', type=str, choices=['kld', 'pad', 'jensen', 'low_conf', 'high_entropy', 'random'], default='jensen')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=256)

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
if not args.seed==None:
    save_path = os.path.join(args.save_path, f'seed{args.seed}',args.query_algorithm)
else:
    save_path = os.path.join(args.save_path, 'current', args.query_alogrithm)
if not os.path.isdir(save_path):
    os.mkdir(save_path)

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
            binary_loader, query_loader = dataset.BINARYLOADER(lbl_idx, ulbl_idx, args.batch_size, train_transform, args.dataset, args.data_path)
            
        else:
            #4. 선별된 데이터를 바탕으로 다시 모델을 학습하고 #2로 이동
            train_transform = utils.get_rand_augment(args.dataset)
            test_transform = utils.get_test_augment(args.dataset)
            loaders = dataset.DATALOADERS(lbl_idx, ulbl_idx, args.batch_size, train_transform, test_transform, args.dataset, args.data_path)
            lbl_loader, ulbl_loader, test_loader = loaders.get_loaders()
            binary_loader, query_loader = dataset.BINARYLOADER(lbl_idx, ulbl_idx, args.batch_size, train_transform, args.dataset, args.data_path)
            
        base_model = ResNet18()
        main_fc = nn.Linear(512,10)
        binary_fc = nn.Linear(512,2)

        main_model = nn.Sequential(base_model, main_fc)
        main_model = main_model.to(device)
        
        # binary_model = nn.Sequential(base_model, binary_fc)
        binary_model = MLP(in_channels=3*32*32, hidden_channels=[256,256,256,256,256,256,2], norm_layer=torch.nn.BatchNorm1d, inplace=False)
        binary_model = binary_model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        lbl_optimizer = torch.optim.Adam(main_model.parameters(), lr=1e-3, weight_decay=5e-4)
        lbl_scheduler = MultiStepLR(lbl_optimizer, milestones=[160])
        
        bn_optimzier = torch.optim.Adam(binary_model.parameters(), lr=1e-2, weight_decay=5e-3)
        bn_scheduler = MultiStepLR(bn_optimzier, milestones=[40,80])
            
        curr_path = os.path.join(save_path, f'episode{i}')
        if not os.path.isdir(curr_path):
            os.mkdir(curr_path)

        with open(curr_path+'/lbl_idx.pkl', 'wb') as f:
            pickle.dump(lbl_idx, f)
        with open(curr_path+'/ulbl_idx.pkl', 'wb') as f:
            pickle.dump(ulbl_idx, f)
        
        print('main classification -------------------------------------------------------')
        best_acc = 0
        for j in range(args.epoch):
            utils.train(j, main_model, lbl_loader, criterion, lbl_optimizer, device)
            acc = utils.test(j, main_model, test_loader, criterion, curr_path, args.dataset, device, best_acc)
        if acc > best_acc: best_acc = acc
        with open(save_path+'/total_acc.txt', 'a') as f:
            f.write(f'seed : {args.seed}, episode : {i}, acc : {best_acc}\n')
            
        #2. 학습된 모델을 이용하여 train에 속한지 아닌지를 확인하는 binary classification을 진행
        if not (i == args.episode-1):
            print('binary classification -------------------------------------------------------')
            if args.query_algorithm == 'pad' or args.query_algorithm == 'kld' or args.query_algorithm == 'jensen':
                # utils.model_freeze(base_model)
                for j in range(args.epoch2):
                    utils.binary_train(j, binary_model, binary_loader, criterion, bn_optimzier, device)
                # utils.model_unfreeze(base_model)
            #3. binary classification의 결과를 바탕으로 데이터를 선별(confidence? entropy?)
            if args.query_algorithm == 'kld' or args.query_algorithm == 'jensen':
                query_criterion = nn.KLDivLoss(reduction='none')
            else:
                query_criterion = nn.CrossEntropyLoss()
            selected_ulb_idx = utils.domain_gap_prediction(binary_model, query_criterion, query_loader, ulbl_idx, args.query_algorithm, device, args.addendum)
            
            lbl_idx = np.array(lbl_idx)
            ulbl_idx = np.array(ulbl_idx)
            
            print(selected_ulb_idx[:3])
            selected_idx = ulbl_idx[selected_ulb_idx]
            print(selected_idx[:3])
            lbl_idx = np.concatenate((lbl_idx, selected_idx))
            ulbl_idx = np.delete(ulbl_idx, selected_ulb_idx)
        