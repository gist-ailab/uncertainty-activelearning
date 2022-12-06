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

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/data')
# parser.add_argument('--data_path', type=str, default='/SSDb/Workspace/hyj/cifar10')
parser.add_argument('--save_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/single_stage_confidence')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
parser.add_argument('--query_algorithm', type=str, choices=['high_unseen', 'low_conf', 'high_entropy', 'random'], default='low_conf')
parser.add_argument('--addendum', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lbl_smoothing', type=int, default=0.0)
parser.add_argument('--load', type=int, default=0)

args = parser.parse_args()


if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
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

if __name__ == "__main__":
    for i in range(args.load, args.episode):
        curr_path = os.path.join(save_path, f'episode{i}')
        if not os.path.isdir(curr_path):
            os.mkdir(curr_path)
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
            
        else:
            if os.path.exists(curr_path+'/lbl_idx.pkl') and os.path.exists(curr_path+'/ulbl_idx.pkl'):
                lbl_idx = pickle.load(open(curr_path+'/lbl_idx.pkl', 'rb'))
                ulbl_idx = pickle.load(open(curr_path+'/ulbl_idx.pkl', 'rb'))
            train_transform = utils.get_rand_augment(args.dataset)
            test_transform = utils.get_test_augment(args.dataset)
            loaders = dataset.DATALOADERS(lbl_idx, ulbl_idx, args.batch_size, train_transform, test_transform, args.dataset, args.data_path)
            lbl_loader, ulbl_loader, test_loader = loaders.get_loaders()
            
        main_model = ResNet18()
        main_model = main_model.to(device)
        
        main_criterion = nn.CrossEntropyLoss()
        main_optimizer = torch.optim.SGD(main_model.parameters(), lr=1e-1, weight_decay=5e-4)
        main_scheduler = MultiStepLR(main_optimizer, milestones=[160])

        with open(curr_path+'/lbl_idx.pkl', 'wb') as f:
            pickle.dump(lbl_idx, f)
        with open(curr_path+'/ulbl_idx.pkl', 'wb') as f:
            pickle.dump(ulbl_idx, f)
        
        print('main classification -------------------------------------------------------')
        best_acc = 0
        for j in range(args.epoch):
            # utils.train(j, main_model, lbl_loader, main_criterion, main_optimizer, device)
            # acc = utils.test(j, main_model, test_loader, main_criterion, curr_path, args.dataset, device, best_acc)
            utils.train(j, main_model, lbl_loader, main_criterion, main_optimizer, device)
            acc = utils.test(j, main_model, test_loader, main_criterion, curr_path, args.dataset, device, best_acc)
            if acc > best_acc: best_acc = acc
        with open(save_path+'/total_acc.txt', 'a') as f:
            f.write(f'seed : {args.seed}, episode : {i}, acc : {best_acc}\n')
        
        model_paths = []
        curr_model_path = os.path.join(curr_path, args.dataset,'model.pt')
        model_paths.append(curr_model_path)
        
        selected_ulb_idx = utils.query_algorithm(main_model, main_criterion, ulbl_loader, ulbl_idx, device, model_paths, args.addendum)
        lbl_idx = np.array(lbl_idx)
        ulbl_idx = np.array(ulbl_idx)
        
        selected_idx = ulbl_idx[selected_ulb_idx]
        lbl_idx = np.concatenate((lbl_idx, selected_idx))
        ulbl_idx = np.delete(ulbl_idx, selected_ulb_idx)