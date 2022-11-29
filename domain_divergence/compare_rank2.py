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
from copy import deepcopy

# label model5을 한 것과 안한 것에서 data 간의 rank가 바뀌는지 확인
# label model5 안 한 것과 label model5 = 0.10에서 각각의 confidence의 rank를 출력해 plot해서 관계가 있는지 확인

parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/data')
parser.add_argument('--data_path', type=str, default='/SSDb/Workspaces/yunjae.heo/cifar10')
parser.add_argument('--save_path', type=str, default='/ailab_mat/personal/heo_yunjae/Parameters/Uncertainty/domian_divergence/callibration_test')
parser.add_argument('--gpu', type=str, default='6')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, choices=['cifar10', 'stl10'], default='cifar10')
# parser.add_argument('--query_algorithm', type=str, choices=['high_unseen', 'low_conf', 'high_entropy', 'random'], default='high_entropy')
parser.add_argument('--query_algorithm', type=str, choices=['high_unseen', 'low_conf', 'high_entropy', 'random'], default='low_conf')
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()

if not args.seed==None:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10(args.data_path, download=True, transform=utils.get_rand_augment('cifar10'))
    subset_idx = [random.randint(0, 50000) for i in range(10000)]
    train_subset = Subset(train_dataset, subset_idx)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    
    test_dataset = datasets.CIFAR10(args.data_path, download=True, transform=utils.get_test_augment(args.dataset), train=False)
    test_subset = Subset(test_dataset, [random.randint(0,10000) for i in range(1000)])
test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

model5 = ResNet18()
model5 = model5.to(device)
criterion1 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model5.parameters(), lr=1e-2, weight_decay=5e-4)

model6 = ResNet18()
model6 = model6.to(device)
criterion2 = nn.CrossEntropyLoss()
optimizer2 = torch.optim.SGD(model6.parameters(), lr=1e-2, weight_decay=5e-4)

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

if not os.path.isfile(args.save_path+'/model5.pt'):
    best_acc1 = 0
    for i in range(200):
        train(i, model5, train_loader, criterion1, optimizer1, device)
        if i > 160:
            acc = test(i, model5, test_loader, criterion1, device)
            if best_acc1 < acc:
                model_para1 = deepcopy(model5.state_dict())
                best_acc1 = acc
    torch.save(model_para1, args.save_path+'/model5.pt')

if not os.path.isfile(args.save_path+'/model6.pt'):
    best_acc2 = 0
    for i in range(200):
        train(i, model6, train_loader, criterion2, optimizer2, device)
        if i > 160:
            acc = test(i, model6, test_loader, criterion2, device)
            if best_acc2 < acc:
                model_para2 = deepcopy(model6.state_dict())
                best_acc2 = acc
    torch.save(model_para2, args.save_path+'/model6.pt')

model_para1 = torch.load(args.save_path+'/model5.pt')
model_para2 = torch.load(args.save_path+'/model6.pt')

model5.load_state_dict(model_para1)
model6.load_state_dict(model_para2)

test_loader = tqdm(test_loader)
model5_list = torch.tensor([], dtype=float).to(device)
for batch_idx, (inputs, targets) in enumerate(test_loader):
    # inputs, _ = inputs.to(device), targets.to(device)
    # st_output = model5(inputs)
    # st_max_conf = torch.max(F.softmax(st_output, dim=1, dtype=float),dim=1)
    # model5_list = torch.cat((model5_list,st_max_conf.values),0)
    inputs = inputs.to(device)
    outputs = model5(inputs)
    outputs = F.softmax(outputs, dim=1)
    entropy = -outputs*outputs.log()
    entropy = entropy.sum(dim=1)
    model5_list = torch.cat((model5_list,entropy),0)
model5_list = model5_list.detach().cpu().numpy()

test_loader = tqdm(test_loader)
model6_list = torch.tensor([], dtype=float).to(device)
for batch_idx, (inputs, targets) in enumerate(test_loader):
    # inputs, _ = inputs.to(device), targets.to(device)
    # st_output = model6(inputs)
    # st_max_conf = torch.max(F.softmax(st_output, dim=1, dtype=float),dim=1)
    # model6_list = torch.cat((model6_list,st_max_conf.values),0)
    inputs = inputs.to(device)
    outputs = model6(inputs)
    outputs = F.softmax(outputs, dim=1)
    entropy = -outputs*outputs.log()
    entropy = entropy.sum(dim=1)
    model6_list = torch.cat((model6_list,entropy),0)
model6_list = model6_list.detach().cpu().numpy()

plt.scatter(model5_list, model6_list)
plt.savefig(args.save_path+'/compare_rank2_entropy.jpg')

model5_arg = np.argsort(model5_list)
model6_arg = np.argsort(model6_list)

model5_argpart = model5_arg[:100]
model6_argpart = model6_arg[:100]
# print(model5_list)

argsum = np.concatenate((model5_argpart, model6_argpart))
print(len(argsum) - len(np.unique(argsum)))