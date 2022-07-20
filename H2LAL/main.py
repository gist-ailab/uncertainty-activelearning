import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
import numpy as np
import glob

# from models import *
from models.resnet_128 import *
from loader import Loader, Loader2, General_Loader
from util import progress_bar

os.environ["CUDA_VISIBLE_DEVICES"]='4'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

server_name = 'hinton'
parameter_path = f'/home/{server_name}/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/pt4al/cifar10/rotation'
data_path = f'/home/{server_name}/NAS_AIlab_dataset/dataset/cifar10'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
           'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

testset = General_Loader(is_train=False,  transform=transform_test, name_dict=classes, path=data_path)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=32)

print('==> Building model..')
net = ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Training
def train(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(net, criterion, epoch, cycle):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, parameter_path + f'/checkpoint/main_{cycle}.pth')
        best_acc = acc

# make unlabeled set, UNLABEL = [paths]
def make_unlabeled(data_path):
    img_path = glob.glob(data_path+'/train/*/*')
    return img_path

# make sorted confidence list, Unlabel -> Labeled, conf : [(conf, path)]
def make_conflist(net, unlabel_loader):
    confidence_list = []
    with torch.no_grad():
        for batch_idx, (inputs, _, path) in enumerate(unlabel_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            confidence = torch.max(F.softmax(outputs, dim=-1))
            confidence_list.append((confidence, path))
    confidence_list.sort(key=lambda x:x[0])
    return confidence_list

# Select K datas for Active Learning
def k_selection(conflist, k=1000):
    pass

