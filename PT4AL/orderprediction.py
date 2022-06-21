'''Train CIFAR10 with PyTorch.'''
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

from models import *
from loader import Loader, OrderPredictionLoader, General_Loader
from utils import progress_bar
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser(description='PyTorch pt4al Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

parameter_path = '/home/hinton/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/orderprediction'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize([144,144]),
    transforms.RandomCrop([128,128]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

trainset = OrderPredictionLoader(is_train=True, transform=transform_test, path='/home/hinton/NAS_AIlab_dataset/dataset/NIA_AIhub/herb_rotation')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = OrderPredictionLoader(is_train=False,  transform=transform_test, path='/home/hinton/NAS_AIlab_dataset/dataset/NIA_AIhub/herb_rotation')
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

print(next(iter(trainset))[0].shape)

# Model
print('==> Building model..')
net = ResNet18()
# print(net)
net.linear = nn.Linear(512, 2)
net = net.to(device)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

# Training
def train(epoch):
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


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(testloader):
            inputs, targets= inputs.to(device), targets.to(device)
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
    with open(parameter_path+'/best_orderprediction.txt','a') as f:
        f.write(str(acc)+':'+str(epoch)+'\n')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(parameter_path+'/checkpoint'):
            os.mkdir(parameter_path+'/checkpoint')
        # save rotation weights
        torch.save(state, parameter_path+'/checkpoint/orderprediction.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+121):
    train(epoch)
    if epoch%10==0:
        test(epoch)
    scheduler.step()