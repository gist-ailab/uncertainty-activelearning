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

# from models import *
from models.resnet_128 import *
from loader import Loader, OrderPredictionLoader, General_Loader_withpath
from utils import progress_bar
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]='6'

parser = argparse.ArgumentParser(description='PyTorch pt4al Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

server_name = 'hinton'
parameter_path = f'/home/{server_name}/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/pt4al/cifar10/classification_loss'
data_path = f'/home/{server_name}/NAS_AIlab_dataset/dataset/cifar10'

# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.Resize([144,144]),
#     transforms.RandomCrop([128,128]),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.Resize([128,128]),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
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

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

# classes = {'007_강황':0, '013_분꽃':1, '018_배초향':2, '022_부추':3, '029_백도라지':4,
#            '040_고려엉겅퀴':5, '096_곰보배추':6, '100_도꼬마리':7, '110_흰민들레':8, '120_좀향유':9}

# trainset = General_Loader_withpath(is_train=True, transform=transform_test, name_dict=classes, path='/home/hinton/NAS_AIlab_dataset/dataset/NIA_AIhub/herb_rotation')
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

# testset = General_Loader_withpath(is_train=True,  transform=transform_test, name_dict=classes, path='/home/hinton/NAS_AIlab_dataset/dataset/NIA_AIhub/herb_rotation')
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

trainset = General_Loader_withpath(is_train=True, transform=transform_test, name_dict=classes, path=data_path)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=16)

testset = General_Loader_withpath(is_train=True,  transform=transform_test, name_dict=classes, path=data_path)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=16)

# print(next(iter(trainset))[0].shape)

# Model
print('==> Building model..')
net = ResNet18()
# print(net)
net.linear = nn.Linear(512, len(classes))
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
    for batch_idx, (inputs, targets, _ ) in enumerate(trainloader):
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
        for batch_idx, (inputs, targets, _ ) in enumerate(testloader):
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
    with open(parameter_path+'/best_classification.txt','a') as f:
        f.write(str(acc)+':'+str(epoch)+'\n')
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(parameter_path+'/checkpoint'):
            os.mkdir(parameter_path+'/checkpoint')
        # save rotation weights
        torch.save(state, parameter_path+'/checkpoint/classification.pth')
        best_acc = acc

def write_loss(epoch):
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

            loss = loss.item()
            s = str(float(loss)) + '//' + str(path[0]) +'//' + str(float(predicted.eq(targets).sum().item())) + "\n"

            with open(parameter_path+'/test_classification_loss.txt', 'a') as f:
                f.write(s)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

# for epoch in range(start_epoch, start_epoch+121):
#     train(epoch)
#     test(epoch)
#     scheduler.step()

testset = General_Loader_withpath(is_train=False,  transform=transform_test, name_dict=classes, path=data_path)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)

checkpoint = torch.load(parameter_path+'/checkpoint/classification.pth')
net.load_state_dict(checkpoint['net'])
write_loss(1)