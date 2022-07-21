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
from loader import General_Loader_withpath, Loader, Loader2, General_Loader
from util import progress_bar

os.environ["CUDA_VISIBLE_DEVICES"]='6,7'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

server_name = 'hinton'
parameter_path = f'/home/{server_name}/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/h2lal/cifar10'
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
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=32)

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
        if not os.path.isdir(parameter_path+'/checkpoint'):
            os.mkdir(parameter_path+'/checkpoint')
        torch.save(state, parameter_path + f'/checkpoint/main_{cycle}.pth')
        best_acc = acc

# make unlabeled set, UNLABEL = [paths]
def make_unlabeled(data_path):
    img_path = glob.glob(data_path+'/train/*/*')
    return img_path

# make sorted confidence list, conf : [(conf, path)]
def make_conflist(net, unlabel_loader):
    confidence_list = []
    with torch.no_grad():
        for _, (inputs, _, path) in enumerate(unlabel_loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            confidence = torch.max(F.softmax(outputs, dim=-1))
            confidence_list.append((confidence, path))
    confidence_list.sort(key=lambda x:x[0])
    return confidence_list

# Select K datas for Active Learning
def k_selection(confidence_list, epi, k=1000):
    n = len(confidence_list)//(10-1)
    target_data = confidence_list[n*epi:n*(epi+1)]
    return [target_data[i][1] for i in range(k)]

if __name__ == '__main__':
    unlabeled = make_unlabeled(data_path=data_path)
    # print(unlabeled[:5])
    labeled = []
    selected_data = []
    episode = 10
    subset_size = 10000
    init_data_num = 1000
    for epi in range(episode):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        best_acc = 0
        print('Episode ', epi+1)

        if epi == 0:# init_stage
            subset = random.sample(unlabeled, subset_size)
            labeled.extend(subset[:init_data_num])
            # remove labeled data from unlabeled data
            for data in labeled:
                unlabeled.pop(unlabeled.index(data))
            # make train loader
            trainset = General_Loader(is_train=True, transform=transform_train, name_dict=classes, path=data_path, path_list=labeled)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=16)
        else:
            checkpoint = torch.load(parameter_path+f'/checkpoint/main_{epi-1}.pth')
            net.load_state_dict(checkpoint['net'])

            labeled.extend(selected_data)
            for data in selected_data:
                unlabeled.pop(unlabeled.index(data))
            trainset = General_Loader(is_train=True, transform=transform_train, name_dict=classes, path=data_path, path_list=labeled)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=16)

        for epoch in range(200):
            train(net, criterion, optimizer, epoch, trainloader)
            test(net, criterion, epoch, epi)
            scheduler.step()
        with open(parameter_path+f'/main_best.txt', 'a') as f:
            f.write(str(epi) + ' ' + str(best_acc)+'\n')


        subset = random.sample(unlabeled, subset_size)
        unlblset = General_Loader_withpath(is_train=True, transform=transform_train, name_dict=classes, path=data_path, path_list=subset)
        unlbloader = torch.utils.data.DataLoader(unlblset, batch_size=100, shuffle=False, num_workers=16)

        confidence_list = make_conflist(net,unlbloader)
        selected_data = k_selection(confidence_list, epi)
        print(confidence_list[:3])
        print(selected_data[:3])