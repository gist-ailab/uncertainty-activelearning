#classification loss가 있을 때 loss와 uncertatinty와의 관계 확인
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models.resnet_with_feature import *
from loader import Loader, OrderPredictionLoader, General_Loader_withpath
from utils import progress_bar
import os

os.environ["CUDA_VISIBLE_DEVICES"]='4'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

server_name = 'hinton'
parameter_path = f'/home/{server_name}/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/pt4al/cifar10/classification_loss'
data_path = f'/home/{server_name}/NAS_AIlab_dataset/dataset/cifar10'

classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

net = ResNet18()
net.linear = nn.Linear(512, len(classes))
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load(parameter_path+'/checkpoint/classification.pth')
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

traindata = open('/home/hinton/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/pt4al/cifar10/random/loss/batch_0.txt', 'r').readlines()
trainset = General_Loader_withpath(is_train=True,  transform=transform_test, name_dict=classes, path=data_path, path_list=traindata)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=32)

testset = General_Loader_withpath(is_train=False,  transform=transform_test, name_dict=classes, path=data_path)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=32)

def loss_confidence_distance(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with open(parameter_path+'/classification_loss_confidence_distance.txt', 'w') as f:
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(testloader):
                inputs, targets= inputs.to(device), targets.to(device)
                outputs, feature = net(inputs)
                
                # print(outputs)
                confidence = torch.max(F.softmax(outputs, dim=-1))
                # print(confidence)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = loss.item()
                distance = 0.0
                for train_idx, (tinputs, ttargets, tpath) in enumerate(trainloader):
                    tinputs = tinputs.to(device)
                    toutputs, tfeature = net(tinputs)
                    distance += (torch.sum((feature - tfeature)**2))**0.5
                distance /= len(trainloader)
                s = str(float(loss)) + '//' + str(float(confidence)) + '//' + str(float(distance)) + '//' + str(path[0]) + "\n"
                
                f.write(s)
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

loss_confidence_distance(1)