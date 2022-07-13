#classification loss가 있을 때 loss와 uncertatinty와의 관계 확인
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

from models.resnet_with_feature import *
from loader import Loader, OrderPredictionLoader, General_Loader_withpath
from utils import progress_bar
import os
from sklearn.metrics import pairwise_distances

os.environ["CUDA_VISIBLE_DEVICES"]='4'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

server_name = 'hinton'
parameter_path = f'/home/{server_name}/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/pt4al/cifar10/loss_analysis'
data_path = f'/home/{server_name}/NAS_AIlab_dataset/dataset/cifar10'

classes = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

net = ResNet18()
net.linear = nn.Linear(512, len(classes))
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# checkpoint = torch.load(parameter_path+'/checkpoint/classification.pth')
# net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

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

traindata = open('/home/hinton/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/pt4al/cifar10/random/loss/batch_0.txt', 'r').readlines()
traindata = [traindata[5*i] for i in range(len(traindata)//5)]

trainset = General_Loader_withpath(is_train=True,  transform=transform_train, name_dict=classes, path=data_path, path_list=traindata)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=250, shuffle=True, num_workers=32)

testset = General_Loader_withpath(is_train=False,  transform=transform_test, name_dict=classes, path=data_path)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=32)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

def train(epoch):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, path) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = net(inputs)

        # print(type(outputs))
        # print(type(targets))

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

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

def loss_confidence_distance(train_feature):
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
                feature = feature.cpu()
                distance = np.min(pairwise_distances(feature, train_feature))
                # distance = len(trainloader)
                s = str(float(loss)) + '//' + str(float(confidence)) + '//' + str(float(distance)) + '//' + str(path[0]) + "\n"
                
                f.write(s)
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def get_train_vector():
    train_feature = torch.empty((len(trainloader), 512))
    for train_idx, (inputs, _, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        _, feature = net(inputs)
        train_feature[train_idx] = feature
    return train_feature

best_acc = 0
# for i in range(120):
#     train(i)

checkpoint = torch.load(parameter_path+'/checkpoint/classification.pth')
net.load_state_dict(checkpoint['net'])

trainset = General_Loader_withpath(is_train=True,  transform=transform_train, name_dict=classes, path=data_path, path_list=traindata)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=32)

train_feature = get_train_vector()
train_feature = train_feature.cpu()
loss_confidence_distance(train_feature)