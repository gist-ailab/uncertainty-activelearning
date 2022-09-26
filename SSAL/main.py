import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import pickle
from torchvision import datasets
import torchvision.models as models
from utils import *
from torch.utils.data.sampler import SubsetRandomSampler

# Initial accuracy를 올릴 수 있는 init sampling query에 대한 연구

seed = 1
torch.random.manual_seed(seed)

loss_list_path = '/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/SimCLR-2/runs/Sep23_15-11-09_ailab-server-bengio/checkpoints/sampled_idx.pkl'
# init_path = '/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/SimCLR-2/runs/Sep23_15-11-09_ailab-server-bengio/checkpoints/model.pth'
init_path = None
root_save_path = '/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/SSAL'
save_path = os.path.join(root_save_path, 'woinit')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(loss_list_path, 'rb') as f:
    loss_list = pickle.load(f)

train_idx = get_high_sim(loss_list)

train_dataset = datasets.CIFAR10('/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         transform=get_rand_augment('cifar10'))
train_sampler = SubsetRandomSampler(train_idx)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler,
                                  drop_last=True, shuffle=False)

test_dataset = datasets.CIFAR10('/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/uncertainty-activelearning/SSAL/semi/data', 
                                         download=False,
                                         train = False,
                                         transform=get_test_augment('cifar10'))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = models.resnet18()
if init_path != None:
    checkpoint = torch.load(init_path)
    checkpoint = std_convert(checkpoint)
    model.load_state_dict(checkpoint)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

def train(epoch, model, train_loader, criterion, optimizer):
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
        pbar.set_postfix({'loss':train_loss, 'acc':100*correct/total})

def test(epoch, model, test_loader, criterion):
    model.eval()
    global best_acc
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

            pbar.set_postfix({'loss':test_loss, 'acc':100*correct/total})

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/main_{epoch}.pth')
        best_acc = acc

if __name__ == "__main__":
    best_acc = 0
    for i in range(200):
        train(i, model, train_loader, criterion, optimizer)
        if i>100 or i%10==0:
            test(i, model, test_loader, criterion)