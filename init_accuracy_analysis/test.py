import os, sys
from glob import glob
from unittest import TestLoader
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import SGD
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import random
import pickle
import copy
import argparse
from tqdm import tqdm
from datasets import base_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--episode', type=int)
parser.add_argument('--data_path', default='/SSDb/Workspaces/yunjae.heo/cifar10/test')
parser.add_argument('--save_path', default='/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/init_analysis')
parser.add_argument('--subset_size', type=int, default=10000)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classes = [path.split('/')[-1] for path in glob(args.data_path+'/*')]

print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
filenames = glob(args.data_path+'/*/*')
testsetdata = base_dataset(filenames, transform_test, classes)
testsetloader = DataLoader(testsetdata, args.batch_size, shuffle=False)

model1 = resnet18()
model1.fc = nn.Linear(512, len(classes))
model1 = model1.cuda()

path = '/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/init_analysis/valid/valid_epi0_epoch196_acc93.435.pkl'
with open(path, 'rb') as f:
    stdict = pickle.load(path)
# stdict = torch.load(path)
model1.load_state_dict(stdict)

def test(net, testloader):
    net.eval()
    tqdmloader = tqdm(testloader)
    for _, (inputs, targets) in enumerate(tqdmloader):
        # print(inputs, targets)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        tqdmloader.set_postfix_str(f'Acc = {acc}%')
    tqdmloader.close()
    
if __name__ == '__main__':
    test(model1, testsetloader)