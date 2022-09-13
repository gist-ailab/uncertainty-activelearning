import os, sys
from glob import glob
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

# args later....
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--episode', type=int)
args = parser.parse_args()

data_path = '/SSDb/Workspaces/yunjae.heo/cifar10/train'
save_path = '/home/bengio/NAS_AIlab_dataset/personal/heo_yunjae/Parameters/Uncertainty/init_analysis'
subset_size = 10000
# episode = 0
total_epochs = 200
batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#get subset and write
def partition(L,size):
    L_copy = copy.deepcopy(L)
    random.shuffle(L_copy)
    return L_copy[:size], L_copy[size:]

classes = [path.split('/')[-1] for path in glob(data_path+'/*')]
filenames = glob(data_path+'/*/*')
subset, validset = partition(filenames, subset_size)
    
#make dataloaders
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

validsetdata = base_dataset(validset, transform_train, classes)
validsetloader = DataLoader(validsetdata, batch_size, shuffle=True)

#make model, loss function & optimizer
model1 = resnet18()
model1.fc = nn.Linear(512, len(classes))
model1 = model1.cuda()

criterion1 = nn.CrossEntropyLoss()
optimizer1 = SGD(model1.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[160])

#train model with validset
def train(net, criterion, optimizer, epoch, trainloader, mode, subsetnum = None):
    print('\nEpoch: %d' % epoch)
    global bestacc
    loc_bestacc = 0
    statedict = None
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    tqdmloader = tqdm(trainloader)
    for _, (inputs, targets) in enumerate(tqdmloader):
        # print(inputs, targets)
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
        acc = 100.*correct/total
        tqdmloader.set_postfix_str(f'Acc = {acc}%')
        if acc >= loc_bestacc:
            loc_bestacc = acc
            statedict = net.state_dict()
    if mode == 'valid' and loc_bestacc > bestacc:
        torch.save(statedict, save_path+f'/{mode}_epi{args.episode}_epoch{epoch}_acc{loc_bestacc}.pkl')
        bestacc = loc_bestacc
    elif mode == 'subset' and loc_bestacc > bestacc:
        torch.save(statedict, save_path+f'/{mode}_epi{args.episode}_sub{subsetnum}_epoch{epoch}_acc{loc_bestacc}.pkl')
        bestacc = loc_bestacc
    tqdmloader.close()

if __name__ == '__main__':
    #valid
    bestacc = 0
    result = open(save_path+'/valid_results.txt', 'ab')
    for i in range(total_epochs):
        train(model1, criterion1, optimizer1, i, validsetloader, mode='valid')
        scheduler1.step()
        pickle.dump([i, bestacc], result)
    result.close()
    #subset
    for subsetnum in range(5):
        bestacc = 0
        
        sub = random.choice(subset, 1000)
        subsetdata = base_dataset(sub, transform_train, classes)
        subsetloader = DataLoader(subsetdata, batch_size, shuffle=True)
        
        model2 = resnet18()
        model2.fc = nn.Linear(512, len(classes))
        model2 = model2.cuda()

        criterion2 = nn.CrossEntropyLoss()
        optimizer2 = SGD(model2.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[160])
        
        #save subsets
        if not os.path.isdir(save_path+'/subsets'):
            os.mkdir(save_path+'/subsets')
        with open(save_path+'/subsets'+f'/episode{args.episode}_sub{subsetnum}.txt', 'w') as f:
            pickle.dump(sub, f)
        
        result = open(save_path+'/subset_results.txt', 'ab')
        for i in range(total_epochs):
            train(model2, criterion2, optimizer2, i, subsetloader, mode='subset', subsetnum=subsetnum)
            scheduler2.step()
            pickle.dump([i, bestacc], result)
        result.close()