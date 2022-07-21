import glob
import os
from PIL import Image, ImageFilter

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2

class OrderPredictionLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA'):
        self.is_train = is_train
        self.transform = transform
        self.label_dict = {'normal':0, 'modified':1}
        if self.is_train==0:
            self.img_path = glob.glob(path+'/train/*/*')
        else:
            self.img_path = glob.glob(path+'/train/*/*')
    
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        
        if self.is_train:
            img = self.transform(img)
            if random.randint(0,1):
                _,w,_ = img.shape
                img1 = img[:,:w//2,:]
                img2 = img[:,w//2:,:]
                img3 = torch.cat((img2,img1),dim=1)
                return img3, 1
            else: return img, 0
        else:
            img = self.transform(img)
            if random.randint(0,1):
                _,w,_ = img.shape
                img1 = img[:,:w//2,:]
                img2 = img[:,w//2:,:]
                img3 = torch.cat((img2,img1),dim=1)
                return img3, 1, self.img_path[idx]
            else: return img, 0, self.img_path[idx]

class RotationLoader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA'):
        self.is_train = is_train
        self.transform = transform
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train==0: # train
            self.img_path = glob.glob(path+'/train/*/*')
        elif self.is_train==2:
            self.img_path = glob.glob(path+'/test/*/*')
        else: # self.is_train==1
            self.img_path = glob.glob(path+'/train/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)

        if self.is_train==1:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]

class General_Loader(Dataset):
    def __init__(self, is_train=True, transform=None, name_dict = None, path='./DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.name_dict = name_dict
        
        if self.is_train:
            if path_list is None:
                self.img_path = glob.glob(path+'/train/*/*')
            else:
                self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob(path+'/test/*/*')
            else:
                self.img_path = path_list
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        # print(self.img_path[idx][:-1])
        if self.is_train:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                # img = cv2.imread(self.img_path[idx][:-1])
                img = cv2.imread(self.img_path[idx])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                # img = cv2.imread(self.img_path[idx][:-1])
                img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = self.name_dict[self.img_path[idx].split('/')[-2]]
        # print(img, label)
        return img, label

class General_Loader_withpath(Dataset):
    def __init__(self, is_train=True, transform=None, name_dict = None, path='./DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.name_dict = name_dict
        
        if self.is_train:
            if path_list is None:
                self.img_path = glob.glob(path+'/train/*/*')
            else:
                self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob(path+'/test/*/*')
            else:
                self.img_path = path_list
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, idx):
        # print(self.img_path[idx][:-1])
        if self.is_train:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                # img = cv2.imread(self.img_path[idx][:-1])
                img = cv2.imread(self.img_path[idx])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                # img = cv2.imread(self.img_path[idx][:-1])
                img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = self.name_dict[self.img_path[idx].split('/')[-2]]
        return img, label, self.img_path[idx]

class Loader2(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.name_dict = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

        if self.is_train: # train
            # self.img_path = glob.glob(path+'/train/*/*')
            if path_list is None:
                self.img_path = glob.glob(path+'/train/*/*')
            else:
                self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob(path+'/test/*/*') # for loss extraction
            else:
                self.img_path = path_list
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train:
            # img = cv2.imread(self.img_path[idx][:-1])
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx][:-1])
            else:
                # print('1 ',self.img_path[idx][:-1])
                img = cv2.imread(self.img_path[idx][:-1])
                # print('2 ',img)
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                img = cv2.imread(self.img_path[idx][:-1])
                
        img = Image.fromarray(img)
        img = self.transform(img)
        
        label = self.name_dict[self.img_path[idx].split('/')[-2]]
        
        # print('3 ',label)

        return img, label


class Loader(Dataset):
    def __init__(self, is_train=True, transform=None, path='./DATA'):
        self.classes = 10 
        self.is_train = is_train
        self.transform = transform
        if self.is_train: # train
            self.img_path = glob.glob('./DATA/train/*/*')
        else:
            self.img_path = glob.glob('./DATA/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label