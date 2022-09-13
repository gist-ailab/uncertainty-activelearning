import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

class base_dataset(nn.Module):
    def __init__(self, path_list, transform, classes):
        self.pl = path_list
        self.tf = transform
        self.classes = classes
    
    def __getitem__(self, idx):
        image = Image.open(self.pl[idx])
        image = self.tf(image)
        label = self.pl[idx].split('/')[-2]
        return image, self.classes.index(label)
    
    def __len__(self):
        return len(self.pl)