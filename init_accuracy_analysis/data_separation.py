import glob
import torch
import torch.nn.functional as F 
from torch.optim import Adam
from torchvision.models import resnet18
from torch.utils.data import dataloader
from torch.nn import CrossEntropyLoss

