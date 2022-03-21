import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from collections import OrderedDict

''' Func to read dataset path and apply transforms'''
def transformation(pth):
  data_transforms= transforms.Compose(
    [
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  dt=datasets.ImageFolder(pth,transform=data_transforms)

  return dt

''' Func to create dataloader'''
def data_load(dt,batch_size,num_workers):
  x=torch.utils.data.DataLoader(dt, batch_size=batch_size, num_workers=num_workers, shuffle=True)

  return x