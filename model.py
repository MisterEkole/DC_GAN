import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from collections import OrderedDict

import torch.nn.functional as F

'''
Building the GAN
GAN is made up of a Discriminator and Generator
'''
#building the conv layer

def conv_layer(cin, cout,kernel_size,stride=2,padding=1,batch_norm=True):
  '''Creating a convolutional layer with batch norm(optional)'''
  layers=[]
  conv=nn.Conv2d(cin, cout,
                 kernel_size,stride, padding, bias=False)
  #appen layers

  layers.append(conv)

  if batch_norm:
    layers.append(nn.BatchNorm2d(cout))
  
  return nn.Sequential(*layers)




class Discriminator(nn.Module):

  def __init__(self, conv_dim=32):
    super(Discriminator,self).__init__()

    ''' Defining the convolution layers of the discrimator'''
    self.conv_dim=conv_dim

    self.conv1=conv_layer(3,conv_dim,4,batch_norm=False)
    self.conv2=conv_layer(conv_dim, conv_dim*2,4, batch_norm=True)
    self.conv3=conv_layer(conv_dim*2,conv_dim*4,4)
   

    self.fc=nn.Linear(conv_dim*4*4*4,1)


  def forward(self,x):
    output=F.leaky_relu(self.conv1(x),0.2)
    output=F.leaky_relu(self.conv2(output),0.2)
    output=F.leaky_relu(self.conv3(output),0.2)

    #flatten outputs
    output=output.view(-1,self.conv_dim*4*4*4)

    output=self.fc(output)
    #output=F.sigmoid(self.conv4(output))

    return output

'''Building the deconv func'''

def deconv_layer(cin, cout,kernel_size,stride=2,padding=1,batch_norm=True):
  '''Creating a transposed convolutional layer with batch norm(optional)'''
  layers=[]
  transpose_conv=nn.ConvTranspose2d(cin, cout,
                 kernel_size,stride, padding, bias=False)
  #append layers

  layers.append(transpose_conv)

  if batch_norm:
    layers.append(nn.BatchNorm2d(cout))
  
  return nn.Sequential(*layers)


'''Building the generator'''

class Generator(nn.Module):
  def __init__(self,z_in,conv_dim=32):
    super(Generator, self).__init__()
    self.conv_dim=conv_dim

    self.fc=nn.Linear(z_in,conv_dim*4*4*4)

    self.d_conv1=deconv_layer(conv_dim*4,conv_dim*2,4)
    self.d_conv2=deconv_layer(conv_dim*2, conv_dim,4)
    self.d_conv3=deconv_layer(conv_dim,3,4,batch_norm=True)
   


  def forward(self, x):
    #fully connected layer and reshaping
    output=self.fc(x)
    output=output.view(-1, self.conv_dim*4,4,4)

    #hidden conv transpose + relu

    output=F.relu(self.d_conv1(output))
    output=F.relu(self.d_conv2(output))
    

    #final layer +tanh activation func

    output=self.d_conv3(output)
    output=F.tanh(output)

    return output
