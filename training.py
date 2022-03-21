import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from collections import OrderedDict

import torch.nn.functional as F
import dataloader
import model

pth='D:/Dev Projects/AI_Projects/DC_GAN/data' #path of dataset on my local machine

train_data=dataloader.transformation(pth)

train_loader=dataloader.data_load(train_data, batch_size=64, num_workers=2)


'''Instantianting the whole network, setting hyperparams'''

conv_dim=32
z_size=100

discrim_net=model.Discriminator(conv_dim)
gen_net=model.Generator(z_in=z_size, conv_dim=conv_dim)

'''Defining the loss, generator loss and discriminator loss'''
if torch.cuda.is_available():
  gen_net.cuda()
  discrim_net.cuda()
  print("Training on GPU")

else:
  print(" Training on CPU")


def loss_real(D_out,smooth=False):
  batch_size=D_out.size(0)
  #label smoothing

  if smooth:
    #smoothing and real labels=0.9
    labels=torch.ones(batch_size)*0.9
  else:
    #real labels
    labels=torch.ones(batch_size)

  if torch.cuda.is_available():
    labels=labels.cuda()

  criterion=nn.BCEWithLogitsLoss()
  loss=criterion(D_out.squeeze(),labels)

  return loss



def loss_fake(D_out):
  batch_size=D_out.size(0)

  labels=torch.zeros(batch_size)
  labels=torch.zeros(batch_size)

  if torch.cuda.is_available():
    labels= labels.cuda()
  criterion=nn.BCEWithLogitsLoss()

  loss=criterion(D_out.squeeze(), labels)

  return loss
#setting the optimmisers for both generator and discriminator

lr=0.0002
beta1=0.5
beta2=0.999

discrim_optim=optim.Adam(discrim_net.parameters(),lr,[beta1,beta2])
gen_optim=optim.Adam(gen_net.parameters(), lr, [beta1,beta2])

# helper scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max - min) + min
    return x

num_epochs=1000
#loss and generated, fake samples

samples=[]
losses=[]
print_every=300

sample_size=16
fixed_z=np.random.uniform(-1,1,size=(sample_size,z_size))
fixed_z=torch.from_numpy(fixed_z).float()

for epoch in range(num_epochs):
  for batch_i, (real_img, _) in enumerate(train_loader):
    batch_size=real_img.size(0)

    real_img=scale(real_img)
    #---------------------------
    #  Training the discriminator
    #---------------------------

    discrim_optim.zero_grad()
    #train with real img

    if torch.cuda.is_available():
      real_img=real_img.cuda()
    

    D_real=discrim_net(real_img)
    d_loss_real=loss_real(D_real) #calculate the discriminator loss on real images

    #training with fake images

    z=np.random.uniform(-1,1,size=(batch_size,z_size)) #generating fake images
    z=torch.from_numpy(z).float()

    if torch.cuda.is_available():
      z=z.cuda()
    fake_img=gen_net(z)

    #calculate dicriminator loss on fake images

    D_fake=discrim_net(fake_img)
    D_fake_loss=loss_fake(D_fake)

    # adding loss and perfom backprop

    d_loss=d_loss_real +D_fake_loss
    d_loss.backward()
    discrim_optim.step()


    #---------------------------------
    # Training the Generator
    #--------------------------------

    gen_optim.zero_grad()

    #Training generator with fake images and flipped labels

    z=np.random.uniform(-1,1,size=(batch_size, z_size))
    z=torch.from_numpy(z).float()

    if torch.cuda.is_available():
      z=z.cuda()

    fake_img=gen_net(z)

    #calculate the discriminator loss on fake images using flipped labels

    D_fake=discrim_net(fake_img)
    g_loss=loss_real(D_fake) #using real loss to flip labels

    g_loss.backward()
    gen_optim.step()

    #printing some loss stats

    if batch_i % print_every==0:
      losses.append((d_loss.item(),g_loss.item()))
      print(
          'Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch+1, num_epochs, d_loss.item(), g_loss.item())
      )

    
  '''Generating and saving sample images after each epoch'''

  gen_net.eval() #generating sample images

  if torch.cuda.is_available():
    fixed_z=fixed_z.cuda()
  samples_z=gen_net(fixed_z)
  samples.append(samples_z)
  gen_net.train()

#saving generated sample images as pickle file
import pickle as pkl
with open('train_samples.pkl','wb') as f:
  pkl.dump(samples,f)
  

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()


#viewing samples of generated images

def sample_view(epoch, samples):
  fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
  for ax, img in zip(axes.flatten(), samples[epoch]):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    im = ax.imshow(img.reshape((32,32,3)))
    
    
_=sample_view(-1,samples)