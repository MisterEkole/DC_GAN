# Deep Convolutional Generative Adversarial Network

Generative adversarial networks (GANs), introduced in 2014 by Goodfellow et al.,are
an alternative to VAEs for learning latent spaces of images. They enable the generation
of fairly realistic synthetic images by forcing the generated images to be statistically
almost indistinguishable from real ones.

A GAN is made up of two parts:
* A Generator Network: that takes a random input vector and decodes it to synthetic image
* A Discriminator Network: that takes either synthetic or real image as input and predict whether the image came from training set or was created by generator network

This repository presents an implementation in pytorch of a DCGAN(Deep Convolutional Generative Adversarial Network); a basic GAN with generator and discriminator being deep convnet
The model was trained on abstract images dataset from kaggle. The goal here was to use DCGAN to generate abstract  fake images from real ones.

### Model Architecture

#### Discriminator
* This is a ConvNet without any pooling layer.
* The inputs of the discriminator are 32*32*3 Tensor images
* A few convolution hidden layers
* A fully connected layer with BCEWithLogitsLoss for output

#### Generator
* A fully connected input layer reshaped into a deep and narrow layer 4*4*512
* BatchNormalisation and LeakyRelu activation function
* A few transpose convolutional layers
* A last output layer with tanh activation function

### Dependenciec
* Pytorch
* Numpy
* Google Colab
* Torchvision
* Matplotlib
* Python 3.8
