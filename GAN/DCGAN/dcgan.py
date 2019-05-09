import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#what should GAN have: generator: fake image maker and discriminator: real and generated image
# fake image generate: input: random bits, network: upsample cnn network,
# discriminators: cnn nets with 1 output

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data)


def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    ###########################
    ######### TO DO ###########
    ###########################
    random_noise = torch.FloatTensor(batch_size, dim).uniform_(-1, 1)
    return random_noise

def discriminator(batch_size):
    """
    Build and return a PyTorch model for the DCGAN discriminator
    using the architecture described below:

    * Reshape into image tensor (Use Unflatten!)
    * 32 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
    * Max Pool 2x2, Stride 2
    * 64 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
    * Max Pool 2x2, Stride 2
    * Flatten
    * Fully Connected size 4 x 4 x 64, Leaky ReLU(alpha=0.01)
    * Fully Connected size 1

    """
    return nn.Sequential(
        ###########################
        ######### TO DO ###########
        ###########################
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1,32,kernel_size=5),
        nn.LeakyReLU(0.01, inplace = True),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(32,64,kernel_size=5),
        nn.LeakyReLU(0.01, inplace = True),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        Flatten(),
        nn.Linear(64*4*4,64*4*4),
        nn.LeakyReLU(0.01, inplace = True),
        nn.Linear(64*4*4,1)
    )


def generator():
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described below:

    * Fully connected of size 1024, ReLU
    * BatchNorm
    * Fully connected of size 7 x 7 x 128, ReLU
    * BatchNorm
    * Reshape into Image Tensor
    * 64 conv2d^T filters of 4x4, stride 2, 'same' padding, ReLU
    * BatchNorm
    * 1 conv2d^T filter of 4x4, stride 2, 'same' padding, TanH
    * Should have a 28x28x1 image, reshape back into 784 vector

    Note: for conv2d^T you should use torch.nn.ConvTranspose2d
          Plese see the documentation for it in the pytorch site

    """
    return nn.Sequential(
        ###########################
        ######### TO DO ###########
        ###########################
        nn.Linear(noise_dim,1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024,7*7*128),
        nn.ReLU(),
        nn.BatchNorm1d(7*7*128),
        Unflatten(batch_size, 128, 7, 7),
        nn.ConvTranspose2d(128,64,kernel_size = 4,stride = 2,padding = 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,1,kernel_size = 4,stride = 2,padding = 1),
        nn.Tanh(),
        Flatten()
    )

def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    optimizer = optim.Adam(model.parameters(),lr=1e-3, eps=1e-08, betas=(0.5, 0.999))
    return optimizer


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    bce = (input.clamp(min=0) - input * target + (1 +  (-input.abs()).exp()).log()).mean()
    return bce



def main():
    #get data
    #train the network
    #test it

if __name__ == "__main__":
    main()
