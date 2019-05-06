import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import time
import math
from collections import OrderedDict
from torch.autograd import Variable

#what should GAN have: generator: fake image maker and discriminator: real and generated image
# fake image generate: input: random bits, network: upsample cnn network,
# discriminators: cnn nets with 1 output
