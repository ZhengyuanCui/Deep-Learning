import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time

import math
from collections import OrderedDict

class VGG16(nn.module):
    def __init__(self):
        super(VGG16,self).__inte__()
