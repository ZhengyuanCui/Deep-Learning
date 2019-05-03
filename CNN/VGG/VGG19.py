import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time

import math
from collections import OrderedDict

BATACH_SIZE = 100
LEARNING_RATE = 0.01
EPOCH = 10

class VGG19(nn.module):
    def __init__(self):
        super(VGG19,self).__inte__()
        #conv layer
        self.conv = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,size = 2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,size = 2),

            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,size = 2),

            nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,size = 2),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,size = 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Linear(4096,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Linear(4096,1000),
            nn.BatchNorm1d(1000),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1,, 4096)
        out = self.fc(out)
        return out

def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(EPOCH):
        start = time.time()
        running_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            lables = lables.to(device)

            optimizer.zero_grad()
            output = net()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 = 99:
                end = time.time()
                print('[epoch %d, iter %5d] loss:%.3f time: %.3f' %
                        epoch+1, i+1, running_loss/100, end-start)
                start = time.time()
                running_loss = 0
    print('Finish training')


def test(testloader, net):
    correct = 0
    total = 0
    with torch.no_grad:
        for images, labels in testloader:
             images = images.to(device)
             lables = lables.to(device)

             output = net(images)
             _,predict = torch.max(output.data,1)
             total += lables.size(0)
             correct += (predict ==labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

def main():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5,0.5,0.5],
                             std = [0.25,0.25,0.25]),
        ])
