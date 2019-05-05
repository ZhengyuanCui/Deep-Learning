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
EPOCH = 20

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19,self).__init__()
        #conv layer
        self.conv = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,stride = 2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,stride = 2),

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

            nn.MaxPool2d(kernel_size = 2,stride = 2),

            nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,stride = 2),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2,stride = 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512,100),
            #nn.BatchNorm1d(100),
            nn.Softmax(),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 512)
        out = self.fc(out)
        return out

def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(EPOCH):
        start = time.time()
        running_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                end = time.time()
                print('[epoch %d, iter %5d] loss:%.3f time: %.3f' %
                        (epoch+1, i+1, running_loss/100, end-start))
                start = time.time()
                running_loss = 0
    print('Finish training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
             images = images.to(device)
             labels = labels.to(device)

             output = net(images)
             _,predict = torch.max(output.data,1)
             total += labels.size(0)
             correct += (predict ==labels).sum().item()
    print('Accuracy of the network on the %d test images: %d %%' % (
        total, 100 * correct / total))
    print(correct)

def main():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5,0.5,0.5],
                             std = [0.25,0.25,0.25]),
        ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)

    net = VGG19().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)
    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)

if __name__ =="__main__":
    main()
