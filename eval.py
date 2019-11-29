import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from utils import progress_bar
import os
from models.mobilenetv2 import mobilenetv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#loaddata -- cifar100 - testData
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)

#Model download

net = torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
net = net.cuda()
print(net)
