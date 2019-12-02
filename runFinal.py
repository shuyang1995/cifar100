# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:38:09 2019

@author: Zhihan Chen
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import myModels
from torch.autograd import Variable
from utils import progress_bar
from keras.datasets import cifar100

import os
import time

from risk_control import risk_control


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#confidence prob
delta = 0.001
batch_size = 1
#loaddata -- cifar100 - testData
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#load data for big model
(_, _), (x_test, y_test) = cifar100.load_data()
x_test = x_test.astype('float32')
    
#Model download here
net = myModels.MobileNetV2()
net = net.to(device)
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

#print(net)
#eval:
net.eval()
correct_small = 0.0
total_small = 0
correct_big = 0.0
total_big = 0
model=myModels.cifar100vgg(train=False)
res = {}
kappa= np.array([])
residuals=np.array([])

start_time=time.time()
threshold = 0.10
with torch.no_grad():
    for n_iter, (image, label) in enumerate(testloader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
            image = Variable(image).to(device)
            label = Variable(label).to(device)
            output = net(image)           
            score, pred = output.topk(1, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            if(score>threshold):
                correct_small += correct[:, :1].sum()
                total_small += batch_size
            else:
                predicted_x = model.predict(x_test[n_iter])
                pred = np.argmax(predicted_x,1)
                correct_big +=  sum(pred==label.cpu().numpy())
                total_big += batch_size
print("small model: {}/{} big model: {}/{}".format(correct_small, total_small, correct_big, total_big))