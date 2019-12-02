import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import myModels
from torch.autograd import Variable
from utils import progress_bar
import os
import pickle
from risk_control import risk_control
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#confidence prob
delta = 0.001

#loaddata -- cifar100 - testData
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=True, num_workers=2)

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
correct_1 = 0.0
correct_5 = 0.0
total = 0

res = {}
kappa= np.array([])
residuals=np.array([])

with torch.no_grad():
    for n_iter, (image, label) in enumerate(testloader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
            image = Variable(image).cuda()
            label = Variable(label).cuda()
            output = net(image)
            
            kappa = np.append(kappa,np.max(output.cpu().numpy(), 1))
            print(len(kappa))
            residuals = np.append(residuals,(torch.argmax(output,-1).cpu().numpy()!=label.cpu().numpy()))
bound_cal = risk_control()
[theta, b_star] = bound_cal.bound(0.50,delta,kappa,residuals)
