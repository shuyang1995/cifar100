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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#loaddata -- cifar100 - testData
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

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

for n_iter, (image, label) in enumerate(testloader):
        if n_iter > 300:
            break
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)
        scores, pred = output.topk(5, 1, largest=True, sorted=True)
        #scores shape: 100*5
        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
        #compute top 5
        correct_5 += correct[:, :5].sum()
        #compute top1
        correct_1 += correct[:, :1].sum()
        #correct[:,:1] shape: 100*1
        #record:
        res[n_iter] = [scores, correct[:, :1]]
        #print("score shape is ",scores.size())
        #print("correct1 shape is ",correct[:,:1].size())

file = open("scores", "wb")
pickle.dump(res, file)
file.close()

print()
print("Top 1 err: ", 1 - correct_1 / len(testloader.dataset))
print("Top 5 err: ", 1 - correct_5 / len(testloader.dataset))
print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
