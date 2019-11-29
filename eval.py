import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from utils import progress_bar
import os

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
net = models.MobileNetV2()
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
for n_iter, (image, label) in enumerate(testloader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)
        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
        #compute top 5
        correct_5 += correct[:, :5].sum()
        #compute top1
        correct_1 += correct[:, :1].sum()
    print()
    print("Top 1 err: ", 1 - correct_1 / len(testloader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(testloader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
