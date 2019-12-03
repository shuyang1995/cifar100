import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from utils import progress_bar
from utils import WarmUpLR
import myModels
import os
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)
#loaddata -- cifar100
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN,CIFAR100_TRAIN_STD),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN,CIFAR100_TRAIN_STD),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

# Model
print('==> Building model..')
net = myModels.mobilenetv2()
net.cuda()

#train_process
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(),lr=0.001, momentum=0.9, weight_decay=5e-4)
#train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
iter_per_epoch = len(trainloader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)

def train(epoch):

#    print('\nEpoch: %d' % epoch)
#    if epoch >= 2:
#        train_scheduler.step(epoch)
#    net.train()
#    train_loss = 0
#    correct = 0
#    total = 0
#    for batch_idx, (inputs, targets) in enumerate(trainloader):
#        inputs, targets = inputs.to(device), targets.to(device)
#        optimizer.zero_grad()
#        outputs = net(inputs)
#        loss = criterion(outputs, targets)
#        loss.backward()
#        optimizer.step()
#        train_loss += loss.item()
#        _, predicted = outputs.max(1)
#        total += targets.size(0)
#        correct += predicted.eq(targets).sum().item()
#        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_index, (images, labels) in enumerate(trainloader):
        if epoch <= 1:
            warmup_scheduler.step()
        images = Variable(images)
        labels = Variable(labels)
        labels = labels.cuda()
        images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar(batch_index, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_index+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
