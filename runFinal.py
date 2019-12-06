import argparse
import os
import time

from conf import settings
from utils import get_network, get_test_dataloader
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *


parser = argparse.ArgumentParser(description='2-stage classification model on cifar100')
parser.add_argument('--small', default='mobilenetv2', type=str,  help='small model name under models/')
parser.add_argument('--large', default='seresnet50', type=str,  help='large model name under models/')
parser.add_argument('--small_path', '--sp', default='mobilenetv2', type=str,  help='small model path')
parser.add_argument('--large_path', '--lp', default='mobilenetv2', type=str,  help='large model path')
parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
parser.add_argument('-s', type=bool, default=False, help='whether shuffle the dataset')
args = parser.parse_args()

from models import mobilenetv2 as small_net
from models import seresnet50 as large_net
start_time=time.time()

def main():
    snet = small_net()
    lnet = large_net()
    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    snet.load_state_dict(torch.load(args.small_path), args.gpu)
    print(snet)
    snet.eval()
    lnet.load_state_dict(torch.load(args.large_path), args.gpu)
    print(lnet)
    lnet.eval()
    
    correct_1_small = 0.0
    correct_5_small = 0.0
    correct_1_large = 0.0
    correct_5_large = 0.0
    threshold = 0.10
    total_small = 0
    total_large = 0
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
            image = Variable(image).to(device)
            label = Variable(label).to(device)
            output = snet(image)           
            score, pred = output.topk(1, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            if(score>threshold*100):
                correct_1_small += correct[:, :1].sum()
                total_small += args.b
            else:
                output = lnet(image)
                _, pred = output.topk(1, 1, largest=True, sorted=True)
                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()
                correct_1_large += correct[:, :1].sum()
                total_large += args.b
    print("Top1 acc: small model: {}/{} big model: {}/{}".format(correct_1_small, total_small, correct_1_large, total_large))
    print("Top5 acc: small model: {}/{} big model: {}/{}".format(correct_1_small, total_small, correct_1_large, total_large))


if __name__=='__main__':
    main()

