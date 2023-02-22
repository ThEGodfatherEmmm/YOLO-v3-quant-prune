import os
import argparse
import torch
from torch2trt import torch2trt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import time

parser = argparse.ArgumentParser(description='PyTorch to TensorRT')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--refine', default='./logs/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs_trt', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--cfg', action='store_true', default=False,
                    help='cfg or not')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

checkpoint = torch.load(args.refine)
if(args.cfg):
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
else:
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
# model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
model.load_state_dict(checkpoint['state_dict'])

if args.cuda:
    model.cuda().half()
    print('using CUDA')

# data = torch.randn((1, 3, 224, 224)).cuda().float()
test_set = datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]))
data,target=test_set[0]
data = Variable(torch.unsqueeze(data, dim=0).float(), requires_grad=False).cuda().half()

model.eval()
model_trt = torch2trt(model, [data], fp16_mode=True, max_batch_size=64)
print('conversion succeeded')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    start = time.clock()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda().half(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    end = time.clock()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('infer_time:', end-start)
    return correct / float(len(test_loader.dataset))

def test_trt():
    model.eval()
    test_loss = 0
    correct = 0
    start = time.clock()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda().half(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model_trt(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    end = time.clock()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('infer_time:', end-start)
    return correct / float(len(test_loader.dataset))

prec1 = test()
prec2 = test_trt()

torch.save(model_trt.state_dict(), os.path.join(args.save, 'vgg_model_trt.pth.tar'))
print('finished')