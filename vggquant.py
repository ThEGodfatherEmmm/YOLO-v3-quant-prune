from collections import OrderedDict
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
import models
# Quant
import quant
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Prune settings
parser = argparse.ArgumentParser(
    description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')

parser.add_argument('--model', default='./logs/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')

parser.add_argument('--save', default='./logs_quant', type=str, metavar='',
                    help='path to save pruned model (default: none)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--quant_method', default='linear',
                    help='linear|minmax|log|tanh')
parser.add_argument('--n_sample', type=int, default=20,
                    help='number of samples to infer the scaling factor')
parser.add_argument('--param_bits', type=int, default=8,
                    help='bit-width for parameters')
parser.add_argument('--bn_bits', type=int, default=32,
                    help='bit-width for running mean and std')
parser.add_argument('--fwd_bits', type=int, default=8,
                    help='bit-width for layer output')
parser.add_argument('--overflow_rate', type=float,
                    default=0.0, help='overflow rate')
parser.add_argument('--cfg', action='store_true', default=False,
                    help='cfg or not')


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    starter, ender = torch.cuda.Event(
        enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = len(test_loader)
    timings = np.zeros((repetitions, 1))
    for rep, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        starter.record()
        output = model(data)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    total_time = np.sum(timings)
    print(' * TotalTime {total_time:.1f}ms Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f} FPS@1 {mean_fps:.2f}'.format(
        total_time = total_time, mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # model = vgg(dataset=args.dataset, depth=args.depth)

    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model) 
            if(args.cfg):
                model = models.__dict__[args.arch](
                dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
                cfg = checkpoint['cfg']
            else:
                model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))
    # print(model)
    if args.cuda:
        model.cuda()
    test(model)
    # Real quant
    if args.param_bits < 32:
        state_dict = model.state_dict()

        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'running' in k:
                if args.bn_bits >= 32:
                    # print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = args.bn_bits
            else:
                bits = args.param_bits
            if args.quant_method == 'linear':
                sf = bits - 1. - \
                    quant.compute_integral_part(
                        v, overflow_rate=args.overflow_rate)
                v_quant = quant.linear_quantize(v, sf, bits=bits)
                # print("Orign v is ", v)
                # print("Quant v is ", v_quant)
            elif args.quant_method == 'log':
                overflow_rate=args.overflow_rate
                v_quant = quant.log_linear_quantize(v, overflow_rate, bits=bits)
            elif args.quant_method == 'minmax':
                v_quant = quant.min_max_quantize(v, bits=bits)
            elif args.quant_method == 'minmax_log':
                v_quant = quant.log_minmax_quantize(v, bits=bits)
            else:
                v_quant = quant.tanh_quantize(v, bits=bits)
            state_dict_quant[k] = v_quant
            # print(k, bits)

        model.load_state_dict(state_dict_quant)

    # quantize forward activation
    if args.fwd_bits < 32:
        model_raw = quant.duplicate_model_with_quant(model, bits=args.fwd_bits, overflow_rate=args.overflow_rate,
                                                     counter=args.n_sample, type=args.quant_method)  # repalce layers

    # torch.save(model_raw.state_dict(), './tensorrt/vgg_quant.pt')
    if args.cfg:   
        torch.save({'cfg': cfg, 'state_dict':model_raw.state_dict()}, os.path.join(args.save, args.quant_method+'.pth.tar'))
    else:    
        torch.save({'state_dict':model_raw.state_dict()}, os.path.join(args.save, args.quant_method+'.pth.tar'))
    # acc = test(model)
    test(model_raw)

    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))


if __name__ == '__main__':
    main()
