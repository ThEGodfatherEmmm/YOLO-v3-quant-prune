from torch.autograd import Variable
import torch
from torch import nn
from collections import OrderedDict
import math
from IPython import embed


def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * len(sorted_value))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = float(v.data.cpu())
    sf = math.ceil(math.log2(v+1e-12))
    return sf


def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input) - 1
    delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)
    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value


def log_linear_quantize(input, overflow_rate, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    s = torch.sign(input)
    input0 = torch.log(torch.abs(input))
    sf = bits - 1. - \
                    compute_integral_part(
                        torch.abs(input0), overflow_rate=overflow_rate)
    v = linear_quantize(input0, sf, bits-1)
    v = torch.exp(v) * s
    return v


def min_max_quantize(input, bits):
    # Finsh the minmax quant according to the theory
    # pass
    max = torch.max(input)
    min = torch.min(input)
    R = math.pow(2.0, bits-1) - 1
    # q = torch.round((input-min)*R/(max-min+1e-7))*(max-min)+min
    # print(input)
    q = torch.floor((input-min)/(max-min+1e-7)*R+0.5)*(max-min)/R+min
    # q = q.type(torch.int8)
    # print(q)
    return q


def log_minmax_quantize(input, bits):
    # Finsh the Log minmax quant according to the theory
    # pass
    s = torch.sign(input)
    input = torch.abs(input)
    input = torch.log(input)
    max = torch.max(input)
    min = torch.min(input)
    R = math.pow(2.0, bits-1) - 1
    q = torch.floor((input-min)/(max-min+1e-7)*R+0.5)*(max-min)/R+min
    q = torch.exp(q) * s

    return q


def tanh_quantize(input, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(input)
    input = torch.tanh(input)  # [-1, 1]
    input_rescale = (input + 1.0) / 2  # [0, 1]
    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = 2 * v - 1  # [-1, 1]

    v = 0.5 * torch.log((1 + v) / (1 - v))  # arctanh
    return v


class LinearQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LinearQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            sf_new = self.bits - 1 - \
                compute_integral_part(input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)


class LogQuant(nn.Module):
    def __init__(self, name, bits, sf=None, overflow_rate=0.0, counter=10):
        super(LogQuant, self).__init__()
        self.name = name
        self._counter = counter

        self.bits = bits
        self.sf = sf
        self.overflow_rate = overflow_rate

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        if self._counter > 0:
            self._counter -= 1
            log_abs_input = torch.log(torch.abs(input))
            sf_new = self.bits - 1 - \
                compute_integral_part(log_abs_input, self.overflow_rate)
            self.sf = min(self.sf, sf_new) if self.sf is not None else sf_new
            return input
        else:
            output = log_linear_quantize(input, self.sf, self.bits)
            return output

    def __repr__(self):
        return '{}(sf={}, bits={}, overflow_rate={:.3f}, counter={})'.format(
            self.__class__.__name__, self.sf, self.bits, self.overflow_rate, self.counter)


class NormalQuant(nn.Module):
    def __init__(self, name, bits, quant_func):
        super(NormalQuant, self).__init__()
        self.name = name
        self.bits = bits
        self.quant_func = quant_func

    @property
    def counter(self):
        return self._counter

    def forward(self, input):
        output = self.quant_func(input, self.bits)
        return output

    def __repr__(self):
        return '{}(bits={})'.format(self.__class__.__name__, self.bits)


def duplicate_model_with_quant(model, bits, overflow_rate=0.0, counter=10, type='linear'):
    """assume that original model has at least a nn.Sequential"""
    assert type in ['linear', 'minmax', 'log', 'tanh', 'minmax_log']
    if isinstance(model, nn.Sequential):
        l = OrderedDict()
        for k, v in model._modules.items():
            if isinstance(v, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.AvgPool2d)):
                l[k] = v
                if type == 'linear':
                    quant_layer = LinearQuant('{}_quant'.format(
                        k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                elif type == 'log':
                    # quant_layer = LogQuant('{}_quant'.format(k), bits=bits, overflow_rate=overflow_rate, counter=counter)
                    quant_layer = NormalQuant('{}_quant'.format(
                        k), bits=bits, quant_func=log_minmax_quantize)
                elif type == 'minmax':
                    quant_layer = NormalQuant('{}_quant'.format(
                        k), bits=bits, quant_func=min_max_quantize)
                elif type == 'minmax_log':
                    quant_layer = NormalQuant('{}_quant'.format(
                        k), bits=bits, quant_func=log_minmax_quantize)
                else:
                    quant_layer = NormalQuant('{}_quant'.format(
                        k), bits=bits, quant_func=tanh_quantize)
                l['{}_{}_quant'.format(k, type)] = quant_layer
            else:
                l[k] = duplicate_model_with_quant(
                    v, bits, overflow_rate, counter, type)
        m = nn.Sequential(l)
        return m
    else:
        for k, v in model._modules.items():
            model._modules[k] = duplicate_model_with_quant(
                v, bits, overflow_rate, counter, type)
        return model
