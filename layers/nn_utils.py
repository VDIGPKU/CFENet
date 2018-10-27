import torch
import torch.nn as nn
from senet import *

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def vgg(input_channel=3, batch_norm=False):
    layers = []
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
    in_channels = input_channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def get_basemodel(basename):
    if basename == 'vgg':
        base = vgg(input_channel=3)
        return nn.ModuleList(base)
    if basename == 'seresnet50':
        base = se_resnet50(1000, pretrained='imagenet')
        return base

class CFEM(nn.Module):

    def __init__(self, in_planes, out_planes, stride = 1, scale = 0.1, groups=8, thinning=2, k = 7, dilation=1):
        super(CFEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        second_in_planes = in_planes // thinning

        p = (k-1)//2
        self.cfem_a = list()
        self.cfem_a += [BasicConv(in_planes, in_planes, kernel_size = (1,k), stride = 1, padding = (0,p), groups = groups, relu = False)]
        self.cfem_a += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size=3, stride=stride, padding=dilation, groups = 4, dilation=dilation)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = (k, 1), stride = 1, padding = (p, 0), groups = groups, relu = False)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a = nn.ModuleList(self.cfem_a)

        self.cfem_b = list()
        self.cfem_b += [BasicConv(in_planes, in_planes, kernel_size = (k,1), stride = 1, padding = (p,0), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = 3, stride=stride, padding=dilation,groups =4,dilation=dilation)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = (1, k), stride = 1, padding = (0, p), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_b = nn.ModuleList(self.cfem_b)


        self.ConvLinear = BasicConv(2 * second_in_planes, out_planes, kernel_size = 1, stride = 1, relu = False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size = 1, stride = stride, relu = False)
        self.relu = nn.ReLU(inplace = False)

    def forward(self,x):

        x1 = self.cfem_a[0](x)
        x1 = self.cfem_a[1](x1)
        x1 = self.cfem_a[2](x1)
        x1 = self.cfem_a[3](x1)
        x1 = self.cfem_a[4](x1)

        x2 = self.cfem_b[0](x)
        x2 = self.cfem_b[1](x2)
        x2 = self.cfem_b[2](x2)
        x2 = self.cfem_b[3](x2)
        x2 = self.cfem_b[4](x2)

        out = torch.cat([x1, x2], 1)

        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

def get_CFEM(cfe_type='large', in_planes=512, out_planes=512, stride=1, scale=1, groups=8, dilation=1):
    assert cfe_type in ['large', 'normal', 'light'], 'no that type of CFEM'
    if cfe_type == 'large':
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=2)
    elif cfe_type == 'normal':
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=4)
    else:
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=8)
