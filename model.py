from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init


# Deep Residual Learning for Image Recognition. arXiv:1512.03385
#only minor change for setting color channel, num_class
"""
Reference:
    [1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
    image recognition,” IEEE Conference on Computer Vision and Pattern
    Recognition, pp. 770–778, 2016.
"""


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',BN = True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if BN:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                if BN:
                    self.bn3 = nn.BatchNorm2d(self.expansion * planes)
                else:
                    self.bn3 = nn.Sequential()

                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     self.bn3
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, color_channel = 3,num_classes=10,in_planes=16,BN=True,use_norm=False):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(color_channel, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if BN:
            self.bn1 = nn.BatchNorm2d(in_planes)
        else:
            self.bn1 = nn.Sequential()
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1,BN=BN)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2,BN=BN)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2,BN=BN)
        self.avg_pool=nn.AvgPool2d(8)

        if use_norm:
            self.linear = NormedLinear(in_planes*4, num_classes)
        else:
            self.linear = nn.Linear(in_planes*4, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride,BN):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,BN=BN))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out=self.avg_pool(out)
        feature = out.view(out.size(0), -1)
        logits = self.linear(feature)
        return logits

    def get_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out=self.avg_pool(out)
        feature = out.view(out.size(0), -1)
        return feature

def Resnet20(color_channel,num_classes=10,BN=True,use_norm=False):
    num_res_blocks = int((20 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],
                  color_channel,num_classes=num_classes,BN=BN,use_norm=use_norm)

def Resnet32(color_channel,num_classes=10,BN=True,use_norm=False):
    num_res_blocks = int((32 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],
                  color_channel,num_classes=num_classes,BN=BN,use_norm=use_norm)

def Resnet44(color_channel,num_classes=10,BN=True,use_norm=False):
    num_res_blocks = int((44 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],
                  color_channel,num_classes=num_classes,BN=BN,use_norm=use_norm)

def Resnet56(color_channel,num_classes=10,BN=True,use_norm=False):
    num_res_blocks = int((56 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],
                  color_channel,num_classes=num_classes,BN=BN,use_norm=use_norm)

def Resnet110(color_channel,num_classes=10,BN=True,use_norm=False):
    num_res_blocks = int((110 - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],
                  color_channel,num_classes=num_classes,BN=BN,use_norm=use_norm)

def Resnet_n(layer,color_channel,num_classes=10,BN=True,use_norm=False):
    num_res_blocks = int((layer - 2) / 6)
    return ResNet(BasicBlock, [num_res_blocks, num_res_blocks, num_res_blocks],
                  color_channel,num_classes=num_classes,BN=BN,use_norm=use_norm)










