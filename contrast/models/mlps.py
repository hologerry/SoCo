# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm1d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = x.unsqueeze(-1)
        x = self.bn1(x)
        x = x.squeeze(-1)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


class MLP2d_3Layer(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d_3Layer, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1(inner_dim, inner_dim)
        self.bn2 = nn.BatchNorm2d(inner_dim)
        self.relu2 = nn.ReLU(inplace=True)

        self.linear3 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.linear3(x)

        return x


def Proj_Head(in_dim=2048, inner_dim=4096, out_dim=256):
    return MLP(in_dim, inner_dim, out_dim)


def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP(in_dim, inner_dim, out_dim)


def Proj_Head2d(in_dim=2048, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


def Pred_Head2d(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)
