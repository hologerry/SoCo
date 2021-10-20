# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import torch.nn as nn


class FastRCNNConvFCHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ac1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ac2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ac3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.ac4 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=12544, out_features=1024, bias=True)
        self.fc_relu1 = nn.ReLU()

    def forward(self, roi_feature_map):
        conv1_out = self.conv1(roi_feature_map)
        bn1_out = self.bn1(conv1_out)
        ac1_out = self.ac1(bn1_out)

        conv2_out = self.conv2(ac1_out)
        bn2_out = self.bn2(conv2_out)
        ac2_out = self.ac2(bn2_out)

        conv3_out = self.conv3(ac2_out)
        bn3_out = self.bn3(conv3_out)
        ac3_out = self.ac3(bn3_out)

        conv4_out = self.conv4(ac3_out)
        bn4_out = self.bn4(conv4_out)
        ac4_out = self.ac4(bn4_out)

        flat = self.flatten(ac4_out)
        fc1_out = self.fc1(flat)
        fc_relu1_out = self.fc_relu1(fc1_out)

        return fc_relu1_out
