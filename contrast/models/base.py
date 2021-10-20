# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base model with: a encoder
    """

    def __init__(self, base_encoder, args):
        super(BaseModel, self).__init__()

        # create the encoders
        self.encoder = base_encoder(low_dim=args.feature_dim, head_type=args.head_type)

    def forward(self, x1, x2):
        """
        Input: x1, x2 or x, y_idx
        Output: logits, labels
        """
        raise NotImplementedError
