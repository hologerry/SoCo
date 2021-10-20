# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops
from torch.distributed import get_world_size

from .base import BaseModel
from .mlps import Pred_Head, Proj_Head


class SoCo_C4(BaseModel):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05self.output_size22
    """

    def __init__(self, base_encoder, args):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SoCo_C4, self).__init__(base_encoder, args)

        self.contrast_num_negative = args.contrast_num_negative
        self.contrast_momentum = args.contrast_momentum
        self.contrast_temperature = args.contrast_temperature
        self.output_size = args.output_size
        self.aligned = args.aligned

        # create the encoder
        self.encoder = base_encoder(low_dim=args.feature_dim, head_type='pass', use_roi_align_on_c4=True)
        self.projector = Proj_Head()
        self.predictor = Pred_Head()

        # create the encoder_k
        self.encoder_k = base_encoder(low_dim=args.feature_dim, head_type='pass', use_roi_align_on_c4=True)
        self.projector_k = Proj_Head()

        self.roi_avg_pool = nn.AvgPool2d(self.output_size, stride=1)

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        # create the queue
        self.register_buffer("queue", torch.randn(args.feature_dim, self.contrast_num_negative))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.K = int(args.num_instances * 1. / get_world_size() / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / get_world_size() / args.batch_size * (args.start_epoch - 1))
        # print('Initial', get_rank(), self.k, self.K)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.contrast_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1
        # print('Update', get_rank(), self.k, _contrast_momentum)

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    def regression_loss_bboxs(self, vectors_q, vectors_k, correspondence):
        M, C = vectors_q.shape
        N, L, P = correspondence.shape
        assert L == P
        assert N * L == M
        vectors_q = vectors_q.view(N, L, C)
        vectors_k = vectors_k.view(N, L, C)
        # vectors_q: N, L, C
        # vectors_k: N, L, C
        vectors_k = torch.transpose(vectors_k, 1, 2)
        sim = torch.bmm(vectors_q, vectors_k)  # N, L, L
        loss = (sim * correspondence).sum(-1).sum(-1) / (correspondence.sum(-1).sum(-1) + 1e-6)
        return -2 * loss.mean()

    def forward(self, im_1, im_2, bboxs1, bboxs2, corres):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        feat_1 = self.encoder(im_1, bboxs=bboxs1)  # queries: NxC
        proj_1 = self.projector(feat_1)
        pred_1 = self.predictor(proj_1)
        pred_1 = F.normalize(pred_1, dim=1)

        feat_2 = self.encoder(im_2, bboxs=bboxs2)
        proj_2 = self.projector(feat_2)
        pred_2 = self.predictor(proj_2)
        pred_2 = F.normalize(pred_2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.encoder_k(im_1, bboxs=bboxs1)  # keys: NxC
            proj_1_ng = self.projector_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            feat_2_ng = self.encoder_k(im_2, bboxs=bboxs2)
            proj_2_ng = self.projector_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

        # compute loss
        corres_2_1 = corres.transpose(1, 2)  # transpose dim 1 dim 2, map bboxs2 to bboxs1
        loss = self.regression_loss_bboxs(pred_1, proj_2_ng, corres) + self.regression_loss_bboxs(pred_2, proj_1_ng, corres_2_1)
        return loss
