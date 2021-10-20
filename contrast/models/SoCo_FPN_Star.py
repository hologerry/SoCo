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
from .box_util import append_batch_index_to_bboxs_and_scale
from .fast_rcnn_conv_fc_head import FastRCNNConvFCHead
from .fpn import FPN
from .mlps import Pred_Head, Proj_Head


class SoCo_FPN_Star(BaseModel):
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
        Just cleaned unused forward tensors
        """
        super(SoCo_FPN_Star, self).__init__(base_encoder, args)

        self.contrast_num_negative = args.contrast_num_negative
        self.contrast_momentum = args.contrast_momentum
        self.contrast_temperature = args.contrast_temperature
        self.output_size = args.output_size
        self.aligned = args.aligned

        norm_cfg = dict(type='BN', requires_grad=True)

        # align to detectron2 !
        # create the encoder
        self.encoder = base_encoder(low_dim=args.feature_dim, head_type='multi_layer')
        self.neck = FPN(in_channels=args.in_channels, out_channels=args.out_channels, num_outs=args.num_outs, start_level=args.start_level,
                        end_level=args.end_level, add_extra_convs=args.add_extra_convs, extra_convs_on_inputs=args.extra_convs_on_inputs,
                        relu_before_extra_convs=args.relu_before_extra_convs, norm_cfg=norm_cfg)
        self.head = FastRCNNConvFCHead()
        self.projector = Proj_Head(in_dim=1024)  # head channel
        self.predictor = Pred_Head()

        # create the encoder_k
        self.encoder_k = base_encoder(low_dim=args.feature_dim, head_type='multi_layer')
        self.neck_k = FPN(in_channels=args.in_channels, out_channels=args.out_channels, num_outs=args.num_outs, start_level=args.start_level,
                          end_level=args.end_level, add_extra_convs=args.add_extra_convs, extra_convs_on_inputs=args.extra_convs_on_inputs,
                          relu_before_extra_convs=args.relu_before_extra_convs, norm_cfg=norm_cfg)
        self.head_k = FastRCNNConvFCHead()
        self.projector_k = Proj_Head(in_dim=1024)  # head channel

        self.roi_avg_pool = nn.AvgPool2d(self.output_size, stride=1)

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.neck.parameters(), self.neck_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.neck)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.neck_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.head_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

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

        for param_q, param_k in zip(self.neck.parameters(), self.neck_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.head.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)


    def regression_loss_bboxs_aware(self, vectors_q, vectors_k, correspondence):
        N, M, C = vectors_q.shape  # N, P * L, C
        N, M1, M2 = correspondence.shape  # N, P * L, P * L
        assert M == M1 == M2
        # vectors_q: N, L, C
        # vectors_k: N, L, C
        vectors_k = torch.transpose(vectors_k, 1, 2)  # N, C, P * L
        sim = torch.bmm(vectors_q, vectors_k)  # N, P * L, P * L
        loss = (sim * correspondence).sum(-1).sum(-1) / (correspondence.sum(-1).sum(-1) + 1e-6)
        return -2 * loss.mean()

    def roi_align_feature_map(self, feature_map, bboxs):
        feature_map = feature_map.type(dtype=bboxs.dtype)  # feature map will be convert to HalfFloat in favor of amp
        N, C, H, W = feature_map.shape
        N, L, _ = bboxs.shape

        output_size = (self.output_size, self.output_size)

        bboxs_q_with_batch_index = append_batch_index_to_bboxs_and_scale(bboxs, H, W)
        aligned_features = tvops.roi_align(input=feature_map, boxes=bboxs_q_with_batch_index, output_size=output_size, aligned=self.aligned)
        # N*L, C, output_size, output_size
        return aligned_features

    def forward(self, im_1, im_2, im_3, im_4, bboxs1_12, bboxs1_13, bboxs1_14, bboxs2, bboxs3, bboxs4, corres_12, corres_13, corres_14):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        N, L, _ = bboxs1_12.shape
        feats_1 = self.encoder(im_1)
        fpn_feats_1 = self.neck(feats_1)  # p2, p3, p4, p5, num_outs = 4
        assert len(fpn_feats_1) == 4

        preds_1_12 = [None] * len(fpn_feats_1)
        for i, feat_1_12 in enumerate(fpn_feats_1):
            feat_roi_1_12 = self.roi_align_feature_map(feat_1_12, bboxs1_12)
            feat_vec_1_12 = self.head(feat_roi_1_12)
            proj_1_12 = self.projector(feat_vec_1_12)
            pred_1_12 = self.predictor(proj_1_12)
            pred_1_12 = F.normalize(pred_1_12, dim=1)  # N * L, C
            pred_1_12 = pred_1_12.reshape((N, L, -1))  # N, L, C
            preds_1_12[i] = pred_1_12

        preds_1_12 = torch.cat(preds_1_12, dim=1)  # N, P * L, C


        preds_1_13 = [None] * len(fpn_feats_1)
        for i, feat_1_13 in enumerate(fpn_feats_1):
            feat_roi_1_13 = self.roi_align_feature_map(feat_1_13, bboxs1_13)
            feat_vec_1_13 = self.head(feat_roi_1_13)
            proj_1_13 = self.projector(feat_vec_1_13)
            pred_1_13 = self.predictor(proj_1_13)
            pred_1_13 = F.normalize(pred_1_13, dim=1)  # N * L, C
            pred_1_13 = pred_1_13.reshape((N, L, -1))  # N, L, C
            preds_1_13[i] = pred_1_13

        preds_1_13 = torch.cat(preds_1_13, dim=1)  # N, P * L, C


        preds_1_14 = [None] * len(fpn_feats_1)
        for i, feat_1_14 in enumerate(fpn_feats_1):
            feat_roi_1_14 = self.roi_align_feature_map(feat_1_14, bboxs1_14)
            feat_vec_1_14 = self.head(feat_roi_1_14)
            proj_1_14 = self.projector(feat_vec_1_14)
            pred_1_14 = self.predictor(proj_1_14)
            pred_1_14 = F.normalize(pred_1_14, dim=1)  # N * L, C
            pred_1_14 = pred_1_14.reshape((N, L, -1))  # N, L, C
            preds_1_14[i] = pred_1_14

        preds_1_14 = torch.cat(preds_1_14, dim=1)  # N, P * L, C


        feats_2 = self.encoder(im_2)
        fpn_feats_2 = self.neck(feats_2)  # p2, p3, p4, p5, num_outs = 4
        assert len(fpn_feats_2) == 4

        preds_2 = [None] * len(fpn_feats_2)
        for i, feat_2 in enumerate(fpn_feats_2):
            feat_roi_2 = self.roi_align_feature_map(feat_2, bboxs2)
            feat_vec_2 = self.head(feat_roi_2)
            proj_2 = self.projector(feat_vec_2)
            pred_2 = self.predictor(proj_2)
            pred_2 = F.normalize(pred_2, dim=1)
            pred_2 = pred_2.reshape((N, L, -1))  # N, L, C
            preds_2[i] = pred_2

        preds_2 = torch.cat(preds_2, dim=1)  # N, P * L, C


        feats_3 = self.encoder(im_3)
        fpn_feats_3 = self.neck(feats_3)  # p2, p3, p4, p5, num_outs = 4
        assert len(fpn_feats_3) == 4

        preds_3 = [None] * len(fpn_feats_3)
        for i, feat_3 in enumerate(fpn_feats_3):
            feat_roi_3 = self.roi_align_feature_map(feat_3, bboxs3)
            feat_vec_3 = self.head(feat_roi_3)
            proj_3 = self.projector(feat_vec_3)
            pred_3 = self.predictor(proj_3)
            pred_3 = F.normalize(pred_3, dim=1)
            pred_3 = pred_3.reshape((N, L, -1))  # N, L, C
            preds_3[i] = pred_3

        preds_3 = torch.cat(preds_3, dim=1)  # N, P * L, C


        feats_4 = self.encoder(im_4)
        fpn_feats_4 = self.neck(feats_4)  # p2, p3, p4, p5, num_outs = 4
        assert len(fpn_feats_4) == 4

        preds_4 = [None] * len(fpn_feats_4)
        for i, feat_4 in enumerate(fpn_feats_4):
            feat_roi_4 = self.roi_align_feature_map(feat_4, bboxs4)
            feat_vec_4 = self.head(feat_roi_4)
            proj_4 = self.projector(feat_vec_4)
            pred_4 = self.predictor(proj_4)
            pred_4 = F.normalize(pred_4, dim=1)
            pred_4 = pred_4.reshape((N, L, -1))  # N, L, C
            preds_4[i] = pred_4

        preds_4 = torch.cat(preds_4, dim=1)  # N, P * L, C


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feats_1_ng = self.encoder_k(im_1)
            fpn_feats_1_ng = self.neck_k(feats_1_ng)  # p2, p3, p4, p5, num_outs = 4
            assert len(fpn_feats_1_ng) == 4
            projs_1_12_ng = [None] * len(fpn_feats_1_ng)
            for i, feat_1_12_ng in enumerate(fpn_feats_1_ng):
                feat_roi_1_12_ng = self.roi_align_feature_map(feat_1_12_ng, bboxs1_12)
                feat_vec_1_12_ng = self.head_k(feat_roi_1_12_ng)
                proj_1_12_ng = self.projector_k(feat_vec_1_12_ng)
                proj_1_12_ng = F.normalize(proj_1_12_ng, dim=1)
                proj_1_12_ng = proj_1_12_ng.reshape((N, L, -1))
                projs_1_12_ng[i] = proj_1_12_ng

            projs_1_12_ng = torch.cat(projs_1_12_ng, dim=1)  # N, P * L, C


            projs_1_13_ng = [None] * len(fpn_feats_1_ng)
            for i, feat_1_13_ng in enumerate(fpn_feats_1_ng):
                feat_roi_1_13_ng = self.roi_align_feature_map(feat_1_13_ng, bboxs1_13)
                feat_vec_1_13_ng = self.head_k(feat_roi_1_13_ng)
                proj_1_13_ng = self.projector_k(feat_vec_1_13_ng)
                proj_1_13_ng = F.normalize(proj_1_13_ng, dim=1)
                proj_1_13_ng = proj_1_13_ng.reshape((N, L, -1))
                projs_1_13_ng[i] = proj_1_13_ng

            projs_1_13_ng = torch.cat(projs_1_13_ng, dim=1)  # N, P * L, C


            projs_1_14_ng = [None] * len(fpn_feats_1_ng)
            for i, feat_1_14_ng in enumerate(fpn_feats_1_ng):
                feat_roi_1_14_ng = self.roi_align_feature_map(feat_1_14_ng, bboxs1_14)
                feat_vec_1_14_ng = self.head_k(feat_roi_1_14_ng)
                proj_1_14_ng = self.projector_k(feat_vec_1_14_ng)
                proj_1_14_ng = F.normalize(proj_1_14_ng, dim=1)
                proj_1_14_ng = proj_1_14_ng.reshape((N, L, -1))
                projs_1_14_ng[i] = proj_1_14_ng

            projs_1_14_ng = torch.cat(projs_1_14_ng, dim=1)  # N, P * L, C


            feats_2_ng = self.encoder_k(im_2)
            fpn_feats_2_ng = self.neck_k(feats_2_ng)  # p2, p3, p4, p5, num_outs = 4
            assert len(fpn_feats_2_ng) == 4
            projs_2_ng = [None] * len(fpn_feats_2_ng)
            for i, feat_2_ng in enumerate(fpn_feats_2_ng):
                feat_roi_2_ng = self.roi_align_feature_map(feat_2_ng, bboxs2)
                feat_vec_2_ng = self.head_k(feat_roi_2_ng)
                proj_2_ng = self.projector_k(feat_vec_2_ng)
                proj_2_ng = F.normalize(proj_2_ng, dim=1)
                proj_2_ng = proj_2_ng.reshape((N, L, -1))
                projs_2_ng[i] = proj_2_ng

            projs_2_ng = torch.cat(projs_2_ng, dim=1)  # N, P * L, C


            feats_3_ng = self.encoder_k(im_3)
            fpn_feats_3_ng = self.neck_k(feats_3_ng)  # p2, p3, p4, p5, num_outs = 4
            assert len(fpn_feats_3_ng) == 4
            projs_3_ng = [None] * len(fpn_feats_3_ng)
            for i, feat_3_ng in enumerate(fpn_feats_3_ng):
                feat_roi_3_ng = self.roi_align_feature_map(feat_3_ng, bboxs3)
                feat_vec_3_ng = self.head_k(feat_roi_3_ng)
                proj_3_ng = self.projector_k(feat_vec_3_ng)
                proj_3_ng = F.normalize(proj_3_ng, dim=1)
                proj_3_ng = proj_3_ng.reshape((N, L, -1))
                projs_3_ng[i] = proj_3_ng

            projs_3_ng = torch.cat(projs_3_ng, dim=1)  # N, P * L, C


            feats_4_ng = self.encoder_k(im_4)
            fpn_feats_4_ng = self.neck_k(feats_4_ng)  # p2, p3, p4, p5, num_outs = 4
            assert len(fpn_feats_4_ng) == 4
            projs_4_ng = [None] * len(fpn_feats_4_ng)
            for i, feat_4_ng in enumerate(fpn_feats_4_ng):
                feat_roi_4_ng = self.roi_align_feature_map(feat_4_ng, bboxs4)
                feat_vec_4_ng = self.head_k(feat_roi_4_ng)
                proj_4_ng = self.projector_k(feat_vec_4_ng)
                proj_4_ng = F.normalize(proj_4_ng, dim=1)
                proj_4_ng = proj_4_ng.reshape((N, L, -1))
                projs_4_ng[i] = proj_4_ng

            projs_4_ng = torch.cat(projs_4_ng, dim=1)  # N, P * L, C

        # compute loss
        corres_12_2to1 = corres_12.transpose(1, 2)  # transpose dim 1 dim 2, map bboxs2 to bboxs1
        corres_13_3to1 = corres_13.transpose(1, 2)  # transpose dim 1 dim 2, map bboxs3 to bboxs1
        corres_14_4to1 = corres_14.transpose(1, 2)  # transpose dim 1 dim 2, map bboxs4 to bboxs1
        loss_bbox_aware_12 = self.regression_loss_bboxs_aware(preds_1_12, projs_2_ng, corres_12) + self.regression_loss_bboxs_aware(preds_2, projs_1_12_ng, corres_12_2to1)
        loss_bbox_aware_13 = self.regression_loss_bboxs_aware(preds_1_13, projs_3_ng, corres_13) + self.regression_loss_bboxs_aware(preds_3, projs_1_13_ng, corres_13_3to1)
        loss_bbox_aware_14 = self.regression_loss_bboxs_aware(preds_1_14, projs_4_ng, corres_14) + self.regression_loss_bboxs_aware(preds_4, projs_1_14_ng, corres_14_4to1)

        loss = loss_bbox_aware_12 + loss_bbox_aware_13 + loss_bbox_aware_14

        return loss
