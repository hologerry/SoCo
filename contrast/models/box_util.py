# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------

import torch


def append_batch_index_to_bboxs_and_scale(bboxs, H, W):
    N, L, _ = bboxs.size()
    batch_index = torch.arange(N, device=bboxs.device)
    batch_index = batch_index.reshape(N, 1)
    batch_index = batch_index.repeat(1, L)
    batch_index = batch_index.reshape(N, L, 1)

    bboxs_coord = bboxs[:, :, :4].clone()  # must clone
    bboxs_coord[:, :, 0] = bboxs_coord[:, :, 0] * W  # x1
    bboxs_coord[:, :, 2] = bboxs_coord[:, :, 2] * W  # x2
    bboxs_coord[:, :, 1] = bboxs_coord[:, :, 1] * H  # y1
    bboxs_coord[:, :, 3] = bboxs_coord[:, :, 3] * H  # y2
    bboxs_with_index = torch.cat([batch_index, bboxs_coord], dim=2)
    bboxs_with_index = bboxs_with_index.reshape(N*L, 5)
    return bboxs_with_index
