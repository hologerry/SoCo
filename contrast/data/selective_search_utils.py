# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import numpy as np


def append_prop_id(image_proposals):
    prop_ids = np.arange(1, image_proposals.shape[0]+1)  # we start from 1, refer to clip_bboxs, which set
    prop_ids = np.reshape(prop_ids, (image_proposals.shape[0], 1))
    image_proposals_with_prop_id = np.concatenate((image_proposals, prop_ids), axis=1)

    return image_proposals_with_prop_id
