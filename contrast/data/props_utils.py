# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import numpy as np


def select_props(all_props, select_strategy, select_k):
    # all_props: numpy array
    if select_k > all_props.shape[0]:  # if we do not have k proposals, we just select all
        select_strategy = 'none'
    selected_proposals = None
    if select_strategy == 'none':
        selected_proposals = all_props
    elif select_strategy == 'random':
        selected_idx = np.random.choice(all_props.shape[0], size=select_k, replace=False)
        selected_proposals = all_props[selected_idx]
    elif select_strategy == 'top':
        selected_proposals = all_props[:select_k]
    elif select_strategy == 'area':
        areas = np.zeros((all_props.shape[0], ))
        for i in range(all_props.shape[0]):
            area = all_props[i][2] * all_props[i][3]
            areas[i] = area
        areas_index = areas.argsort()
        selected_proposals = all_props[areas_index[::-1]][:select_k]
    else:
        raise NotImplementedError
    return selected_proposals


def convert_props(all_props):
    all_props_np = np.array(all_props)
    if all_props_np.shape[0] == 0:
        return all_props_np
    all_props_np = all_props_np[:, :4]
    all_props_np[:, 2] = all_props_np[:, 0] + all_props_np[:, 2] - 1  # x2 = x1 + w - 1
    all_props_np[:, 3] = all_props_np[:, 1] + all_props_np[:, 3] - 1  # y2 = y1 + h - 1
    return all_props_np
