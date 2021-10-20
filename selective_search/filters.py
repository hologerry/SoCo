# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


from math import sqrt

'''
regions : array of dict
    [
        {
            'rect': (left, top, width, height),
            'labels': [...],
            'size': component_size
        },
        ...
    ]
'''

def filter_none(regions):
    rects = []
    for region in regions:
        cur_rect = region['rect']
        w, h = cur_rect[2], cur_rect[3]
        if w >= 32 and h >= 32:
            rects.append(cur_rect)
    return rects


def filter_ratio(regions, r=2):
    rects = []
    for region in regions:
        cur_rect = region['rect']
        w, h = cur_rect[2], cur_rect[3]
        if w >= 32 and h >= 32 and (1.0 / r) <= (w / h) <= r:
            rects.append(cur_rect)
    return rects


def filter_size(regions, image_size, min_ratio=0.1, max_ratio=1.0):
    rects = []
    ih, iw = image_size  # get from ss_props label, which is h, w
    img_sqrt_size = sqrt(iw * ih)
    for region in regions:
        cur_rect = region['rect']
        w, h = cur_rect[2], cur_rect[3]
        prop_sqrt_size = sqrt(w * h)
        if w >= 32 and h >= 32 and min_ratio <= (prop_sqrt_size / img_sqrt_size) <= max_ratio:
            rects.append(cur_rect)
    return rects


def filter_ratio_size(regions, image_size, r=2, min_ratio=0.1, max_ratio=1.0):
    rects = []
    ih, iw = image_size  # get from ss_props label, which is h, w
    img_sqrt_size = sqrt(iw * ih)
    for region in regions:
        cur_rect = region['rect']
        w, h = cur_rect[2], cur_rect[3]
        prop_sqrt_size = sqrt(w * h)
        if w >= 32 and h >= 32 and (1.0 / r) <= (w / h) <= r and min_ratio <= (prop_sqrt_size / img_sqrt_size) <= max_ratio:
            rects.append(cur_rect)
    return rects
