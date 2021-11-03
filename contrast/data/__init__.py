# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import os

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .dataset import (ImageFolder, ImageFolderImageAsymBboxAwareMulti3ResizeExtraJitter1,
                      ImageFolderImageAsymBboxAwareMultiJitter1,
                      ImageFolderImageAsymBboxAwareMultiJitter1Cutout,
                      ImageFolderImageAsymBboxCutout)
from .sampler import SubsetSlidingWindowSampler
from .transform import get_transform


def get_loader(aug_type, args, two_crop=False, prefix='train', return_coord=False):
    transform = get_transform(args, aug_type, args.crop, args.image_size, crop1=args.crop1,
                              cutout_prob=args.cutout_prob, cutout_ratio=args.cutout_ratio,
                              image3_size=args.image3_size, image4_size=args.image4_size)

    # dataset
    if args.zip:
        if args.dataset == 'ImageNet':
            train_ann_file = prefix + f"_{args.split_map}.txt"
            train_prefix = prefix + ".zip@/"
            if args.ss_props:
                train_props_file = prefix + f"_{args.filter_strategy}.json"
            elif args.rpn_props:
                train_props_file = f"rpn_props_nms_{args.nms_threshold}.json"
        elif args.dataset == 'COCO':   # NOTE: for coco, we use same scheme as ImageNet
            prefix = 'trainunlabeled2017'
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
            train_props_file = prefix + f"_{args.filter_strategy}.json"
        elif args.dataset == 'Object365':
            prefix = 'train'
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
            train_props_file = prefix + f"_{args.filter_strategy}.json"
        elif args.dataset == 'ImageNetObject365':
            prefix = 'train'
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
            train_props_file = prefix + f"_{args.filter_strategy}.json"
        elif args.dataset == 'OpenImage':
            prefix = 'train'
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
            train_props_file = prefix + f"_{args.filter_strategy}.json"
        elif args.dataset == 'ImageNetOpenImage':
            prefix = 'train'
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
            train_props_file = prefix + f"_{args.filter_strategy}.json"
        elif args.dataset == 'ImageNetObject365OpenImage':
            prefix = 'train'
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
            train_props_file = prefix + f"_{args.filter_strategy}.json"
        else:
            raise NotImplementedError('Dataset {} is not supported. We only support ImageNet'.format(args.dataset))

        if aug_type == 'ImageAsymBboxCutout':
            train_dataset = ImageFolderImageAsymBboxCutout(args.data_dir, train_ann_file, train_prefix,
                                                           train_props_file, image_size=args.image_size, select_strategy=args.select_strategy,
                                                           select_k=args.select_k, weight_strategy=args.weight_strategy,
                                                           jitter_prob=args.jitter_prob, jitter_ratio=args.jitter_ratio, padding_k=args.padding_k,
                                                           aware_range=args.aware_range, aware_start=args.aware_start, aware_end=args.aware_end,
                                                           max_tries=args.max_tries,
                                                           transform=transform, cache_mode=args.cache_mode,
                                                           dataset=args.dataset)

        elif aug_type == 'ImageAsymBboxAwareMultiJitter1':
            train_dataset = ImageFolderImageAsymBboxAwareMultiJitter1(args.data_dir, train_ann_file, train_prefix,
                                                                      train_props_file, image_size=args.image_size, select_strategy=args.select_strategy,
                                                                      select_k=args.select_k, weight_strategy=args.weight_strategy,
                                                                      jitter_prob=args.jitter_prob, jitter_ratio=args.jitter_ratio, padding_k=args.padding_k,
                                                                      aware_range=args.aware_range, aware_start=args.aware_start, aware_end=args.aware_end,
                                                                      max_tries=args.max_tries,
                                                                      transform=transform, cache_mode=args.cache_mode,
                                                                      dataset=args.dataset)

        elif aug_type == 'ImageAsymBboxAwareMultiJitter1Cutout':
            train_dataset = ImageFolderImageAsymBboxAwareMultiJitter1Cutout(args.data_dir, train_ann_file, train_prefix,
                                                                            train_props_file, image_size=args.image_size, select_strategy=args.select_strategy,
                                                                            select_k=args.select_k, weight_strategy=args.weight_strategy,
                                                                            jitter_prob=args.jitter_prob, jitter_ratio=args.jitter_ratio, padding_k=args.padding_k,
                                                                            aware_range=args.aware_range, aware_start=args.aware_start, aware_end=args.aware_end,
                                                                            max_tries=args.max_tries,
                                                                            transform=transform, cache_mode=args.cache_mode,
                                                                            dataset=args.dataset)

        elif aug_type == 'ImageAsymBboxAwareMulti3ResizeExtraJitter1':
            train_dataset = ImageFolderImageAsymBboxAwareMulti3ResizeExtraJitter1(args.data_dir, train_ann_file, train_prefix,
                                                                                  train_props_file, image_size=args.image_size, image3_size=args.image3_size,
                                                                                  image4_size=args.image4_size,
                                                                                  select_strategy=args.select_strategy,
                                                                                  select_k=args.select_k, weight_strategy=args.weight_strategy,
                                                                                  jitter_prob=args.jitter_prob, jitter_ratio=args.jitter_ratio, padding_k=args.padding_k,
                                                                                  aware_range=args.aware_range, aware_start=args.aware_start, aware_end=args.aware_end,
                                                                                  max_tries=args.max_tries,
                                                                                  transform=transform, cache_mode=args.cache_mode,
                                                                                  dataset=args.dataset)
        elif aug_type == 'NULL':
            train_dataset = ImageFolder(args.data_dir, train_ann_file, train_prefix,
                                        transform, two_crop=two_crop, cache_mode=args.cache_mode,
                                        dataset=args.dataset, return_coord=return_coord)
        else:
            raise NotImplementedError

    else:
        train_folder = os.path.join(args.data_dir, prefix)
        train_dataset = ImageFolder(train_folder, transform=transform, two_crop=two_crop, return_coord=return_coord)
        raise NotImplementedError

    # sampler
    indices = np.arange(dist.get_rank(), len(train_dataset), dist.get_world_size())
    if args.use_sliding_window_sampler:
        sampler = SubsetSlidingWindowSampler(indices,
                                             window_stride=args.window_stride // dist.get_world_size(),
                                             window_size=args.window_size // dist.get_world_size(),
                                             shuffle_per_epoch=args.shuffle_per_epoch)
    elif args.zip and args.cache_mode == 'part':
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSampler(train_dataset)

    # # dataloader
    return DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True, sampler=sampler, drop_last=True)
