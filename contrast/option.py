# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import argparse
import os

from contrast import resnet
from contrast.util import MyHelpFormatter

model_names = sorted(name for name in resnet.__all__
                     if name.islower() and callable(resnet.__dict__[name]))


def parse_option(stage='pre_train'):
    parser = argparse.ArgumentParser(f'contrast {stage} stage', formatter_class=MyHelpFormatter)
    # develop
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    # dataset
    parser.add_argument('--data_dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--crop', type=float, default=0.2 if stage == 'pre_train' else 0.08, help='minimum crop')
    parser.add_argument('--crop1', type=float, default=1.0, help='minimum crop for view1 when asym asym crop')
    parser.add_argument('--aug', type=str, default='NULL', choices=['NULL', 'ImageAsymBboxCutout', 'ImageAsymBboxAwareMultiJitter1',
                                                                    'ImageAsymBboxAwareMultiJitter1Cutout', 'ImageAsymBboxAwareMulti3ResizeExtraJitter1'],
                        help='which augmentation to use.')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset')
    parser.add_argument('--split_map', type=str, default='map')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='cache mode: no for no cache, full for cache all data, part for only cache part of data')
    parser.add_argument('--dataset', type=str, default='ImageNet', choices=['ImageNet', 'VOC', 'COCO'], help='dataset type')
    parser.add_argument('--ann_file', type=str, default='', help='annotation file')
    parser.add_argument('--image_size', type=int, default=224, help='image crop size')
    parser.add_argument('--image3_size', type=int, default=112, help='image crop size')
    parser.add_argument('--image4_size', type=int, default=112, help='image crop size')

    parser.add_argument('--num_workers', type=int, default=4, help='num of workers per GPU to use')
    # sliding window sampler
    parser.add_argument('--swin_', type=int, default=131072, help='window size in sliding window sampler')
    parser.add_argument('--window_stride', type=int, default=16384, help='window stride in sliding window sampler')
    parser.add_argument('--use_sliding_window_sampler', action='store_true',
                        help='whether to use sliding window sampler')
    parser.add_argument('--shuffle_per_epoch', action='store_true',
                        help='shuffle indices in sliding window sampler per epoch')
    if stage == 'linear':
        parser.add_argument('--total_batch_size', type=int, default=256, help='total train batch size for all GPU')
    else:
        parser.add_argument('--batch_size', type=int, default=64, help='batch_size for single gpu')

    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=model_names,
                        help="backbone architecture")
    if stage == 'pre_train':
        parser.add_argument('--model', type=str, required=True, help='which model to use')
        parser.add_argument('--contrast_temperature', type=float, default=0.07, help='temperature in instance cls loss')
        parser.add_argument('--contrast_momentum', type=float, default=0.999,
                            help='momentum parameter used in MoCo and InstDisc')
        parser.add_argument('--contrast_num_negative', type=int, default=65536,
                            help='number of negative samples used in MoCo and InstDisc')
        parser.add_argument('--feature_dim', type=int, default=128, help='feature dimension')
        parser.add_argument('--head_type', type=str, default='mlp_head', help='choose head type')
        parser.add_argument('--lambda_img', type=float, default=0., help='loss weight of image_to_image loss')
        parser.add_argument('--lambda_cross', type=float, default=1., help='loss weight of image_to_point loss')

        # FPN default args
        parser.add_argument('--in_channels', type=list, default=[256, 512, 1024, 2048], help='FPN feature map input channels')
        parser.add_argument('--out_channels', type=int, default=256, help='FPN feature map output channels')
        parser.add_argument('--start_level', type=int, default=1, help='FPN start level')
        parser.add_argument('--end_level', type=int, default=-1, help='FPN end level')
        parser.add_argument('--add_extra_convs', type=int, default=1, help='FPN add extra convs')
        parser.add_argument('--extra_convs_on_inputs', type=int, default=0, help='FPN extra_convs_on_inputs')
        parser.add_argument('--relu_before_extra_convs', type=int, default=1, help='FPN relu_before_extra_convs')
        parser.add_argument('--no_norm_on_lateral', type=int, default=0, help='FPN no_norm_on_lateral')
        parser.add_argument('--num_outs', type=int, default=3, help='FPN num_outs, use p3~p5')

        # Head default args
        parser.add_argument('--head_in_channels', type=int, default=256, help='Head feature map input channels')
        parser.add_argument('--head_feat_channels', type=int, default=256, help='Head feature map feat channels')
        parser.add_argument('--head_stacked_convs', type=int, default=4, help='Head stacked convs')

    # optimization
    if stage == 'pre_train':
        parser.add_argument('--base_learning_rate', '--base_lr', type=float, default=0.03,
                            help='base learning when batch size = 256. final lr is determined by linear scale')
    else:
        parser.add_argument('--learning_rate', type=float, default=30, help='learning rate')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='sgd',
                        help='for optimizer choice.')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup_epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4 if stage == 'pre_train' else 0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--amp_opt_level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--start_epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    # misc
    parser.add_argument('--output_dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto_resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    if stage == 'linear':
        parser.add_argument('--pretrained_model', type=str, required=True, help="pretrained model path")
        parser.add_argument('-e', '--eval', action='store_true', help='only evaluate')
    else:
        parser.add_argument('--pretrained_model', type=str, default="", help="pretrained model path")

    # selective search
    parser.add_argument('--ss_props', action='store_true', help='use selective search propos to calculate weight map')
    parser.add_argument('--filter_strategy', type=str, default='none', help='filter strategy')
    parser.add_argument('--select_strategy', type=str, default='none', help='select strategy')
    parser.add_argument('--select_k', type=int, default=0, help='select strategy k, required when select strategy is not none')
    parser.add_argument('--weight_strategy', type=str, default='bbox', help='weight map strategy')
    # we use same select_strategy for weight map and bbox to ins
    parser.add_argument('--bbox_size_range', type=tuple, default=(32, 112, 224))
    parser.add_argument('--iou_thres', type=float, default=0.5)
    parser.add_argument('--output_size', type=int, default=7)
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--jitter_prob', type=float, default=0.5)
    parser.add_argument('--jitter_ratio', type=float, default=0.2)
    parser.add_argument('--padding_k', type=int, default=32)
    parser.add_argument('--max_tries', type=int, default=5)
    parser.add_argument('--aware_range', type=list, default=[48, 96, 192, 224, 0])
    parser.add_argument('--aware_start', type=int, default=0, help="starting from using P?")
    parser.add_argument('--aware_end', type=int, default=4, help="ending from using P?, not included")

    parser.add_argument('--cutout_prob', type=float, default=0.5)
    parser.add_argument('--cutout_ratio', type=tuple)
    parser.add_argument('--cutout_ratio_min', type=float, default=0.1)
    parser.add_argument('--cutout_ratio_max', type=float, default=0.2)

    parser.add_argument('--max_props', type=int, default=32)
    parser.add_argument('--aspect_ratio', type=float, default=3)
    parser.add_argument('--min_size_ratio', type=float, default=0.3)
    parser.add_argument('--max_size_ratio', type=float, default=0.8)

    args = parser.parse_args()

    if stage == 'pre_train':
        # Due to base command line can not directly pass bool values, we use int, 0 -> False, 1 -> True
        args.add_extra_convs = bool(args.add_extra_convs)
        args.extra_convs_on_inputs = bool(args.extra_convs_on_inputs)
        args.relu_before_extra_convs = bool(args.relu_before_extra_convs)
        args.no_norm_on_lateral = bool(args.no_norm_on_lateral)
        args.cutout_ratio = (args.cutout_ratio_min, args.cutout_ratio_max)

    return args
