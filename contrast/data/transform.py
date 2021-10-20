# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import numpy as np
from PIL import ImageFilter, ImageOps
from torchvision import transforms

from . import transform_ops


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_transform(args, aug_type, crop, image_size=224, crop1=0.9, cutout_prob=0.5, cutout_ratio=(0.1, 0.2),
                  image3_size=224, image4_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if aug_type == 'ImageAsymBboxCutout':
        transform_whole_img = transform_ops.WholeImageResizedParams(image_size)
        transform_img = transform_ops.RandomResizedCropParams(image_size, scale=(crop, 1.))
        transform_flip = transform_ops.RandomHorizontalFlipImageBbox()

        transform_post_1 = transform_ops.ComposeImage([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ])
        transform_post_2 = transform_ops.ComposeImage([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        transform_cutout = transform_ops.RandomCutoutInBbox(image_size, cutout_prob=cutout_prob, cutout_ratio=cutout_ratio)
        transform = (transform_whole_img, transform_img, transform_flip, transform_post_1, transform_post_2, transform_cutout)


    elif aug_type == 'ImageAsymBboxAwareMultiJitter1':
        transform_whole_img = transform_ops.WholeImageResizedParams(image_size)
        transform_img = transform_ops.RandomResizedCropParams(image_size, scale=(crop, 1.))
        transform_img_small = transform_ops.RandomResizedCropParams(image_size//2, scale=(crop, 1.))
        transform_flip_flip = transform_ops.RandomHorizontalFlipImageBboxBbox()
        transform_flip = transform_ops.RandomHorizontalFlipImageBbox()
        transform_post_1 = transform_ops.ComposeImage([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ])
        transform_post_2 = transform_ops.ComposeImage([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        transform = (transform_whole_img, transform_img, transform_img_small, transform_flip_flip, transform_flip, transform_post_1, transform_post_2)


    elif aug_type == 'ImageAsymBboxAwareMultiJitter1Cutout':
        transform_whole_img = transform_ops.WholeImageResizedParams(image_size)
        transform_img = transform_ops.RandomResizedCropParams(image_size, scale=(crop, 1.))
        transform_img_small = transform_ops.RandomResizedCropParams(image_size//2, scale=(crop, 1.))
        transform_flip_flip = transform_ops.RandomHorizontalFlipImageBboxBbox()
        transform_flip = transform_ops.RandomHorizontalFlipImageBbox()
        transform_post_1 = transform_ops.ComposeImage([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ])
        transform_post_2 = transform_ops.ComposeImage([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        transform_cutout = transform_ops.RandomCutoutInBbox(image_size, cutout_prob=cutout_prob, cutout_ratio=cutout_ratio)
        transform = (transform_whole_img, transform_img, transform_img_small, transform_flip_flip, transform_flip, transform_post_1, transform_post_2, transform_cutout)


    elif aug_type == 'ImageAsymBboxAwareMulti3ResizeExtraJitter1':
        transform_whole_img = transform_ops.WholeImageResizedParams(image_size)
        transform_img = transform_ops.RandomResizedCropParams(image_size, scale=(crop, 1.))
        transform_img_small = transform_ops.RandomResizedCropParams(image3_size, scale=(crop, 1.))
        transform_img_resize = transforms.Resize(image4_size)
        transform_flip_flip_flip = transform_ops.RandomHorizontalFlipImageBboxBboxBbox()
        transform_flip = transform_ops.RandomHorizontalFlipImageBbox()
        transform_post_1 = transform_ops.ComposeImage([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ])
        transform_post_2 = transform_ops.ComposeImage([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        transform = (transform_whole_img, transform_img, transform_img_small, transform_img_resize, transform_flip_flip_flip, transform_flip, transform_post_1, transform_post_2)


    elif aug_type == 'NULL':  # used in linear evaluation
        transform = transform_ops.Compose([
            transform_ops.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_ops.RandomHorizontalFlipCoord(),
            transforms.ToTensor(),
            normalize,
        ])

    elif aug_type == 'val':  # used in validate
        transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
    else:
        supported = '[ImageAsymBboxCutout, ImageAsymBboxAwareMultiJitter1, ImageAsymBboxAwareMultiJitter1Cutout, ImageAsymBboxAwareMulti3ResizeExtraJitter1, NULL]'
        raise NotImplementedError(f'aug_type "{aug_type}" not supported. Should in {supported}')

    return transform
