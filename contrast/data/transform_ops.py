# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import math
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
from torchvision.transforms import functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def crop_tensor(img_tensor, top, left, height, width):
    return img_tensor[:, :, left:left+width, top:top+height]


def resize_tensor(img_tensor, size, interpolation='bilinear'):
    return TF.interpolate(img_tensor, size, mode=interpolation, align_corners=False)


def resized_crop_tensor(img_tensor, top, left, height, width, size, interpolation='bilinear'):
    """
    tensor version of F.resized_crop
    """
    assert isinstance(img_tensor, torch.Tensor)
    img_tensor = crop_tensor(img_tensor, top, left, height, width)
    img_tensor = resize_tensor(img_tensor, size, interpolation)
    return img_tensor



class ComposeImage(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        coord = None
        for t in self.transforms:
            if 'RandomResizedCropCoord' in t.__class__.__name__:
                img, coord = t(img)
            elif 'FlipCoord' in t.__class__.__name__:
                img, coord = t(img, coord)
            else:
                img = t(img)
        return img, coord

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class WholeImageResizedParams(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation

    @staticmethod
    def get_params(img):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)

        return 0, 0, height, width, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        params = self.get_params(img)
        params_np = np.array(params)
        i, j, h, w, _, _ = params

        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), params_np

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string





class RandomResizedCropParams(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        params = self.get_params(img, self.scale, self.ratio)
        params_np = np.array(params)
        i, j, h, w, _, _ = params

        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), params_np

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string





class RandomHorizontalFlipImageBbox(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxs, params):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bboxs (torch tensor): [[y1, x1, y2, x2]] in [0, 1]
            weight (torch tensor): (1, h, w), in [0, 1]
            params (numpy array), i, j, h(crop image), w(crop image), height(raw image), width(raw image)
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            bboxs_new = bboxs.clone()
            bboxs_new[:, 0] = 1.0 - bboxs[:, 2]  # x1 = x2
            bboxs_new[:, 2] = 1.0 - bboxs[:, 0]  # x2 = x1
            # change x, keep y, w, h
            params_new = np.copy(params)
            params_new[1] = params[5] - params[3] - params[1]
            return F.hflip(img), bboxs_new, params_new
        return img, bboxs, params

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)




class RandomCutoutInBbox(object):
    """RandomCutout in bboxs
    """
    def __init__(self, size, cutout_prob, cutout_ratio=(0.1, 0.2)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        assert isinstance(cutout_ratio, tuple)
        self.cutout_prob = cutout_prob
        self.cutout_ratio = cutout_ratio
        self.width = self.size[0]
        self.height = self.size[1]
    
    def __call__(self, img, resized_bboxs, view_size):
        """ img is tensor
            resized_bboxs is in (0, 1), need to rescale to pixel wise size with view_size
        """
        new_img = img.clone()
        for bbox in resized_bboxs:
            cutout_r = random.random()
            if cutout_r < self.cutout_prob:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1 = x1 * view_size[0]
                x2 = x2 * view_size[0]
                y1 = y1 * view_size[1]
                y2 = y2 * view_size[1]
                bbox_w = x2 - x1 + 1
                bbox_h = y2 - y1 + 1
                bbox_area = bbox_w * bbox_h

                target_area = random.uniform(*self.cutout_ratio) * bbox_area

                w = int(round(math.sqrt(target_area)))
                h = w
                center_cut_x = random.randint(x1, x2)
                center_cut_y = random.randint(y1, y2)
                cut_x1 = max(center_cut_x - w // 2, 0)
                cut_x2 = min(center_cut_x + w // 2, self.width)
                cut_y1 = max(center_cut_y - h // 2, 0)
                cut_y2 = min(center_cut_y + h // 2, self.height)

                # img is tensor 3, H, W
                new_img[:, cut_y1:cut_y2+1, cut_x1:cut_x2+1] = 0.0
        return new_img


class RandomHorizontalFlipImageBboxBbox(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxs, bboxs_p, params):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bboxs (torch tensor): [[y1, x1, y2, x2]] in [0, 1]
            bboxs_p (torch tensor): [[y1, x1, y2, x2]] in [0, 1]  another bboxs for current img
            weight (torch tensor): (1, h, w), in [0, 1]
            params (numpy array), i, j, h(crop image), w(crop image), height(raw image), width(raw image)
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            bboxs_new = bboxs.clone()
            bboxs_new[:, 0] = 1.0 - bboxs[:, 2]  # x1 = x2
            bboxs_new[:, 2] = 1.0 - bboxs[:, 0]  # x2 = x1

            bboxs_p_new = bboxs_p.clone()
            bboxs_p_new[:, 0] = 1.0 - bboxs_p[:, 2]  # x1 = x2
            bboxs_p_new[:, 2] = 1.0 - bboxs_p[:, 0]  # x2 = x1
            # change x, keep y, w, h
            params_new = np.copy(params)
            params_new[1] = params[5] - params[3] - params[1]
            return F.hflip(img), bboxs_new, bboxs_p_new, params_new
        return img, bboxs, bboxs_p, params

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalFlipImageBboxBboxBbox(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxs, bboxs_p, bboxs_q, params):
        """
        Args:
            img (PIL Image): Image to be flipped.
            bboxs (torch tensor): [[y1, x1, y2, x2]] in [0, 1]
            bboxs_p (torch tensor): [[y1, x1, y2, x2]] in [0, 1]  another bboxs for current img
            bboxs_q (torch tensor): [[y1, x1, y2, x2]] in [0, 1]  another bboxs for current img
            weight (torch tensor): (1, h, w), in [0, 1]
            params (numpy array), i, j, h(crop image), w(crop image), height(raw image), width(raw image)
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            bboxs_new = bboxs.clone()
            bboxs_new[:, 0] = 1.0 - bboxs[:, 2]  # x1 = x2
            bboxs_new[:, 2] = 1.0 - bboxs[:, 0]  # x2 = x1

            bboxs_p_new = bboxs_p.clone()
            bboxs_p_new[:, 0] = 1.0 - bboxs_p[:, 2]  # x1 = x2
            bboxs_p_new[:, 2] = 1.0 - bboxs_p[:, 0]  # x2 = x1

            bboxs_q_new = bboxs_q.clone()
            bboxs_q_new[:, 0] = 1.0 - bboxs_q[:, 2]  # x1 = x2
            bboxs_q_new[:, 2] = 1.0 - bboxs_q[:, 0]  # x2 = x1
            # change x, keep y, w, h
            params_new = np.copy(params)
            params_new[1] = params[5] - params[3] - params[1]
            return F.hflip(img), bboxs_new, bboxs_p_new, bboxs_q_new, params_new
        return img, bboxs, bboxs_p, bboxs_q, params

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCropCoord(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)
        coord = torch.Tensor([float(j) / (width - 1), float(i) / (height - 1),
                              float(j + w - 1) / (width - 1), float(i + h - 1) / (height - 1)])
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), coord

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomHorizontalFlipCoord(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            return F.hflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
