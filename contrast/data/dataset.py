import io
import json
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
from PIL import Image

from .bboxs_utils import (cal_overlap_params, clip_bboxs,
                          get_common_bboxs_ids, get_overlap_props, get_correspondence_matrix,
                          pad_bboxs_with_common, bboxs_to_tensor, resize_bboxs,
                          assign_bboxs_to_feature_map, get_aware_correspondence_matrix, jitter_props)
from .props_utils import select_props, convert_props
from .selective_search_utils import append_prop_id
from .zipreader import ZipReader, is_zip_path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_dataset_with_ann(ann_file, img_prefix, extensions, dataset='ImageNet'):
    images = []

    with open(ann_file, "r") as f:
        contents = f.readlines()
        for line_str in contents:
            path_contents = [c for c in line_str.split()]
            im_file_name = path_contents[0]
            class_index = int(path_contents[1])

            assert str.lower(os.path.splitext(im_file_name)[-1]) in extensions
            item = (os.path.join(img_prefix, im_file_name), class_index)

            images.append(item)

    return images


def make_props_dataset_with_ann(ann_file, props_file, select_strategy, select_k, dataset='ImageNet', rpn_props=False, rpn_score_thres=0.5):
    with open(props_file, "r") as f:
        props_dict = json.load(f)
    # make ImageNet or VOC dataset
    with open(ann_file, "r") as f:
        contents = f.readlines()
        images_props = [None] * len(contents)
        for i, line_str in enumerate(contents):
            path_contents = [c for c in line_str.split('\t')]
            im_file_name = path_contents[0]
            basename = os.path.basename(im_file_name).split('.')[0]
            all_props = props_dict[basename]
            converted_props = convert_props(all_props)
            images_props[i] = converted_props  # keep all propos

        del contents
    del props_dict
    return images_props


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, ann_file='', img_prefix='', transform=None, target_transform=None,
                 cache_mode="no", dataset='ImageNet'):
        # image folder mode
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
        # zip mode
        else:
            samples = make_dataset_with_ann(os.path.join(root, ann_file),
                                            os.path.join(root, img_prefix),
                                            extensions,
                                            dataset)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.labels = [y_1k for _, y_1k in samples]
        self.classes = list(set(self.labels))

        self.transform = transform
        self.target_transform = target_transform

        self.cache_mode = cache_mode
        if self.cache_mode != "no":
            self.init_cache()

    def init_cache(self):
        assert self.cache_mode in ["part", "full"]
        n_sample = len(self.samples)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        samples_bytes = [None for _ in range(n_sample)]
        start_time = time.time()
        for index in range(n_sample):
            if index % (n_sample//10) == 0:
                t = time.time() - start_time
                logger = logging.getLogger(__name__)
                logger.info(f'cached {index}/{n_sample} takes {t:.2f}s per block')
                start_time = time.time()
            path, target = self.samples[index]
            if self.cache_mode == "full" or index % world_size == global_rank:
                samples_bytes[index] = (ZipReader.read(path), target)
            else:
                samples_bytes[index] = (path, target)
        self.samples = samples_bytes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class DatasetFolderProps(data.Dataset):
    def __init__(self, root, loader, extensions, ann_file='', img_prefix='', train_props_file='',
                 select_strategy='', select_k=0,
                 transform=None, target_transform=None,
                 cache_mode="no", dataset='ImageNet', rpn_props=False, rpn_score_thres=0.5):
        # image folder mode
        if ann_file == '':
            _, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx, extensions)
        # zip mode
        else:
            samples = make_dataset_with_ann(os.path.join(root, ann_file),
                                            os.path.join(root, img_prefix),
                                            extensions,
                                            dataset)
            samples_props = make_props_dataset_with_ann(os.path.join(root, ann_file),
                                                        os.path.join(root, train_props_file),
                                                        select_strategy, select_k,
                                                        dataset=dataset, rpn_props=rpn_props, rpn_score_thres=rpn_score_thres)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        if len(samples_props) == 0:
            raise(RuntimeError("Not found the proposal files"))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.samples_props = samples_props
        self.labels = [y_1k for _, y_1k in samples]
        self.classes = list(set(self.labels))

        self.transform = transform
        self.target_transform = target_transform

        self.cache_mode = cache_mode
        if self.cache_mode != "no":
            self.init_cache()

    def init_cache(self):
        assert self.cache_mode in ["part", "full"]
        n_sample = len(self.samples)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        samples_bytes = [None for _ in range(n_sample)]
        start_time = time.time()
        for index in range(n_sample):
            if index % (n_sample//10) == 0:
                t = time.time() - start_time
                logger = logging.getLogger(__name__)
                logger.info(f'cached {index}/{n_sample} takes {t:.2f}s per block')
                start_time = time.time()
            path, target = self.samples[index]
            if self.cache_mode == "full" or index % world_size == global_rank:
                samples_bytes[index] = (ZipReader.read(path), target)
            else:
                samples_bytes[index] = (path, target)
        self.samples = samples_bytes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
    import accimage  # type: ignore
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_img_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, ann_file='', img_prefix='', transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no", dataset='ImageNet',
                 two_crop=False, return_coord=False):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          ann_file=ann_file, img_prefix=img_prefix,
                                          transform=transform, target_transform=target_transform,
                                          cache_mode=cache_mode, dataset=dataset)
        self.imgs = self.samples
        self.two_crop = two_crop
        self.return_coord = return_coord

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            if isinstance(self.transform, tuple) and len(self.transform) == 2:
                img = self.transform[0](image)
            else:
                img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            if isinstance(self.transform, tuple) and len(self.transform) == 2:
                img2 = self.transform[1](image)
            else:
                img2 = self.transform(image)

        if self.return_coord:
            assert isinstance(img, tuple)
            img, coord = img

            if self.two_crop:
                img2, coord2 = img2
                return img, img2, coord, coord2, index, target
            else:
                return img, coord, index, target
        else:
            if isinstance(img, tuple):
                img, coord = img

            if self.two_crop:
                if isinstance(img2, tuple):
                    img2, coord2 = img2
                return img, img2, index, target
            else:
                return img, index, target


class ImageFolderImageAsymBboxCutout(DatasetFolderProps):
    def __init__(self, root, ann_file='', img_prefix='', train_props_file='',
                 image_size=0, select_strategy='', select_k=0, weight_strategy='',
                 jitter_ratio=0.0, padding_k='', aware_range=[], aware_start=0, aware_end=4,
                 max_tries=0,
                 transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no", dataset='ImageNet'):
        super(ImageFolderImageAsymBboxCutout, self).__init__(root, loader, IMG_EXTENSIONS,
                                                   ann_file=ann_file, img_prefix=img_prefix,
                                                   train_props_file=train_props_file,
                                                   select_strategy=select_strategy, select_k=select_k,
                                                   transform=transform, target_transform=target_transform,
                                                   cache_mode=cache_mode, dataset=dataset)
        self.imgs = self.samples
        self.props = self.samples_props
        self.select_strategy = select_strategy
        self.select_k = select_k
        self.weight_strategy = weight_strategy
        self.jitter_ratio = jitter_ratio
        self.padding_k = padding_k
        self.view_size = (image_size, image_size)
        self.debug = False
        self.max_tries = max_tries
        self.least_common = max(self.padding_k // 2, 1)
        self.aware_range = aware_range
        assert len(self.aware_range) == 5, 'Must give P2 P3 P4 P5 P6 size range'
        self.aware_start = aware_start  # starting from 0 means use p2
        self.aware_end = aware_end  # end, if use P6 might be 5

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        image_size = image.size
        image_proposals = self.props[index]  # for cur image, numpy array type, [[x1, y1, x2, y2]] x2 = x1 + w - 1
        if image_proposals.shape[0] == 0:  # if no proposals, insert one single proposal, the whole raw image
            image_proposals = np.array([[0, 0, image_size[0] - 1, image_size[1] - 1]])

        image_proposals_w_id = append_prop_id(image_proposals)  # start from 1

        assert len(self.transform) == 6
        # transform = (transform_whole_img, transform_img, transform_flip, transform_post_1, transform_post_2, transform_cutout)

        tries = 0
        least_common = self.least_common

        while tries < self.max_tries:
            img, params = self.transform[0](image)  # whole image resize
            img2, params2 = self.transform[1](image)  # random crop resize

            params_overlap = cal_overlap_params(params, params2)
            overlap_props = get_overlap_props(image_proposals_w_id, params_overlap)
            selected_image_props = select_props(overlap_props, self.select_strategy, self.select_k)  # check paras are

            # TODO: ensure clipped bboxs width and height are greater than 32
            if selected_image_props.shape[0] >= least_common:  # ok
                break
            least_common = max(least_common // 2, 1)
            tries += 1

        bboxs = clip_bboxs(selected_image_props, params[0], params[1], params[2], params[3])
        bboxs2 = clip_bboxs(selected_image_props, params2[0], params2[1], params2[2], params2[3])
        common_bboxs_ids = get_common_bboxs_ids(bboxs, bboxs2)


        pad1 = self.padding_k - bboxs.shape[0]
        if pad1 > 0:
            # pad_bboxs = jitter_bboxs(bboxs, common_bboxs_ids, self.jitter_ratio, pad1, params[2], params[3])
            pad_bboxs = pad_bboxs_with_common(bboxs, common_bboxs_ids, self.jitter_ratio, pad1, params[2], params[3])
            bboxs = np.concatenate([bboxs, pad_bboxs], axis=0)
        pad2 = self.padding_k - bboxs2.shape[0]
        if pad2 > 0:
            # pad_bboxs2 = jitter_bboxs(bboxs2, common_bboxs_ids, self.jitter_ratio, pad2, params2[2], params2[3])
            pad_bboxs2 = pad_bboxs_with_common(bboxs2, common_bboxs_ids, self.jitter_ratio, pad2, params2[2], params2[3])
            bboxs2 = np.concatenate([bboxs2, pad_bboxs2], axis=0)
        correspondence = get_correspondence_matrix(bboxs, bboxs2)

        resized_bboxs = resize_bboxs(bboxs, params[2], params[3], self.view_size)
        resized_bboxs2 = resize_bboxs(bboxs2, params2[2], params2[3], self.view_size)
        resized_bboxs = resized_bboxs.astype(int)
        resized_bboxs2 = resized_bboxs2.astype(int)

        bboxs = bboxs_to_tensor(bboxs, params)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs2 = bboxs_to_tensor(bboxs2, params2)  # x1y1x2y2 -> x1y1x2y2 (0, 1)


        img, bboxs, params = self.transform[2](img, bboxs, params)  # flip
        img2, bboxs2, params2 = self.transform[2](img2, bboxs2, params2)

        img1_1x = self.transform[3](img)  # color
        img2_1x = self.transform[4](img2)  # color

        img2_1x_cut = self.transform[5](img2_1x, resized_bboxs2)  # cutout

        return img1_1x, img2_1x_cut, bboxs, bboxs2, correspondence, index, target


class ImageFolderImageAsymBboxAwareMultiJitter1(DatasetFolderProps):
    def __init__(self, root, ann_file='', img_prefix='', train_props_file='',
                 image_size=0, select_strategy='', select_k=0, weight_strategy='',
                 jitter_prob=0.0, jitter_ratio=0.0,
                 padding_k='', aware_range=[], aware_start=0, aware_end=4, max_tries=0,
                 transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no", dataset='ImageNet'):
        super(ImageFolderImageAsymBboxAwareMultiJitter1, self).__init__(root, loader, IMG_EXTENSIONS,
                                                                   ann_file=ann_file, img_prefix=img_prefix,
                                                                   train_props_file=train_props_file,
                                                                   select_strategy=select_strategy, select_k=select_k,
                                                                   transform=transform, target_transform=target_transform,
                                                                   cache_mode=cache_mode, dataset=dataset)
        self.imgs = self.samples
        self.props = self.samples_props
        self.select_strategy = select_strategy
        self.select_k = select_k
        self.weight_strategy = weight_strategy
        self.jitter_prob = jitter_prob
        self.jitter_ratio = jitter_ratio
        self.padding_k = padding_k
        self.view_size = (image_size, image_size)
        self.view_size_3 = (image_size//2, image_size//2)
        self.debug = False
        self.max_tries = max_tries
        self.least_common = max(self.padding_k // 2, 1)
        self.aware_range = aware_range
        assert len(self.aware_range) == 5, 'Must give P2 P3 P4 P5 P6 size range'
        self.aware_start = aware_start  # starting from 0 means use p2
        self.aware_end = aware_end  # end, if use P6 might be 5

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        image_size = image.size
        image_proposals = self.props[index]  # for cur image, numpy array type, [[x1, y1, x2, y2]] x2 = x1 + w - 1
        if image_proposals.shape[0] == 0:  # if no proposals, insert one single proposal, the whole raw image
            image_proposals = np.array([[0, 0, image_size[0] - 1, image_size[1] - 1]])

        image_proposals_w_id = append_prop_id(image_proposals)  # start from 1

        assert len(self.transform) == 7
        # transform = (transform_whole_img, transform_img, transform_img_small, transform_flip_flip, transform_flip, transform_post_1, transform_post_2)

        tries = 0
        least_common = self.least_common

        while tries < self.max_tries:
            img, params = self.transform[0](image)  # whole image resize
            img2, params2 = self.transform[1](image)  # random crop resize
            img3, params3 = self.transform[2](image)  # small random crop resize

            params_overlap12 = cal_overlap_params(params, params2)
            overlap_props12 = get_overlap_props(image_proposals_w_id, params_overlap12)
            selected_image_props12 = select_props(overlap_props12, self.select_strategy, self.select_k)  # check paras are

            params_overlap13 = cal_overlap_params(params, params3)
            overlap_props13 = get_overlap_props(image_proposals_w_id, params_overlap13)
            selected_image_props13 = select_props(overlap_props13, self.select_strategy, self.select_k)  # check paras are

            # TODO: ensure clipped bboxs width and height are greater than 32
            if selected_image_props12.shape[0] >= least_common and selected_image_props13.shape[0] >= least_common:  # ok
                break
            least_common = max(least_common // 2, 1)
            tries += 1


        jittered_selected_image_props12 = jitter_props(selected_image_props12, self.jitter_prob, self.jitter_ratio)
        jittered_selected_image_props13 = jitter_props(selected_image_props13, self.jitter_prob, self.jitter_ratio)

        bboxs1_12 = clip_bboxs(jittered_selected_image_props12, params[0], params[1], params[2], params[3])
        bboxs1_13 = clip_bboxs(jittered_selected_image_props13, params[0], params[1], params[2], params[3])
        bboxs2 = clip_bboxs(selected_image_props12, params2[0], params2[1], params2[2], params2[3])
        bboxs3 = clip_bboxs(selected_image_props13, params3[0], params3[1], params3[2], params3[3])
        common_bboxs_ids12 = get_common_bboxs_ids(bboxs1_12, bboxs2)
        common_bboxs_ids13 = get_common_bboxs_ids(bboxs1_13, bboxs3)


        pad1_12 = self.padding_k - bboxs1_12.shape[0]
        if pad1_12 > 0:
            pad_bboxs1_12 = pad_bboxs_with_common(bboxs1_12, common_bboxs_ids12, self.jitter_ratio, pad1_12, params[2], params[3])
            bboxs1_12 = np.concatenate([bboxs1_12, pad_bboxs1_12], axis=0)

        pad1_13 = self.padding_k - bboxs1_13.shape[0]
        if pad1_13 > 0:
            pad_bboxs1_13 = pad_bboxs_with_common(bboxs1_13, common_bboxs_ids13, self.jitter_ratio, pad1_13, params[2], params[3])
            bboxs1_13 = np.concatenate([bboxs1_13, pad_bboxs1_13], axis=0)

        pad2 = self.padding_k - bboxs2.shape[0]
        if pad2 > 0:
            pad_bboxs2 = pad_bboxs_with_common(bboxs2, common_bboxs_ids12, self.jitter_ratio, pad2, params2[2], params2[3])
            bboxs2 = np.concatenate([bboxs2, pad_bboxs2], axis=0)

        pad3 = self.padding_k - bboxs3.shape[0]
        if pad3 > 0:
            pad_bboxs3 = pad_bboxs_with_common(bboxs3, common_bboxs_ids13, self.jitter_ratio, pad3, params3[2], params3[3])
            bboxs3 = np.concatenate([bboxs3, pad_bboxs3], axis=0)


        resized_bboxs1_12 = resize_bboxs(bboxs1_12, params[2], params[3], self.view_size)
        resized_bboxs1_13 = resize_bboxs(bboxs1_13, params[2], params[3], self.view_size)
        resized_bboxs2 = resize_bboxs(bboxs2, params2[2], params2[3], self.view_size)
        resized_bboxs3 = resize_bboxs(bboxs3, params3[2], params3[3], self.view_size_3)
        resized_bboxs1_12 = resized_bboxs1_12.astype(int)
        resized_bboxs1_13 = resized_bboxs1_13.astype(int)
        resized_bboxs2 = resized_bboxs2.astype(int)
        resized_bboxs3 = resized_bboxs3.astype(int)

        bboxs1_12_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs1_12, self.aware_range, self.aware_start, self.aware_end, -1)
        bboxs1_13_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs1_13, self.aware_range, self.aware_start, self.aware_end, -1)
        bboxs2_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs2, self.aware_range, self.aware_start, self.aware_end, -2)
        bboxs3_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs3, self.aware_range, self.aware_start, self.aware_end, -3)


        aware_corres_12 = get_aware_correspondence_matrix(bboxs1_12_with_feature_assign, bboxs2_with_feature_assign)
        aware_corres_13 = get_aware_correspondence_matrix(bboxs1_13_with_feature_assign, bboxs3_with_feature_assign)

        bboxs1_12 = bboxs_to_tensor(bboxs1_12, params)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs1_13 = bboxs_to_tensor(bboxs1_13, params)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs2 = bboxs_to_tensor(bboxs2, params2)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs3 = bboxs_to_tensor(bboxs3, params3)  # x1y1x2y2 -> x1y1x2y2 (0, 1)

        img, bboxs1_12, bboxs1_13, params = self.transform[3](img, bboxs1_12, bboxs1_13, params)  # flip
        img2, bboxs2, params2 = self.transform[4](img2, bboxs2, params2)  # flip
        img3, bboxs3, params3 = self.transform[4](img3, bboxs3, params3)  # flip

        img1 = self.transform[5](img)  # color
        img2 = self.transform[6](img2)  # color
        img3 = self.transform[6](img3)  # color

        return img1, img2, img3, bboxs1_12, bboxs1_13, bboxs2, bboxs3, aware_corres_12, aware_corres_13, index, target


class ImageFolderImageAsymBboxAwareMultiJitter1Cutout(DatasetFolderProps):
    def __init__(self, root, ann_file='', img_prefix='', train_props_file='',
                 image_size=0, select_strategy='', select_k=0, weight_strategy='',
                 jitter_prob=0.0, jitter_ratio=0.0,
                 padding_k='', aware_range=[], aware_start=0, aware_end=4, max_tries=0,
                 transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no", dataset='ImageNet'):
        super(ImageFolderImageAsymBboxAwareMultiJitter1Cutout, self).__init__(root, loader, IMG_EXTENSIONS,
                                                                   ann_file=ann_file, img_prefix=img_prefix,
                                                                   train_props_file=train_props_file,
                                                                   select_strategy=select_strategy, select_k=select_k,
                                                                   transform=transform, target_transform=target_transform,
                                                                   cache_mode=cache_mode, dataset=dataset)
        self.imgs = self.samples
        self.props = self.samples_props
        self.select_strategy = select_strategy
        self.select_k = select_k
        self.weight_strategy = weight_strategy
        self.jitter_prob = jitter_prob
        self.jitter_ratio = jitter_ratio
        self.padding_k = padding_k
        self.view_size = (image_size, image_size)
        self.view_size_3 = (image_size//2, image_size//2)
        self.debug = False
        self.max_tries = max_tries
        self.least_common = max(self.padding_k // 2, 1)
        self.aware_range = aware_range
        assert len(self.aware_range) == 5, 'Must give P2 P3 P4 P5 P6 size range'
        self.aware_start = aware_start  # starting from 0 means use p2
        self.aware_end = aware_end  # end, if use P6 might be 5

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        image_size = image.size
        image_proposals = self.props[index]  # for cur image, numpy array type, [[x1, y1, x2, y2]] x2 = x1 + w - 1
        if image_proposals.shape[0] == 0:  # if no proposals, insert one single proposal, the whole raw image
            image_proposals = np.array([[0, 0, image_size[0] - 1, image_size[1] - 1]])

        image_proposals_w_id = append_prop_id(image_proposals)  # start from 1

        assert len(self.transform) == 8
        # transform = (transform_whole_img, transform_img, transform_img_small, transform_flip_flip, transform_flip, transform_post_1, transform_post_2, transform_cutout)

        tries = 0
        least_common = self.least_common

        while tries < self.max_tries:
            img, params = self.transform[0](image)  # whole image resize
            img2, params2 = self.transform[1](image)  # random crop resize
            img3, params3 = self.transform[2](image)  # small random crop resize

            params_overlap12 = cal_overlap_params(params, params2)
            overlap_props12 = get_overlap_props(image_proposals_w_id, params_overlap12)
            selected_image_props12 = select_props(overlap_props12, self.select_strategy, self.select_k)  # check paras are

            params_overlap13 = cal_overlap_params(params, params3)
            overlap_props13 = get_overlap_props(image_proposals_w_id, params_overlap13)
            selected_image_props13 = select_props(overlap_props13, self.select_strategy, self.select_k)  # check paras are

            # TODO: ensure clipped bboxs width and height are greater than 32
            if selected_image_props12.shape[0] >= least_common and selected_image_props13.shape[0] >= least_common:  # ok
                break
            least_common = max(least_common // 2, 1)
            tries += 1


        jittered_selected_image_props12 = jitter_props(selected_image_props12, self.jitter_prob, self.jitter_ratio)
        jittered_selected_image_props13 = jitter_props(selected_image_props13, self.jitter_prob, self.jitter_ratio)

        bboxs1_12 = clip_bboxs(jittered_selected_image_props12, params[0], params[1], params[2], params[3])
        bboxs1_13 = clip_bboxs(jittered_selected_image_props13, params[0], params[1], params[2], params[3])
        bboxs2 = clip_bboxs(selected_image_props12, params2[0], params2[1], params2[2], params2[3])
        bboxs3 = clip_bboxs(selected_image_props13, params3[0], params3[1], params3[2], params3[3])
        common_bboxs_ids12 = get_common_bboxs_ids(bboxs1_12, bboxs2)
        common_bboxs_ids13 = get_common_bboxs_ids(bboxs1_13, bboxs3)


        pad1_12 = self.padding_k - bboxs1_12.shape[0]
        if pad1_12 > 0:
            pad_bboxs1_12 = pad_bboxs_with_common(bboxs1_12, common_bboxs_ids12, self.jitter_ratio, pad1_12, params[2], params[3])
            bboxs1_12 = np.concatenate([bboxs1_12, pad_bboxs1_12], axis=0)

        pad1_13 = self.padding_k - bboxs1_13.shape[0]
        if pad1_13 > 0:
            pad_bboxs1_13 = pad_bboxs_with_common(bboxs1_13, common_bboxs_ids13, self.jitter_ratio, pad1_13, params[2], params[3])
            bboxs1_13 = np.concatenate([bboxs1_13, pad_bboxs1_13], axis=0)

        pad2 = self.padding_k - bboxs2.shape[0]
        if pad2 > 0:
            pad_bboxs2 = pad_bboxs_with_common(bboxs2, common_bboxs_ids12, self.jitter_ratio, pad2, params2[2], params2[3])
            bboxs2 = np.concatenate([bboxs2, pad_bboxs2], axis=0)

        pad3 = self.padding_k - bboxs3.shape[0]
        if pad3 > 0:
            pad_bboxs3 = pad_bboxs_with_common(bboxs3, common_bboxs_ids13, self.jitter_ratio, pad3, params3[2], params3[3])
            bboxs3 = np.concatenate([bboxs3, pad_bboxs3], axis=0)


        resized_bboxs1_12 = resize_bboxs(bboxs1_12, params[2], params[3], self.view_size)
        resized_bboxs1_13 = resize_bboxs(bboxs1_13, params[2], params[3], self.view_size)
        resized_bboxs2 = resize_bboxs(bboxs2, params2[2], params2[3], self.view_size)
        resized_bboxs3 = resize_bboxs(bboxs3, params3[2], params3[3], self.view_size_3)
        resized_bboxs1_12 = resized_bboxs1_12.astype(int)
        resized_bboxs1_13 = resized_bboxs1_13.astype(int)
        resized_bboxs2 = resized_bboxs2.astype(int)
        resized_bboxs3 = resized_bboxs3.astype(int)

        bboxs1_12_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs1_12, self.aware_range, self.aware_start, self.aware_end, -1)
        bboxs1_13_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs1_13, self.aware_range, self.aware_start, self.aware_end, -1)
        bboxs2_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs2, self.aware_range, self.aware_start, self.aware_end, -2)
        bboxs3_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs3, self.aware_range, self.aware_start, self.aware_end, -3)


        aware_corres_12 = get_aware_correspondence_matrix(bboxs1_12_with_feature_assign, bboxs2_with_feature_assign)
        aware_corres_13 = get_aware_correspondence_matrix(bboxs1_13_with_feature_assign, bboxs3_with_feature_assign)

        bboxs1_12 = bboxs_to_tensor(bboxs1_12, params)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs1_13 = bboxs_to_tensor(bboxs1_13, params)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs2 = bboxs_to_tensor(bboxs2, params2)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs3 = bboxs_to_tensor(bboxs3, params3)  # x1y1x2y2 -> x1y1x2y2 (0, 1)

        img, bboxs1_12, bboxs1_13, params = self.transform[3](img, bboxs1_12, bboxs1_13, params)  # flip
        img2, bboxs2, params2 = self.transform[4](img2, bboxs2, params2)  # flip
        img3, bboxs3, params3 = self.transform[4](img3, bboxs3, params3)  # flip

        img1 = self.transform[5](img)  # color
        img2 = self.transform[6](img2)  # color
        img3 = self.transform[6](img3)  # color

        img2_cutout = self.transform[7](img2, bboxs2, self.view_size)
        img3_cutout = self.transform[7](img3, bboxs3, self.view_size_3)

        return img1, img2_cutout, img3_cutout, bboxs1_12, bboxs1_13, bboxs2, bboxs3, aware_corres_12, aware_corres_13, index, target


class ImageFolderImageAsymBboxAwareMulti3ResizeExtraJitter1(DatasetFolderProps):
    def __init__(self, root, ann_file='', img_prefix='', train_props_file='',
                 image_size=0, image3_size=0, image4_size=0, select_strategy='', select_k=0, weight_strategy='',
                 jitter_prob=0.0, jitter_ratio=0.0,
                 padding_k='', aware_range=[], aware_start=0, aware_end=4, max_tries=0,
                 transform=None, target_transform=None,
                 loader=default_img_loader, cache_mode="no", dataset='ImageNet'):
        super(ImageFolderImageAsymBboxAwareMulti3ResizeExtraJitter1, self).__init__(root, loader, IMG_EXTENSIONS,
                                                                   ann_file=ann_file, img_prefix=img_prefix,
                                                                   train_props_file=train_props_file,
                                                                   select_strategy=select_strategy, select_k=select_k,
                                                                   transform=transform, target_transform=target_transform,
                                                                   cache_mode=cache_mode, dataset=dataset)
        self.imgs = self.samples
        self.props = self.samples_props
        self.select_strategy = select_strategy
        self.select_k = select_k
        self.weight_strategy = weight_strategy
        self.jitter_prob = jitter_prob
        self.jitter_ratio = jitter_ratio
        self.padding_k = padding_k
        self.view_size = (image_size, image_size)
        self.view_size_3 = (image3_size, image3_size)
        self.view_size_4 = (image4_size, image4_size)
        assert image3_size > 0
        assert image4_size > 0
        self.debug = False
        self.max_tries = max_tries
        self.least_common = max(self.padding_k // 2, 1)
        self.aware_range = aware_range
        assert len(self.aware_range) == 5, 'Must give P2 P3 P4 P5 P6 size range'
        self.aware_start = aware_start  # starting from 0 means use p2
        self.aware_end = aware_end  # end, if use P6 might be 5

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        image_size = image.size
        image_proposals = self.props[index]  # for cur image, numpy array type, [[x1, y1, x2, y2]] x2 = x1 + w - 1
        if image_proposals.shape[0] == 0:  # if no proposals, insert one single proposal, the whole raw image
            image_proposals = np.array([[0, 0, image_size[0] - 1, image_size[1] - 1]])

        image_proposals_w_id = append_prop_id(image_proposals)  # start from 1

        assert len(self.transform) == 8
        # transform = (transform_whole_img, transform_img, transform_img_small, transform_img_resize, transform_flip_flip, transform_flip, transform_post_1, transform_post_2)

        tries = 0
        least_common = self.least_common

        while tries < self.max_tries:
            img, params = self.transform[0](image)  # whole image resize
            img2, params2 = self.transform[1](image)  # random crop resize
            img3, params3 = self.transform[2](image)  # small random crop resize

            params_overlap12 = cal_overlap_params(params, params2)
            overlap_props12 = get_overlap_props(image_proposals_w_id, params_overlap12)
            selected_image_props12 = select_props(overlap_props12, self.select_strategy, self.select_k)  # check paras are

            params_overlap13 = cal_overlap_params(params, params3)
            overlap_props13 = get_overlap_props(image_proposals_w_id, params_overlap13)
            selected_image_props13 = select_props(overlap_props13, self.select_strategy, self.select_k)  # check paras are


            # TODO: ensure clipped bboxs width and height are greater than 32
            if selected_image_props12.shape[0] >= least_common and selected_image_props13.shape[0] >= least_common:  # ok
                break
            least_common = max(least_common // 2, 1)
            tries += 1

        img4  = self.transform[3](img2)  # image4 are resized from image 2

        jittered_selected_image_props12 = jitter_props(selected_image_props12, self.jitter_prob, self.jitter_ratio)
        jittered_selected_image_props13 = jitter_props(selected_image_props13, self.jitter_prob, self.jitter_ratio)

        bboxs1_12 = clip_bboxs(jittered_selected_image_props12, params[0], params[1], params[2], params[3])
        bboxs1_13 = clip_bboxs(jittered_selected_image_props13, params[0], params[1], params[2], params[3])
        bboxs2 = clip_bboxs(selected_image_props12, params2[0], params2[1], params2[2], params2[3])
        bboxs3 = clip_bboxs(selected_image_props13, params3[0], params3[1], params3[2], params3[3])
        common_bboxs_ids12 = get_common_bboxs_ids(bboxs1_12, bboxs2)
        common_bboxs_ids13 = get_common_bboxs_ids(bboxs1_13, bboxs3)


        pad1_12 = self.padding_k - bboxs1_12.shape[0]
        if pad1_12 > 0:
            pad_bboxs1_12 = pad_bboxs_with_common(bboxs1_12, common_bboxs_ids12, self.jitter_ratio, pad1_12, params[2], params[3])
            bboxs1_12 = np.concatenate([bboxs1_12, pad_bboxs1_12], axis=0)

        pad1_13 = self.padding_k - bboxs1_13.shape[0]
        if pad1_13 > 0:
            pad_bboxs1_13 = pad_bboxs_with_common(bboxs1_13, common_bboxs_ids13, self.jitter_ratio, pad1_13, params[2], params[3])
            bboxs1_13 = np.concatenate([bboxs1_13, pad_bboxs1_13], axis=0)

        pad2 = self.padding_k - bboxs2.shape[0]
        if pad2 > 0:
            pad_bboxs2 = pad_bboxs_with_common(bboxs2, common_bboxs_ids12, self.jitter_ratio, pad2, params2[2], params2[3])
            bboxs2 = np.concatenate([bboxs2, pad_bboxs2], axis=0)

        pad3 = self.padding_k - bboxs3.shape[0]
        if pad3 > 0:
            pad_bboxs3 = pad_bboxs_with_common(bboxs3, common_bboxs_ids13, self.jitter_ratio, pad3, params3[2], params3[3])
            bboxs3 = np.concatenate([bboxs3, pad_bboxs3], axis=0)

        bboxs1_14 = np.copy(bboxs1_12)

        bboxs4 = np.copy(bboxs2)
        params4 = np.copy(params2)

        resized_bboxs1_12 = resize_bboxs(bboxs1_12, params[2], params[3], self.view_size)
        resized_bboxs1_13 = resize_bboxs(bboxs1_13, params[2], params[3], self.view_size)
        resized_bboxs1_14 = resize_bboxs(bboxs1_14, params[2], params[3], self.view_size)
        resized_bboxs2 = resize_bboxs(bboxs2, params2[2], params2[3], self.view_size)
        resized_bboxs3 = resize_bboxs(bboxs3, params3[2], params3[3], self.view_size_3)
        resized_bboxs4 = resize_bboxs(bboxs4, params4[2], params4[3], self.view_size_4)
        resized_bboxs1_12 = resized_bboxs1_12.astype(int)
        resized_bboxs1_13 = resized_bboxs1_13.astype(int)
        resized_bboxs1_14 = resized_bboxs1_14.astype(int)
        resized_bboxs2 = resized_bboxs2.astype(int)
        resized_bboxs3 = resized_bboxs3.astype(int)
        resized_bboxs4 = resized_bboxs4.astype(int)

        bboxs1_12_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs1_12, self.aware_range, self.aware_start, self.aware_end, -1)
        bboxs1_13_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs1_13, self.aware_range, self.aware_start, self.aware_end, -1)
        bboxs1_14_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs1_14, self.aware_range, self.aware_start, self.aware_end, -1)
        bboxs2_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs2, self.aware_range, self.aware_start, self.aware_end, -2)
        bboxs3_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs3, self.aware_range, self.aware_start, self.aware_end, -3)
        bboxs4_with_feature_assign = assign_bboxs_to_feature_map(resized_bboxs4, self.aware_range, self.aware_start, self.aware_end, -4)


        aware_corres_12 = get_aware_correspondence_matrix(bboxs1_12_with_feature_assign, bboxs2_with_feature_assign)
        aware_corres_13 = get_aware_correspondence_matrix(bboxs1_13_with_feature_assign, bboxs3_with_feature_assign)
        aware_corres_14 = get_aware_correspondence_matrix(bboxs1_14_with_feature_assign, bboxs4_with_feature_assign)

        bboxs1_12 = bboxs_to_tensor(bboxs1_12, params)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs1_13 = bboxs_to_tensor(bboxs1_13, params)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs1_14 = bboxs_to_tensor(bboxs1_14, params)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs2 = bboxs_to_tensor(bboxs2, params2)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs3 = bboxs_to_tensor(bboxs3, params3)  # x1y1x2y2 -> x1y1x2y2 (0, 1)
        bboxs4 = bboxs_to_tensor(bboxs4, params4)  # x1y1x2y2 -> x1y1x2y2 (0, 1)

        img, bboxs1_12, bboxs1_13, bboxs1_14, params = self.transform[4](img, bboxs1_12, bboxs1_13, bboxs1_14, params)  # flip
        img2, bboxs2, params2 = self.transform[5](img2, bboxs2, params2)  # flip
        img3, bboxs3, params3 = self.transform[5](img3, bboxs3, params3)  # flip
        img4, bboxs4, params4 = self.transform[5](img4, bboxs4, params4)  # flip

        img1 = self.transform[6](img)  # color
        img2 = self.transform[7](img2)  # color
        img3 = self.transform[7](img3)  # color
        img4 = self.transform[7](img4)  # color

        return img1, img2, img3, img4, bboxs1_12, bboxs1_13, bboxs1_14, bboxs2, bboxs3, bboxs4, aware_corres_12, aware_corres_13, aware_corres_14, index, target
