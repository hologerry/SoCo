# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import math
import random

import numpy as np
import torch
import torchvision.transforms as T
from mmdet.core import anchor_inside_flags


def overlap_image_bbox_w_id(image_bbox, bbox):
    ix1, iy1, ix2, iy2 = image_bbox
    bx1, by1, bx2, by2, bid = bbox

    if ix1 < bx2 and bx1 < ix2 and iy1 < by2 and by1 < iy2:
        nx1, ny1, nx2, ny2 = max(ix1, bx1), max(iy1, by1), min(ix2, bx2), min(iy2, by2)
        # minus the crop image x, y
        nx1, ny1 = nx1 - ix1, ny1 - iy1
        nx2, ny2 = nx2 - ix1, ny2 - iy1
        return np.array([nx1, ny1, nx2, ny2, bid])
    else:
        return None


def cal_overlap_params(params1, params2):
    y11, x11, h1, w1, _, _ = params1
    y21, x21, h2, w2, _, _ = params2
    y12, x12 = y11 + h1 - 1, x11 + w1 - 1
    y22, x22 = y21 + h2 - 1, x21 + w2 - 1

    if x11 < x22 and x21 < x12 and y11 < y22 and y21 < y12:
        nx1, ny1, nx2, ny2 = max(x11, x21), max(y11, y21), min(x12, x22), min(y12, y22)
        return np.array([nx1, ny1, nx2, ny2])
    else:
        return None


def is_overlap(params_overlap, bbox):
    px1, py1, px2, py2 = params_overlap
    bx1, by1, bx2, by2, _ = bbox

    if px1 < bx2 and bx1 < px2 and py1 < by2 and by1 < py2:
        return True
    else:
        return False


def clip_bboxs(bboxs, top, left, height, width):
    clipped_bboxs_w_id_list = []
    clip_image_bbox = np.array([left, top, left + width - 1, top + height - 1])
    for cur_bbox in bboxs:
        overlap_bbox = overlap_image_bbox_w_id(clip_image_bbox, cur_bbox)
        # assert overlap_bbox is not None, print("clip_image_bbox", clip_image_bbox, "cur_bbox", cur_bbox)
        if overlap_bbox is not None:
            clipped_bboxs_w_id_list.append(overlap_bbox.reshape(1, overlap_bbox.shape[0]))

    if len(clipped_bboxs_w_id_list) > 0:
        clipped_bboxs_w_id = np.concatenate(clipped_bboxs_w_id_list, axis=0)
    else:
        clipped_bboxs_w_id = np.array([[0, 0, width - 1, height - 1, 1]])  # just whole view
    return clipped_bboxs_w_id


def clip_bboxs_in_jitter(bboxs, top, left, height, width):
    clipped_bboxs_w_id_list = []
    clip_image_bbox = np.array([left, top, left + width - 1, top + height - 1])
    for cur_bbox in bboxs:
        overlap_bbox = overlap_image_bbox_w_id(clip_image_bbox, cur_bbox)
        if overlap_bbox is not None:
            clipped_bboxs_w_id_list.append(overlap_bbox.reshape(1, overlap_bbox.shape[0]))

    if len(clipped_bboxs_w_id_list) > 0:
        clipped_bboxs_w_id = np.concatenate(clipped_bboxs_w_id_list, axis=0)
        return clipped_bboxs_w_id
    else:
        return None


def get_overlap_props(proposals, overlap_region):
    if overlap_region is None:
        return np.array([])
    common_props = []
    for prop in proposals:
        if is_overlap(overlap_region, prop):
            common_props.append(prop.reshape(1, prop.shape[0]))
    if len(common_props) > 0:
        common_props = np.concatenate(common_props, axis=0)
        return common_props
    else:
        return np.array([])


def resize_bboxs(clipped_bboxs_w_id, height, width, size):
    bboxs_w_id = np.copy(clipped_bboxs_w_id).astype(float)  # !!!
    bboxs_w_id[:, 0] = bboxs_w_id[:, 0] / width * size[0]  # x1
    bboxs_w_id[:, 1] = bboxs_w_id[:, 1] / height * size[1]  # y1
    bboxs_w_id[:, 2] = bboxs_w_id[:, 2] / width * size[0]  # x2
    bboxs_w_id[:, 3] = bboxs_w_id[:, 3] / height * size[1]  # y2
    return bboxs_w_id


def resize_bboxs_vis(clipped_bboxs_w_id, size):
    bboxs_w_id = np.copy(clipped_bboxs_w_id).astype(float)  # !!!
    bboxs_w_id[:, 0] = bboxs_w_id[:, 0] * size[0]  # x1
    bboxs_w_id[:, 1] = bboxs_w_id[:, 1] * size[1]  # y1
    bboxs_w_id[:, 2] = bboxs_w_id[:, 2] * size[0]  # x2
    bboxs_w_id[:, 3] = bboxs_w_id[:, 3] * size[1]  # y2
    return bboxs_w_id


def resize_bboxs_and_assign_labels(cropped_bboxs, height, width, size, bbox_size_range):
    resized_bboxs_w_labels = torch.empty((cropped_bboxs.shape[0], cropped_bboxs.shape[1]+2), requires_grad=False)
    resized_bboxs_vis = np.zeros_like(cropped_bboxs)
    # 2 is used for p5, p4 one hot, determinted by bbox_size_range
    min_size = bbox_size_range[0]  # (32, 112, 224)
    mid_size = bbox_size_range[1]
    for i in range(cropped_bboxs.shape[0]):
        cur_bbox = cropped_bboxs[i]
        if cur_bbox[2] <= 0 or cur_bbox[3] <= 0:
            # valid coordinates but we do not use it
            resized_bboxs_w_labels[i] = torch.Tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
            continue
        x, y, w, h = cur_bbox
        nx = x / width * size[0]
        ny = y / height * size[1]
        nw = w / width * size[0]
        nh = h / height * size[1]
        resized_bboxs_vis[i] = np.array([nx, ny, nw, nh])
        # NOTE: we swap x, y and w, h to align with image, and turn into 0~1
        short_side = min(nw, nh)
        long_side = max(nw, nh)
        if short_side >= min_size:
            if long_side < mid_size:  # use p4
                resized_bboxs_w_labels[i] = torch.Tensor([ny / size[1], nx / size[0], (ny + nh) / size[1], (nx + nw) / size[0], 1.0, 0.0])
            else:  # use p5
                resized_bboxs_w_labels[i] = torch.Tensor([ny / size[1], nx / size[0], (ny + nh) / size[1], (nx + nw) / size[0], 0.0, 1.0])
        else:
            resized_bboxs_w_labels[i] = torch.Tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    # print("resized bboxs size tensor", resized_bboxs_w_labels)
    # print("resized bboxs size vis numpy", resized_bboxs_vis)
    return resized_bboxs_w_labels, resized_bboxs_vis


def resized_crop_bboxs_assign_labels_with_ids(bboxs_w_id, top, left, height, width, size, bbox_size_range):
    # bboxs_w_id, x, y, w, h, are in image size range, not in [0, 1]
    cropped_bboxs_w_id = clip_bboxs(bboxs_w_id, top, left, height, width)
    resized_cropped_bboxs, resized_bboxs_w_id_vis = resize_bboxs_and_assign_labels(cropped_bboxs_w_id, height, width, size, bbox_size_range)
    # resize_cropped_bboxs_w_id is same shape with bboxs_w_id, we pad zeros to do data batching
    return resized_cropped_bboxs, resized_bboxs_w_id_vis


def clip_and_resize_bboxs_w_ids(bboxs_w_id, top, left, height, width, size):
    clipped_bboxs_w_id = clip_bboxs(bboxs_w_id, top, left, height, width)
    resized_clipped_bboxs_w_id = resize_bboxs(clipped_bboxs_w_id, height, width, size)
    return resized_clipped_bboxs_w_id


def get_common_bboxs_ids(bboxs1, bboxs2):
    w = np.where(np.in1d(bboxs1[:, 4], bboxs2[:, 4]))[0]  # intersect of ids, here bboxs ids are unique
    common_bboxs_ids = bboxs1[w][:, 4]
    return common_bboxs_ids


def jitter_bboxs(bboxs, common_bboxs_ids, jitter_ratio, pad_num, height, width):
    common_indices = np.isin(bboxs[:, 4], common_bboxs_ids)
    common_bboxs = bboxs[common_indices]
    clipped_jittered_bboxs_list = []
    remaining_pad = pad_num
    while remaining_pad > 0:
        selected_bboxs = common_bboxs[np.random.choice(common_bboxs.shape[0], remaining_pad)]
        jitters = np.random.uniform(low=-jitter_ratio, high=jitter_ratio, size=(remaining_pad, 4))

        jittered_bboxs = np.copy(selected_bboxs).astype(float)
        selected_bboxs_w = selected_bboxs[:, 2] - selected_bboxs[:, 0] + 1  # w = x2 - x1 + 1
        selected_bboxs_h = selected_bboxs[:, 3] - selected_bboxs[:, 1] + 1  # h = y2 - y1 + 1
        jittered_w = selected_bboxs_w + jitters[:, 2] * selected_bboxs_w  # nw = w + j * w
        jittered_h = selected_bboxs_h + jitters[:, 3] * selected_bboxs_h  # nh = h + j * h
        jittered_bboxs[:, 0] = selected_bboxs[:, 0] + jitters[:, 0] * selected_bboxs_w  # nx1 = x1 + j * w
        jittered_bboxs[:, 1] = selected_bboxs[:, 1] + jitters[:, 1] * selected_bboxs_h  # ny1 = y1 + j * h
        jittered_bboxs[:, 2] = jittered_bboxs[:, 0] + jittered_w - 1  # nx2 = nx1 + nw - 1
        jittered_bboxs[:, 3] = jittered_bboxs[:, 1] + jittered_h - 1  # ny2 = ny1 + nh - 1

        clipped_jittered_bboxs = clip_bboxs_in_jitter(jittered_bboxs, 0, 0, height, width)
        if clipped_jittered_bboxs is not None and clipped_jittered_bboxs.shape[0] > 0:
            clipped_jittered_bboxs_list.append(clipped_jittered_bboxs)
            remaining_pad -= clipped_jittered_bboxs.shape[0]

    padded_clipped_jittered_bboxs = np.concatenate(clipped_jittered_bboxs_list, axis=0)

    return padded_clipped_jittered_bboxs


def jitter_props(selected_image_props, jitter_prob, jitter_ratio):
    jittered_props = []
    for prop in selected_image_props:
        jitter_r = random.random()
        jittered_prop = np.copy(prop).astype(float)
        if jitter_r < jitter_prob:
            jitter = np.random.uniform(low=-jitter_ratio, high=jitter_ratio, size=(4, ))
            w = prop[2] - prop[0] + 1
            h = prop[3] - prop[1] + 1

            jittered_w = w + jitter[2] * w  # nw = w + j * w
            jittered_h = h + jitter[3] * h  # nh = h + j * h

            jittered_prop[0] = prop[0] + jitter[0] * w  # nx1 = x1 + j * w
            jittered_prop[1] = prop[1] + jitter[1] * h  # ny1 = y1 + j * h

            jittered_prop[2] = jittered_prop[0] + jittered_w - 1  # nx2 = nx1 + nw - 1
            jittered_prop[3] = jittered_prop[1] + jittered_h - 1  # ny2 = ny1 + nh - 1

        jittered_prop = jittered_prop.reshape(1, jittered_prop.shape[0])
        jittered_props.append(jittered_prop)

    if len(jittered_props) > 0:
        jittered_props_np = np.concatenate(jittered_props, axis=0)
        return jittered_props_np
    else:
        return np.array([])


def random_generate_props(image_size, r=3.0, min_ratio=0.3, max_ratio=0.8, max_props=32):
    props = []
    for _ in range(max_props):
        sqrt_area = math.sqrt(image_size[0] * image_size[1])
        target_sqrt_area = random.uniform(min_ratio, max_ratio) * sqrt_area
        target_area = target_sqrt_area * target_sqrt_area
        aspect_ratio = random.uniform(1/r, r)
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < 32 or h < 32 or w > image_size[0] or h > image_size[1]:
            continue
        center_x = random.randint(w // 2, image_size[0]- w // 2)
        center_y = random.randint(h // 2, image_size[1]- h // 2)
        x1 = max(center_x, 0)
        x2 = min(center_x + w // 2, image_size[0])
        y1 = max(center_y, 0)
        y2 = min(center_y + h // 2, image_size[1])
        prop = np.array([x1, y1, x2, y2]).reshape((1, 4))
        props.append(prop)
    if len(props) > 0:
        proposals = np.concatenate(props)
    else:
        proposals = np.array([])
    return proposals


def pad_bboxs_with_common(bboxs, common_bboxs_ids, jitter_ratio, pad_num, height, width):
    common_indices = np.isin(bboxs[:, 4], common_bboxs_ids)
    common_bboxs = bboxs[common_indices]
    selected_bboxs = common_bboxs[np.random.choice(common_bboxs.shape[0], pad_num)]
    return selected_bboxs


def get_correspondence_matrix(bboxs1, bboxs2):
    # intersect of ids, here bboxs ids can be duplicate
    assert bboxs1.shape[0] == bboxs2.shape[0]
    L = bboxs1.shape[0]
    bboxs1_ids = bboxs1[:, 4]
    bboxs2_ids = bboxs2[:, 4]
    bboxs1_ids = np.reshape(bboxs1_ids, (L, 1))
    bboxs2_ids = np.reshape(bboxs2_ids, (1, L))
    bboxs1_m = np.tile(bboxs1_ids, (1, L))
    bboxs2_m = np.tile(bboxs2_ids, (L, 1))
    correspondence_matrix = bboxs1_m == bboxs2_m
    correspondence_matrix = correspondence_matrix.astype(float)
    correspondence_matrix = torch.Tensor(correspondence_matrix)
    return correspondence_matrix


def calculate_centerness_targets_from_bbox(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    left = np.tile(np.array([i for i in range(w)]).reshape(w, 1), (1, h))
    right = np.tile(np.array([w - i - 1 for i in range(w)]).reshape(w, 1), (1, h))
    top = np.tile(np.array([i for i in range(h)]).reshape(1, h), (w, 1))
    bottom = np.tile(np.array([h - i - 1 for i in range(h)]).reshape(1, h), (w, 1))

    left_right_min = np.minimum(left, right)
    left_right_max = np.maximum(left, right) + 1e-6
    top_bottom_min = np.minimum(top, bottom)
    top_bottom_max = np.maximum(top, bottom) + 1e-6

    centerness_targets = (left_right_min / left_right_max) * (top_bottom_min / top_bottom_max)
    centerness_targets = np.sqrt(centerness_targets)
    return centerness_targets


def calculate_weight_map_bboxs(bboxs, size, weight_strategy):
    if weight_strategy == 'no_weights':
        weight_map = np.ones(size).astype(float)
        return weight_map

    weight_map = np.zeros(size).astype(float)
    for bbox in bboxs:
        # print("calculate_weight_map_bboxs bbox", bbox)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if w < 1 or h < 1:
            continue
        if weight_strategy == 'bbox':
            weight_map[x1:x2+1, y1:y2+1] += 1.0
        elif weight_strategy == 'center':
            center_x = (x1 + x2) // 2  # ??
            center_y = (y1 + y2) // 2  # ??
            weight_map[center_x, center_y] += 1.0
        elif weight_strategy == 'gaussian':
            centerness_targets = calculate_centerness_targets_from_bbox(bbox)
            weight_map[x1:x2+1, y1:y2+1] += centerness_targets
        else:
            raise NotImplementedError

    return weight_map


def proposals_to_tensor(props):
    props = props.astype(float)
    props_tensor = torch.Tensor(props)
    return props_tensor


def bboxs_to_tensor(bboxs, params):
    """ x1y1x2y2, -> 0, 1
    """
    _, _, height, width, _, _ = params
    bboxs = bboxs.astype(float)
    bboxs_new = np.copy(bboxs)  # keep ids

    bboxs_new[:, 0] = bboxs_new[:, 0] / width  # x1
    bboxs_new[:, 1] = bboxs_new[:, 1] / height  # y1
    bboxs_new[:, 2] = bboxs_new[:, 2] / width  # x2
    bboxs_new[:, 3] = bboxs_new[:, 3] / height  # y2

    bboxs_tensor = torch.Tensor(bboxs_new)

    return bboxs_tensor


def bboxs_to_tensor_dynamic(bboxs, params, dynamic_params, image_size):
    """ x1y1x2y2, -> 0, 1
    """
    _, _, height, width, _, _ = params
    dynamic_resize = dynamic_params[3]
    bboxs = bboxs.astype(float)
    bboxs_new = np.copy(bboxs)  # keep ids

    # raw_size -> raw_ratio -> dynamic_size -> padded_ratio

    bboxs_new[:, 0] = bboxs_new[:, 0] / width * dynamic_resize[0] / image_size[0] # x1
    bboxs_new[:, 1] = bboxs_new[:, 1] / height * dynamic_resize[1] / image_size[1] # y1
    bboxs_new[:, 2] = bboxs_new[:, 2] / width * dynamic_resize[0] / image_size[0] # x2
    bboxs_new[:, 3] = bboxs_new[:, 3] / height * dynamic_resize[1] / image_size[1] # y2

    bboxs_tensor = torch.Tensor(bboxs_new)

    return bboxs_tensor


def weight_to_tensor(weight):
    """ w, h -> h, w
    """
    weight_tensor = torch.Tensor(weight)

    weight_tensor = weight_tensor.unsqueeze(0)  # channel 1
    weight_tensor = torch.transpose(weight_tensor, 1, 2)
    return weight_tensor


def assign_bboxs_to_feature_map(resized_bboxs, aware_range, aware_start, aware_end, not_used_value=-1):
    """aware_range
    """
    L = resized_bboxs.shape[0]
    P = aware_end - aware_start
    assert P > 0

    bboxs_id_assigned = np.ones((P, L)) * not_used_value  # the not used value is use to be different from 0
    for i, bbox in enumerate(resized_bboxs):
        x1, y1, x2, y2, bid = bbox
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = math.sqrt(h * w)
        for j in range(aware_start, aware_end):
            if size <= aware_range[j]:
                bboxs_id_assigned[j - aware_start, i] = bid
                # print(f"{bid} assigned to feature {j - aware_start}")
                break  # assign bbox to only one feature map

    bboxs_id_assigned = bboxs_id_assigned.reshape((P * L, ))
    return bboxs_id_assigned


def get_aware_correspondence_matrix(bboxs_id_assigned1, bboxs_id_assigned2):
    PL = bboxs_id_assigned1.shape[0]
    bboxs1_ids = np.reshape(bboxs_id_assigned1, (PL, 1))
    bboxs2_ids = np.reshape(bboxs_id_assigned2, (1, PL))
    bboxs1_m = np.tile(bboxs1_ids, (1, PL))
    bboxs2_m = np.tile(bboxs2_ids, (PL, 1))
    correspondence_matrix = bboxs1_m == bboxs2_m
    correspondence_matrix = correspondence_matrix.astype(float)
    correspondence_matrix = torch.Tensor(correspondence_matrix)
    return correspondence_matrix


def get_aware_correspondence_matrix_torch(bboxs_id_assigned1, bboxs_id_assigned2):
    PL = bboxs_id_assigned1.size(0)
    bboxs1_ids = torch.reshape(bboxs_id_assigned1, (PL, 1))
    bboxs2_ids = torch.reshape(bboxs_id_assigned2, (1, PL))
    bboxs1_m = bboxs1_ids.repeat(1, PL)
    bboxs2_m = bboxs2_ids.repeat(PL, 1)
    correspondence_matrix = bboxs1_m == bboxs2_m
    correspondence_matrix = correspondence_matrix.float()

    return correspondence_matrix


def visualize_image_tensor(image_tensor, visualize_name):
    image_pil = T.functional.to_pil_image(image_tensor)
    path = f'self_det/visualization/{visualize_name}.jpg'
    image_pil.save(path)


def assign_gt_bboxs_to_feature_map_with_anchors(anchor_generator, assigner, gt_bboxs, view_size, img_meta, levels=4, not_used_value=-1):
    # assign gt bbox based on number of anchors in the feature map
    # SINGLE image version

    P = levels
    L = gt_bboxs.size(0)  # each image number of gts

    featmap_sizes = []
    for l in range(levels):
        feat_size_0 = int(math.ceil(view_size[0]/2**(l+2)))  # p2 - p5
        feat_size_1 = int(math.ceil(view_size[1]/2**(l+2)))  # p2 - p5
        feat_size_tensor = torch.tensor((feat_size_0, feat_size_1))
        featmap_sizes.append(feat_size_tensor)

    # we compute multi level anchors and valid flags
    multi_level_anchors = anchor_generator.grid_anchors(featmap_sizes, device='cpu')
    multi_level_flags = anchor_generator.valid_flags(featmap_sizes, img_meta['pad_shape'], device='cpu')

    num_level_anchors = [anchors.size(0) for anchors in multi_level_anchors]
    num_level_anchors_agg = [0]
    for level_num in num_level_anchors:
        num_level_anchors_agg.append(num_level_anchors_agg[-1] + level_num)

    # concat all level anchors to a single tensor
    flat_anchors = torch.cat(multi_level_anchors)
    flat_valid_flags = torch.cat(multi_level_flags)

    inside_flags = anchor_inside_flags(flat_anchors, flat_valid_flags,
                                        img_meta['img_shape'][:2],
                                        allowed_border=-1) # -1 is come from the default value of train_cfg.rpn.allowed_border

    anchors = flat_anchors[inside_flags, :]


    gt_bboxs_coord = gt_bboxs[:, :4].clone()  # must clone
    cur_img_assign_result = assigner.assign(anchors, gt_bboxs_coord, None, None)
    assigned_gts = cur_img_assign_result.gt_inds

    bboxs_id_assigned = torch.ones((P, L)) * not_used_value
    for gt_idx in range(L):
        cur_gt_assign = assigned_gts == (gt_idx+1)  # the assign results index are 1-based
        cur_gt_level_assign_sum = [None] * P
        for level_i in range(P):
            level_start = num_level_anchors_agg[level_i]
            level_end = num_level_anchors_agg[level_i+1]
            cur_gt_level_assign_sum[level_i] = torch.sum(cur_gt_assign[level_start:level_end]).item()
        cur_gt_assign_level = cur_gt_level_assign_sum.index(max(cur_gt_level_assign_sum))

        bboxs_id_assigned[cur_gt_assign_level, gt_idx] = gt_bboxs[gt_idx, 4]  # assign bbox id

    bboxs_id_assigned = bboxs_id_assigned.reshape((P * L, ))

    return bboxs_id_assigned
