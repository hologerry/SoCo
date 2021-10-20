# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import multiprocessing as mp
import os
import pickle

import numpy as np
import PIL.Image as Image

from .selective_search import selective_search

imagenet_root = './imagenet_root'
imagenet_root_proposals = './imagenet_root_proposals_mp'

split = 'train'
scale = 300
min_size = 100

processes_num = 48
class_names = sorted(os.listdir(os.path.join(imagenet_root, split)))
classes_num = len(class_names)
classes_per_process = classes_num // processes_num + 1

source_path = os.path.join(imagenet_root, split)
target_path = os.path.join(imagenet_root_proposals, split)


def process_one_class(process_id, classes_per_process, class_names, source_path, target_path):
    print(f"Process id: {process_id} started")
    for i in range(process_id*classes_per_process, process_id*classes_per_process + classes_per_process):
        if i >= len(class_names):
            break
        class_name = class_names[i]
        filenames = sorted(os.listdir(os.path.join(source_path, class_name)))
        os.makedirs(os.path.join(target_path, class_name))
        for filename in filenames:
            base_filename = os.path.splitext(filename)[0]
            img_path = os.path.join(source_path, class_name, filename)
            img = np.array(Image.open(img_path).convert('RGB'))

            img_with_lbl, regions, bboxs = selective_search(img, scale=scale, sigma=0.9, min_size=min_size)

            region_label = img_with_lbl[:, :, 3]
            cur_img_proposal = {}
            cur_img_proposal['label'] = region_label
            cur_img_proposal['regions'] = regions

            cur_img_pro_path = os.path.join(target_path, class_name, base_filename+'.pkl')

            with open(cur_img_pro_path, 'wb') as f:
                pickle.dump(cur_img_proposal, f)
        print("Process ", process_id, "processed class:", class_name)


processes = [mp.Process(target=process_one_class,
                        args=(process_id, classes_per_process, class_names, source_path, target_path))
                        for process_id in range(processes_num)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()
