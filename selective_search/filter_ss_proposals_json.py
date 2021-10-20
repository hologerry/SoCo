import os
import pickle
import multiprocessing as mp
import numpy as np
import json

from filters import filter_none, filter_ratio, filter_size, filter_ratio_size


imagenet_root = './imagenet_root'
imagenet_root_proposals = './imagenet_root_proposals_mp'
filter_strategy = 'ratio3size0308'  # 'ratio2', 'ratio3', 'ratio4', 'size01'
print("filter_strategy", filter_strategy)
filtered_proposals = './imagenet_filtered_proposals'

split = 'train'


filtered_proposals_dict = {}


os.makedirs(filtered_proposals, exist_ok=True)
json_path = os.path.join(filtered_proposals, f'{split}_{filter_strategy}.json')
source_path = os.path.join(imagenet_root_proposals, split)

class_names = sorted(os.listdir(os.path.join(imagenet_root_proposals, split)))

no_props_images = []


for ci, class_name in enumerate(class_names):
    filenames = sorted(os.listdir(os.path.join(source_path, class_name)))
    for fi, filename in enumerate(filenames):
        base_filename = os.path.splitext(filename)[0]
        cur_img_pro_path = os.path.join(source_path, class_name, filename)

        with open(cur_img_pro_path, 'rb') as f:
            cur_img_proposal = pickle.load(f)
            if filter_strategy == 'none':
                filtered_img_rects = filter_none(cur_img_proposal['regions'])
            elif 'ratio' in filter_strategy and 'size' in filter_strategy:
                ratio = float(filter_strategy[5])
                min_size_ratio = float(filter_strategy[-4:-2]) / 10
                max_size_ratio = float(filter_strategy[-2:]) / 10
                filtered_img_rects = filter_ratio_size(cur_img_proposal['regions'], cur_img_proposal['label'].shape, ratio, min_size_ratio, max_size_ratio)
            elif 'ratio' in filter_strategy:
                ratio = float(filter_strategy[-1])
                filtered_img_rects = filter_ratio(cur_img_proposal['regions'], r=ratio)
            elif 'size' in filter_strategy:
                min_size_ratio = float(filter_strategy[-2:]) / 10
                filtered_img_rects = filter_size(cur_img_proposal['regions'], cur_img_proposal['label'].shape, min_size_ratio)
            else:
                raise NotImplementedError
        
        filtered_proposals_dict[base_filename] = filtered_img_rects
        if len(filtered_img_rects) == 0:
            no_props_images.append(base_filename)
            print(f"with strategy {filter_strategy}, image {base_filename} has no proposals")
        if (fi + 1) % 100 == 0:
            print(f"Processed [{ci}/{len(class_names)}] classes, [{fi+1}/{len(filenames)}] images")


print(f"Finished filtering with strategy {filter_strategy}, there are {len(no_props_images)} images have no proposals.")


with open(json_path, 'w') as f:
    json.dump(filtered_proposals_dict, f)


with open(json_path.replace('.json', 'no_props_images.txt'), 'w') as f:
    for image_id in no_props_images:
        f.write(image_id+'\n')

