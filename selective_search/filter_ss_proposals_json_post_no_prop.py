import json
import os
import pickle

from filters import filter_none, filter_ratio


json_path = './imagenet_filtered_proposals/train_ratio3size0308.json'
json_path_post = './imagenet_filtered_proposals/train_ratio3size0308post.json'
no_props_images = open('./imagenet_filtered_proposals/train_ratio3size0308no_props_images.txt').readlines()
imagenet_root_proposals = './imagenet_root_proposals_mp'


with open(json_path, 'r') as f:
    json_dict = json.load(f)


for no_props_image in no_props_images:
    filename = no_props_image.strip()
    class_name = filename.split('_')[0]
    cur_img_pro_path = os.path.join(imagenet_root_proposals, 'train', class_name, filename+'.pkl')

    with open(cur_img_pro_path, 'rb') as f:
        cur_img_proposal = pickle.load(f)
        props_size_ratio = filter_ratio(cur_img_proposal['regions'], r=3)
        props_none = filter_none(cur_img_proposal['regions'])
        print("props_size_ratio", len(props_size_ratio))
        print("props_none", len(props_none))
        if len(props_size_ratio) > 0:
            json_dict[filename] = props_size_ratio
        elif len(props_none) > 0:
            json_dict[filename] = filter_none(cur_img_proposal['regions'])


with open(json_path_post, 'w') as f:
    json.dump(json_dict, f)
