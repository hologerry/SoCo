# --------------------------------------------------------
# SoCo
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yue Gao
# --------------------------------------------------------


import pickle as pkl
import torch
import argparse


def convert_detectron2_Head(input_file_name, output_file_name, start, num_outs, ema=False):
    ckpt = torch.load(input_file_name, map_location="cpu")
    if ema:
        state_dict = ckpt["model_ema"]
        backbone_prefix = "encoder."
        fpn_prefix = "neck."
        head_prefix = "head."
    else:
        state_dict = ckpt["model"]
        backbone_prefix = "module.encoder."
        fpn_prefix = "module.neck."
        head_prefix = "module.head."

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(backbone_prefix):
            old_k = k
            k = k.replace(backbone_prefix, "")
            if "layer" not in k:
                k = "stem." + k
            k = k.replace("layer1", "res2")
            k = k.replace("layer2", "res3")
            k = k.replace("layer3", "res4")
            k = k.replace("layer4", "res5")
            k = k.replace("bn1", "conv1.norm")
            k = k.replace("bn2", "conv2.norm")
            k = k.replace("bn3", "conv3.norm")
            k = k.replace("downsample.0", "shortcut")
            k = k.replace("downsample.1", "shortcut.norm")
            print(old_k, "->", k)
            new_state_dict[k] = v.numpy()

        elif k.startswith(fpn_prefix):
            old_k = k
            k = k.replace(fpn_prefix, "")

            for i in range(num_outs):
                k = k.replace(f"lateral_convs.{i}.conv", f"fpn_lateral{start+i}")
                k = k.replace(f"lateral_convs.{i}.bn", f"fpn_lateral{start+i}.norm")

                k = k.replace(f"fpn_convs.{i}.conv", f"fpn_output{start+i}")
                k = k.replace(f"fpn_convs.{i}.bn", f"fpn_output{start+i}.norm")

            print(old_k, "->", k)
            new_state_dict[k] = v.numpy()

        elif k.startswith(head_prefix):
            old_k = k
            k = k.replace(head_prefix, "box_head.")
            k = k.replace("bn1", "conv1.norm")
            k = k.replace("ac1", "conv1.activation")
            k = k.replace("bn2", "conv2.norm")
            k = k.replace("ac2", "conv2.activation")
            k = k.replace("bn3", "conv3.norm")
            k = k.replace("ac3", "conv3.activation")
            k = k.replace("bn4", "conv4.norm")
            k = k.replace("ac4", "conv4.activation")
            print(old_k, "->", k)
            new_state_dict[k] = v.numpy()

    res = {"model": new_state_dict,
           "__author__": "Yue",
           "matching_heuristics": True}

    with open(output_file_name, "wb") as f:
        pkl.dump(res, f)
    print(f"Saved converted detectron2 Head checkpoint to {output_file_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert Models')
    parser.add_argument('input', metavar='I',
                        help='input model path')
    parser.add_argument('output', metavar='O',
                        help='output path')
    parser.add_argument('start', metavar='S', type=int,
                        help='FPN start')
    parser.add_argument('num_outs', metavar='N', type=int,
                        help='FPN number of outputs')
    parser.add_argument('--ema', action='store_true',
                        help='using ema model')
    args = parser.parse_args()
    convert_detectron2_Head(args.input, args.output, args.start, args.num_outs, args.ema)
