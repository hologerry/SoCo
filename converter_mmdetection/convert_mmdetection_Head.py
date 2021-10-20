# disclaimer: inspired by MoCo official repo.
import torch
import argparse


def convert_mmdetection_Head(input_file_name, output_file_name, ema=False):
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
            k = k.replace(backbone_prefix, "backbone.")
            print(old_k, "->", k)
            new_state_dict[k] = v
        elif k.startswith(fpn_prefix):
            old_k = k
            k = k.replace(fpn_prefix, "neck.")

            print(old_k, "->", k)
            new_state_dict[k] = v
        elif k.startswith(head_prefix):
            old_k = k
            k = k.replace(head_prefix, "roi_head.bbox_head.")
            k = k.replace("conv1", "shared_convs.0.conv")
            k = k.replace("bn1", "shared_convs.0.bn")
            k = k.replace("ac1", "shared_convs.0.activate")

            k = k.replace("conv2", "shared_convs.1.conv")
            k = k.replace("bn2", "shared_convs.1.bn")
            k = k.replace("ac2", "shared_convs.1.activate")

            k = k.replace("conv3", "shared_convs.2.conv")
            k = k.replace("bn3", "shared_convs.2.bn")
            k = k.replace("ac3", "shared_convs.2.activate")

            k = k.replace("conv4", "shared_convs.3.conv")
            k = k.replace("bn4", "shared_convs.3.bn")
            k = k.replace("ac4", "shared_convs.3.activate")

            k = k.replace("fc1", "shared_fcs.0")

            print(old_k, "->", k)
            new_state_dict[k] = v

    res = {"state_dict": new_state_dict,
           "meta": {}}

    torch.save(res, output_file_name)
    print(f"Saved converted mmdetection load checkpoint to {output_file_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert Models')
    parser.add_argument('input', metavar='I',
                        help='input model path')
    parser.add_argument('output', metavar='O',
                        help='output path')
    parser.add_argument('--ema', action='store_true',
                        help='using ema model')
    args = parser.parse_args()
    convert_mmdetection_Head(args.input, args.output, args.ema)
