# disclaimer: inspired by MoCo official repo.
import torch
import argparse


def convert_mmdetection_Backbone(input_file_name, output_file_name, ema=False):
    ckpt = torch.load(input_file_name, map_location="cpu")
    if ema:
        state_dict = ckpt["model_ema"]
        backbone_prefix = "encoder."
    else:
        state_dict = ckpt["model"]
        backbone_prefix = "module.encoder."

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(backbone_prefix):
            old_k = k
            k = k.replace(backbone_prefix, "backbone.")
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
    convert_mmdetection_Backbone(args.input, args.output, args.ema)
