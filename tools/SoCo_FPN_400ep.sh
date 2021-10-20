#!/bin/bash

set -e
set -x

warmup=${1:-10}
epochs=${2:-400}

data_dir="./data/ImageNet-Zip"
output_dir="./SoCo_output/SoCo_FPN_400ep"

master_addr=${MASTER_IP}
master_port=28652

python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes ${OMPI_COMM_WORLD_SIZE} --node_rank ${OMPI_COMM_WORLD_RANK} --master_addr ${master_addr} --master_port ${master_port} \
    SoCo/main_pretrain.py \
    --data_dir ${data_dir} \
    --crop 0.5 \
    --base_lr 1.0 \
    --optimizer lars \
    --weight_decay 1e-5 \
    --amp_opt_level O1 \
    --ss_props \
    --auto_resume \
    --aug ImageAsymBboxAwareMultiJitter1 \
    --zip --cache_mode no \
    --arch resnet50 \
    --model SoCo_FPN \
    --warmup_epoch ${warmup} \
    --epochs ${epochs} \
    --output_dir ${output_dir} \
    --save_freq 1 \
    --batch_size 128 \
    --contrast_momentum 0.99 \
    --filter_strategy ratio3size0308post \
    --select_strategy random \
    --select_k 4 \
    --output_size 7 \
    --aligned \
    --jitter_prob 0.5 \
    --jitter_ratio 0.1 \
    --padding_k 4 \
    --start_level 0 \
    --num_outs 4 \
    --add_extra_convs 0 \
    --extra_convs_on_inputs 0 \
    --relu_before_extra_convs 0 \
    --aware_start 0 \
    --aware_end 4 \
    --cutout_prob 0.5 \
    --cutout_ratio_min 0.1 \
    --cutout_ratio_max 0.3

# python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
#     main_linear.py \
#     --data_dir ${data_dir} \
#     --zip --cache_mode part \
#     --arch resnet50 \
#     --output_dir ${output_dir}/eval \
#     --pretrained_model ${output_dir}/current.pth \
