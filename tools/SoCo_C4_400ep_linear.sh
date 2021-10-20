#!/bin/bash

set -e
set -x

warmup=${1:-10}
epochs=${2:-400}

data_dir="./data/ImageNet-Zip"
output_dir="./self_det_output/SoCo_C4_400ep"

master_addr=${MASTER_IP}
master_port=28652


python -m torch.distributed.launch --master_port ${master_port} --nproc_per_node=8 \
    SoCo/main_linear.py \
    --data_dir ${data_dir} \
    --zip --cache_mode part \
    --arch resnet50 \
    --output_dir ${output_dir}_linear_eval \
    --pretrained_model ${output_dir}/current.pth \
    --save_freq 10 \
    --auto_resume
