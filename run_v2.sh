#!/bin/bash
data_dir="/home/lue_fan/imagenet100"
output_dir="./output/baseline_v2"

python -m torch.distributed.launch --master_port 12343 --nproc_per_node=8 \
    train.py \
    --mlp \
    --aug v2 \
    --batch-size 128 \
    --data-dir ${data_dir} \
    --dataset imagenet \
    --nce-k 65536 \
    --output-dir ${output_dir} \
#    --hsd

python -m torch.distributed.launch --master_port 12344 --nproc_per_node=8 \
    eval.py \
    --dataset imagenet \
    --data-dir ${data_dir} \
    --pretrained-model ${output_dir}/current.pth \
    --output-dir ${output_dir}/eval

