#!/bin/bash
#data_dir="/path/to/imagenet"
data_dir="/home/lue_fan/imagenet100"
output_dir="./output/baseline_v1"

python -m torch.distributed.launch --master_port 12341 --nproc_per_node=8 \
    train.py \
    --batch-size 128 \
    --data-dir ${data_dir} \
    --dataset imagenet \
    --nce-k 65536 \
    --output-dir ${output_dir} \
#    --hsd


python -m torch.distributed.launch --master_port 12342 --nproc_per_node=8 \
    eval.py \
    --dataset imagenet \
    --data-dir ${data_dir} \
    --pretrained-model ${output_dir}/current.pth \
    --output-dir ${output_dir}/eval

