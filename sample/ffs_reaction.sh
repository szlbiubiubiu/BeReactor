#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1 python sample/evaluate_reaction.py \
--save_video_path ./test
# --config ./configs/ffs/ffs_sample.yaml \
# --ckpt ./share_ckpts/ffs.pt \
