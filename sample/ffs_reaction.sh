#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5

# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=1  \
# sample/evaluate_reaction.py --save_video_path ./test
# --config ./configs/ffs/ffs_sample.yaml \
# --ckpt ./share_ckpts/ffs.pt \



OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=1  \
sample/reaction_eva_all_withtalkinghead.py --save_video_path ./test
