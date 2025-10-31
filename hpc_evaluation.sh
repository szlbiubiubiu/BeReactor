#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=1:0:0     # XX hours 
#$ -l h_vmem=7.5G      # 11G RAM per core
#$ -pe smp 16          # 8 cores per GPU
#$ -l gpu=2           # request 1 GPU
##$ -l gpu_type=hopper  
##$ -l gpu_type=volta 
#$ -l gpu_type=ampere   
##$ -l cluster=andrena
##$ -m se
#$-l rocky
# bash
# module load anaconda3/2020.02
source ~/.bashrc
# conda activate
conda activate latte_new

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5 torchrun --nnodes=1 --nproc_per_node=1  \
# sample/reaction_eva_all_withtalkinghead.py --save_video_path ./test

CUDA_VISIBLE_DEVICES=0,1 python sample/evaluation_v2vreaction.py --save_video_path /data/scratch/acw569/face_react/reaction2024_v4 #./test
# CUDA_VISIBLE_DEVICES=0,1 python sample/evaluation_v2v_singlecase.py --save_video_path /data/scratch/acw569/face_react/generated_videos_listenertrain #./test

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2  --rdzv_backend c10d --master_port 0 \
# sample/evaluation_v2vreaction.py --save_video_path ./test

# python -m torch.distributed.run --master_port 25473 