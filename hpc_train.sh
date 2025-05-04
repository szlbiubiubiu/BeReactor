#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=1:0:0     # XX hours 
#$ -l h_vmem=7.5G      # 11G RAM per core
#$ -pe smp 8          # 8 cores per GPU
#$ -l gpu=1           # request 1 GPU
##$ -l gpu_type=hopper   
#$ -l gpu_type=ampere   
##$ -l cluster=andrena
##$ -m se
#$-l rocky
# bash
# module load anaconda3/2020.02
source ~/.bashrc
# conda activate
conda activate latte_new

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 train.py
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 train.py