#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=72:0:0     # XX hours runtime
#$ -l h_vmem=7.5G      # 11G RAM per core
#$ -pe smp 12          # 8 cores per GPU
#$ -l gpu=1           # request 1 GPU
##$ -l gpu_type=hopper   
#$ -l gpu_type=ampere   
##$ -l cluster=andrena
##$ -m se
# bash
# module load anaconda3/2020.02
source ~/.bashrc
# conda activate
conda activate latte_new

torchrun --nnodes=1 --nproc_per_node=2 train.py