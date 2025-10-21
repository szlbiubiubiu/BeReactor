#!/bin/bash
#$ -cwd
#$ -j y
#$ -l h_rt=1:0:0     # XX hours 
#$ -l h_vmem=7.5G      # 11G RAM per core
#$ -pe smp 8          # 8 cores per GPU
#$ -l gpu=1           # request 1 GPU
##$ -l gpu_type=hopper  
##$ -l gpu_type=volta 
##$ -l gpu_type=ampere   
##$ -l cluster=andrena
##$ -m se
#$-l rocky
# bash
# module load anaconda3/2020.02
source ~/.bashrc
# conda activate
conda activate py37

python retinaface_video_extract.py --input /data/scratch/acw569/face_react/2025data/train/video-face-crop --outdir /data/scratch/acw569/face_react/2025data/train/ZL_crop --show-progress
