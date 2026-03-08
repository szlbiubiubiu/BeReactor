#!/bin/bash
##SBATCH -A pilot_andrena
##SBATCH -p andrena  #Andrena 40g

#SBATCH -A pilot_sae_gpu
#SBATCH -p sae  ##Andrena 80g

###SBATCH -A pilot_gpu
###SBATCH -p gpu  #Apocrita
###SBATCH -p gpushort

#SBATCH -t 24:0:0 # 10 hour runtime
#SBATCH --mem-per-cpu=11G # 11 * 8 = 88G total RAM
##SBATCH -n 12 # 8 cores (8 cores per GPU)
#SBATCH -n 1          # 1 Task (Because you are running 1 torchrun command)
#SBATCH -c 12
#SBATCH --gres=gpu:1 # request 2 GPUs

##SBATCH -J environment_name
##SBATCH --constraint="ampere"
##SBATCH --mail-type=ALL

# module load miniforge
#set -e
source ~/.bashrc
# conda activate
conda activate deepface_v3

python retinaface_video_extract.py --input /gpfs/scratch/acw569/dataset/2025data_v2/val/video-face-crop --outdir /gpfs/scratch/acw569/dataset/2025data_v2/val/ZL_crop_test --show-progress
