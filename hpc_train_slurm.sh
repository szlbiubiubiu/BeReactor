#!/bin/bash
##SBATCH -A pilot_andrena
##SBATCH -p andrena  #Andrena 40g

#SBATCH -A pilot_sae_gpu
#SBATCH -p sae  ##Andrena 80g

###SBATCH -A pilot_gpu
###SBATCH -p gpu  #Apocrita
###SBATCH -p gpushort

#SBATCH -t 96:0:0 # 10 hour runtime
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

# Replace the following line with a program or command
conda activate latte_new

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master-port=25541 train.py

# For assistance, please email its-research-support@qmul.ac.uk
