#!/bin/bash
#SBATCH --job-name=bereactor-train
#SBATCH --account=pilot_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=ampere
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=11G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1

set -eo pipefail

# Submit from the repository root so model and data paths resolve consistently.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
if [ -f "${HOME}/.bashrc" ]; then
    source "${HOME}/.bashrc"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-latte_new}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-1}" train.py "$@"
