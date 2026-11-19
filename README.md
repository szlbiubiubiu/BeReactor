# BeReactor

This repository contains the **training code** for the paper:

> [**BeReactor: Audio-visual Appropriate Human-like Expressive Behavioural Response Generation**](https://github.com/szlbiubiubiu/latte_reaction/)
> [Zhonglin Sun](https://szlbiubiubiu.github.io/), [Xun Gong](https://scholar.google.com/citations?user=7PqgUw4AAAAJ&hl=en), [Feng Chen](https://scholar.google.com/citations?user=4VZhTYgAAAAJ&hl=zh-CN), [Siyang Song*](https://scholar.google.com/citations?user=ZKSL1IcAAAAJ&hl=en) (*Corresponding Author)

BeReactor generates listener facial reactions from speaker video and audio using a Latent Diffusion Transformer (Latte).

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/szlbiubiubiu/latte_reaction.git
cd latte_reaction
```

### 2. Create a Conda environment

```bash
conda create -n latte_new python=3.10
conda activate latte_new
```

### 3. Install dependencies

Install PyTorch with CUDA 11.8 first:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
```

### 4. Set up MuseTalk (Talking Head Synthesis, optional)

MuseTalk is used for downstream talking head synthesis and is **not required for training**. Its inference and demo code is excluded from this training repository. If needed separately, download the [pre-packaged MuseTalk folder](https://drive.google.com/file/d/13_Of4eWEgwkm3vMJAuzKIZsdMe7IMehM/view?usp=sharing).

## Training

Run all commands from the repository root. Training requires a CUDA GPU; the entry point is `train.py` and the Slurm launcher is `hpc_train_slurm.sh`.

Prepare the REACT 2025 dataset in the layout expected by the training loader:

```text
/path/to/dataset/
└── train/
    ├── audio/{speaker,listener}/session*/*.wav
    ├── video-face-crop/{speaker,listener}/session*/*.mp4
    ├── ZL_crop_test/{speaker,listener}/session*/*.mp4
    └── facial-attributes/{speaker,listener}/session*/*.npy
```

Each speaker/listener pair must share the session and filename stem across these directories. `ZL_crop_test` contains the additional face crops read by the model. Attribute arrays contain 25 emotion values per frame. Set `fps` in the configuration to match the source videos.

Place the VAE in `pretrained_models/vae/` (including its Diffusers configuration and weights). The facial encoders also expect `pretrained_models/model_ir_se50.pth`, `pretrained_models/79999_iter.pth`, and `pretrained_models/emonet_8.pth`. The AU backbone expects `checkpoints/resnet18-5c106cde.pth`. These data and weight files are not included.

Edit `configs/reaction/reaction_video.yaml` before launching. Set `data_path`, `pretrained_model_path`, and `results_dir`; adjust `local_batch_size`, `num_workers`, `num_frames`, `frame_interval`, `learning_rate`, `max_train_steps`, and `ckpt_every` as needed. Facial encoder paths remain relative to the repository root.

For a local single-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
    --config configs/reaction/reaction_video.yaml \
    --data-path /path/to/dataset \
    --pretrained-model-path ./pretrained_models \
    --results-dir ./results
```

For Slurm, edit the account, partition, GPU constraint, time, memory, and GPU count in `hpc_train_slurm.sh` to match your cluster, then submit:

```bash
sbatch hpc_train_slurm.sh \
    --config configs/reaction/reaction_video.yaml \
    --data-path /path/to/dataset \
    --results-dir ./results
```

The script activates `latte_new` by default; set `CONDA_ENV` to use a different environment. For multiple GPUs on one node, request the same GPU count in `#SBATCH --gres` and `NPROC_PER_NODE`, for example:

```bash
NPROC_PER_NODE=2 sbatch --gres=gpu:2 hpc_train_slurm.sh --data-path /path/to/dataset
```

`local_batch_size` is per GPU. Logs, TensorBoard events, the saved configuration, and `checkpoints/*.pt` are written under `results_dir/<experiment>/`. Checkpoints are saved every `ckpt_every` steps and at the final step.

To continue training, set `resume_from_checkpoint: true` and `resume_ckpt_path: /path/to/checkpoint.pt` in the YAML. New checkpoints restore model weights, EMA, optimizer, scheduler, and step count; older EMA-only checkpoints load weights and restart the optimizer and step count. To initialize a new run from weights, set `pretrained` instead.
