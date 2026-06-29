# BeReactor

This repository contains the **evaluation code** for the paper:

> [**BeReactor: Audio-visual Appropriate Human-like Expressive Behavioural Response Generation**](https://github.com/szlbiubiubiu/latte_reaction/)
> [Zhonglin Sun](https://szlbiubiubiu.github.io/), [Xun Gong](https://scholar.google.com/citations?user=7PqgUw4AAAAJ&hl=en), [Feng Chen](https://scholar.google.com/citations?user=4VZhTYgAAAAJ&hl=zh-CN), [Siyang Song*](https://scholar.google.com/citations?user=ZKSL1IcAAAAJ&hl=en) (*Corresponding Author)

BeReactor is an audio-visual reaction generation system built on a Latent Diffusion Transformer (Latte). Given a speaker video, it generates appropriate and expressive listener facial reactions, optionally guided by LLM-predicted emotion tokens and talking head synthesis.

---

## Repository Structure

```
.
├── configs/
│   ├── reaction/               # Evaluation config for React 2024
│   └── reaction2025/           # Evaluation config for React 2025
├── datasets/
│   └── reaction_dataset.py     # Dataloader for the reaction dataset
├── diffusion/                  # Diffusion utilities (gaussian diffusion, respace, etc.)
├── metric/                     # Evaluation metrics (FRC, FRD, FRDvs, FRVar, TLCC, SMSE)
├── models/                     # Model definitions (Latte backbone, EmoNet, IDNet, etc.)
├── sample/                     # Evaluation scripts
│   ├── evaluation_v2vreaction.py           # Main evaluation: video-to-video reaction
│   ├── evaluation_v2vreaction_language.py  # Evaluation with LLM emotion guidance
│   ├── evaluation_v2v_singlecase.py        # Single-case evaluation / debugging
│   ├── evaluation_do_online.py             # Online evaluation pipeline
│   ├── reaction_eva_all_withtalkinghead.py # Full eval with talking head synthesis
│   ├── evaluate_groundtruth.py             # Compute metrics on ground truth
│   ├── evaluate_reaction.py                # Compute reaction-level metrics
│   ├── evaluate_utmos_folder.py            # Audio quality metric (UTMOSv2)
│   └── evaluate_vbench_folder.py           # Video quality metric (VBench)
├── talkinghead/                # Talking head modules (MuseTalk, EchoMimic)
├── tools/                      # Utility tools (face detection, video/audio merge)
├── eva_from_local.py           # Evaluate from pre-saved emotion tensors
├── hpc_evaluation.sh           # SGE cluster evaluation script
├── hpc_evaluation_slurm.sh     # Slurm cluster evaluation script
└── utils.py / util.py          # Shared utilities
```

---

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
```

Then install the remaining packages from `requirement_nollama.txt`:

```bash
pip install timm
pip install diffusers[torch]==0.24.0
pip install accelerate==0.22.0
pip install tensorboard einops
pip install transformers==4.46.1
pip install av scikit-image decord pandas imageio-ffmpeg omegaconf soundfile
pip install numpy==1.26.4
pip install opencv-python==4.10.0.84
pip install tslearn openai-whisper piper-tts face_alignment
pip install hydra-core
pip install "huggingface_hub<0.23.0"
```

### 4. Set up MuseTalk (Talking Head Synthesis)

MuseTalk is required for the full evaluation pipeline with talking head synthesis. Download the pre-packaged MuseTalk folder from the Google Drive link below and copy it to `talkinghead/muse/`:

**Download link:** https://drive.google.com/file/d/13_Of4eWEgwkm3vMJAuzKIZsdMe7IMehM/view?usp=sharing

### 5. Download Reaction Pretrained models
5.1 external/
unzip and place into ./external
5.2 pretrained models
unzip and place into ./pretrained_models
### 5. Download BeReactor pretrained checkpoint

Download the BeReactor model checkpoint and pass its path via `--ckpt` when running evaluation.

---

## Evaluation

### Main Evaluation (Video-to-Video Reaction Generation)

Generates listener reaction videos and computes reaction quality metrics (TLCC, FRC, FRD, FRDvs, FRVar, SMSE):

```bash
CUDA_VISIBLE_DEVICES=0,1 python sample/evaluation_v2vreaction.py \
    --config ./configs/reaction/reaction_test.yaml \
    --ckpt /path/to/checkpoint.pt \
    --save_video_path ./output/
```

### Evaluation with LLM Emotion Guidance

Uses an LLM to predict emotion tokens for listener generation:

```bash
CUDA_VISIBLE_DEVICES=0,1 python sample/evaluation_v2vreaction_language.py \
    --config ./configs/reaction/reaction_test.yaml \
    --ckpt /path/to/checkpoint.pt \
    --save_video_path ./output/
```

### Full Pipeline with Talking Head Synthesis

Generates listener reaction videos with full talking head synthesis (MuseTalk):

```bash
CUDA_VISIBLE_DEVICES=0,1 python sample/reaction_eva_all_withtalkinghead.py \
    --save_video_path ./output/
```

Or with `torchrun`:

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=1 \
    sample/reaction_eva_all_withtalkinghead.py --save_video_path ./output/
```

### Single-Case Evaluation / Debugging

```bash
CUDA_VISIBLE_DEVICES=0 python sample/evaluation_v2v_singlecase.py \
    --config ./configs/reaction/reaction_test.yaml \
    --ckpt /path/to/checkpoint.pt \
    --save_video_path ./output/
```


## Metrics

| Metric | Description |
|--------|-------------|
| **TLCC** | Temporal Linear Cross-Correlation — synchrony between speaker and listener emotions |
| **FRC** | Facial Reaction Correlation — overall emotion correlation |
| **FRD** | Facial Reaction Diversity — diversity across generated reactions |
| **FRDvs** | FRD vs. ground truth |
| **FRVar** | Facial Reaction Variance — expressiveness of generated reactions |
| **SMSE** | Semantic Mean Squared Error — semantic alignment |
| **UTMOSv2** | Audio naturalness (MOS score) |

---

## Citation

<!-- If you find this work useful for your research, please consider citing:

```bibtex
@article{sun2024bereactor,
  title={BeReactor: Audio-visual Appropriate Human-like Expressive Behavioural Response Generation},
  author={Sun, Zhonglin and Gong, Xun and Chen, Feng and Song, Siyang},
  year={2024}
}
``` -->

---

## License

The code is licensed under [LICENSE](LICENSE).
