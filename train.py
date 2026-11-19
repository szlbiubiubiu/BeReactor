# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train BeReactor with PyTorch DistributedDataParallel and torchrun."""

import argparse
import math
import os
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import get_dataset
from diffusion import create_diffusion
from models import get_models
from utils import (
    cleanup,
    clip_grad_norm_,
    create_logger,
    create_tensorboard,
    requires_grad,
    setup_distributed,
    update_ema,
    write_tensorboard,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str,
        default=str(Path(__file__).parent / "configs/reaction/reaction_video.yaml"),
        help="Training YAML configuration.",
    )
    parser.add_argument("--data-path", help="Dataset root containing the train/ directory.")
    parser.add_argument("--pretrained-model-path", help="Directory containing vae/.")
    parser.add_argument("--results-dir", help="Output directory for logs and checkpoints.")
    return parser.parse_args()


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires at least one CUDA GPU.")
    if args.image_size % 8:
        raise ValueError("image_size must be divisible by 8 for the VAE encoder.")
    if args.max_train_steps <= 0 or args.local_batch_size <= 0:
        raise ValueError("max_train_steps and local_batch_size must be positive.")
    if args.ckpt_every <= 0 or args.log_every <= 0:
        raise ValueError("ckpt_every and log_every must be positive.")
    if args.extras != 3:
        raise ValueError("Reaction training requires extras=3.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    setup_distributed()
    rank = dist.get_rank()
    torch.manual_seed(args.global_seed + rank)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_name = args.model.replace("/", "-")
        experiment_dir = (
            f"{args.results_dir}/{experiment_index:03d}-{model_name}"
            f"-F{args.num_frames}S{args.frame_interval}-{args.dataset}"
        )
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, "config.yaml"))
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    args.latent_size = args.image_size // 8
    model = get_models(args)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        weights = checkpoint.get("ema", checkpoint)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        logger.info(f"Loaded pretrained weights: {args.pretrained}")
        logger.info(f"Missing keys: {missing}; unexpected keys: {unexpected}")

    model = DDP(model.to(device), device_ids=[local_rank], find_unused_parameters=True)
    ema = deepcopy(model.module).to(device)
    requires_grad(ema, False)
    ema.eval()
    diffusion = create_diffusion(timestep_respacing="", learn_sigma=args.learn_sigma)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    lr_scheduler = get_scheduler("constant", optimizer=opt)
    dataset = get_dataset(args)
    sampler = DistributedSampler(
        dataset, num_replicas=dist.get_world_size(), rank=rank,
        shuffle=True, seed=args.global_seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.local_batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("No complete training batches; check data_path and local_batch_size.")
    logger.info(f"Dataset contains {len(dataset):,} paired clips ({args.data_path})")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_steps = 0
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_ckpt_path, map_location="cpu")
        model.module.load_state_dict(checkpoint.get("model", checkpoint["ema"]))
        ema.load_state_dict(checkpoint["ema"])
        if "opt" in checkpoint and "train_steps" in checkpoint:
            opt.load_state_dict(checkpoint["opt"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            train_steps = checkpoint["train_steps"]
        else:
            logger.info("Legacy checkpoint: loaded EMA weights; restarting optimizer and step count.")
        logger.info(f"Loaded checkpoint {args.resume_ckpt_path} at step {train_steps}")

    model.train()
    first_epoch, resume_step = divmod(train_steps, len(loader))
    num_train_epochs = math.ceil(args.max_train_steps / len(loader))
    running_loss = 0.0
    log_steps = 0
    start_time = time()

    def save_checkpoint():
        if rank == 0:
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save({
                "model": model.module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_steps": train_steps,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        dist.barrier()

    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            if epoch == first_epoch and step < resume_step:
                continue
            if train_steps >= args.max_train_steps:
                break

            listener_video = video_data["listener_video_clip"].to(device, non_blocking=True)
            with torch.no_grad():
                latents = torch.stack([
                    vae.encode(clip).latent_dist.sample().mul_(0.18215)
                    for clip in listener_video
                ])
            t = torch.randint(0, diffusion.num_timesteps, (latents.shape[0],), device=device)
            loss_dict = diffusion.training_losses(
                model, latents, t, model_kwargs={"y": None},
                original_x=video_data["speaker_video_clip_crop"].to(device, non_blocking=True),
                listener_ref=video_data["listener_video_clip_crop"].to(device, non_blocking=True),
                listener_video_clip_last=video_data["listener_video_clip_last"].to(device, non_blocking=True),
                sp_audio_clip=video_data["speaker_audio_clip"].to(device, non_blocking=True),
            )
            loss = loss_dict["loss"].mean()
            loss.backward()
            gradient_norm = clip_grad_norm_(
                model.module.parameters(), args.clip_max_norm,
                clip_grad=train_steps >= args.start_clip_iter,
            )
            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            update_ema(ema, model.module)
            train_steps += 1
            running_loss += loss.item()
            log_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}/epoch={epoch:04d}) Loss: {avg_loss:.4f}, "
                    f"Gradient Norm: {gradient_norm:.4f}, Steps/Sec: {log_steps / elapsed:.2f}"
                )
                write_tensorboard(tb_writer, "Train Loss", avg_loss, train_steps)
                write_tensorboard(tb_writer, "Gradient Norm", gradient_norm, train_steps)
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0:
                save_checkpoint()

    if train_steps % args.ckpt_every != 0:
        save_checkpoint()
    if tb_writer is not None:
        tb_writer.close()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    cli_args = parse_args()
    config = OmegaConf.load(cli_args.config)
    for key in ("data_path", "pretrained_model_path", "results_dir"):
        override = getattr(cli_args, key)
        if override is not None:
            config[key] = override
    main(config)
