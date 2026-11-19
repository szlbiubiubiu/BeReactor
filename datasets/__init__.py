"""Dataset construction for reaction training."""

from .reaction_dataset import ReactionDataset_2025_readbothcrop


def get_dataset(args):
    if args.dataset != "reaction":
        raise ValueError(f"Unsupported training dataset: {args.dataset}")
    return ReactionDataset_2025_readbothcrop(
        root_dir=args.data_path,
        split="train",
        clip_length=args.num_frames,
        target_size=args.image_size,
        crop_size=args.image_size,
        fps=args.fps,
        interval=args.frame_interval,
        load_audio=True,
        load_video_s=True,
        load_video_l=True,
        load_emotion_s=True,
        load_emotion_l=True,
        load_3dmm_s=False,
        load_3dmm_l=False,
    )
