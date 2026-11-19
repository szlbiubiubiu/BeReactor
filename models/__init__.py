"""Latte models used for reaction training."""

from .latte import Latte_models


def get_models(args):
    if args.model not in Latte_models:
        raise ValueError(f"Unsupported training model: {args.model}")
    return Latte_models[args.model](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        learn_sigma=args.learn_sigma,
        extras=args.extras,
    )
