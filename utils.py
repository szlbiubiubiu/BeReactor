"""Gradient clipping, logging and distributed training helpers."""

import logging
import os
from collections import OrderedDict
from typing import Iterable, Union

import torch
import torch.distributed as dist
from torch import inf
from torch.utils.tensorboard import SummaryWriter

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad = True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)

    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def create_tensorboard(tensorboard_dir):
    """
    Create a tensorboard that saves losses.
    """
    if dist.get_rank() == 0:
        return SummaryWriter(tensorboard_dir)
    return None

def write_tensorboard(writer, *args):
    '''
    write the loss information to a tensorboard file.
    Only for pytorch DDP mode.
    '''
    if dist.get_rank() == 0:  # real tensorboard
        writer.add_scalar(args[0], args[1], args[2])

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def setup_distributed(backend="nccl"):
    """Initialize the process group using the environment set by torchrun."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
