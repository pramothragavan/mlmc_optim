"""Helper utilities for training."""

import random
import numpy as np
import torch

def get_arg_list(arg):
    """Convert single value or list to list."""
    if isinstance(arg, list):
        return arg
    return [arg]


def get_device(config) -> str:
    """Auto-detect best available device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if config['device'] == 'gpu':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
    elif config['device'] == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
    elif config['device'] == 'mps':
        if torch.backends.mps.is_available():
            return 'mps'
    else:
        return 'cpu'


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

