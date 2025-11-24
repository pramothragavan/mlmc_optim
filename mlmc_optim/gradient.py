"""MLMC gradient estimation."""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


def mlmc_gradient_estimate(
    model: nn.Module,
    criterion: nn.Module,
    data_fine: torch.Tensor,
    target_fine: torch.Tensor,
    data_coarse: Optional[torch.Tensor] = None,
    target_coarse: Optional[torch.Tensor] = None,
    train_mean: Optional[torch.Tensor] = None,
    train_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute MLMC gradient estimate.
    
    Two modes:
    1. Coarse only (data_coarse=None): Compute loss for coarsest level
    2. Pair (data_coarse provided): Compute loss difference (fine - coarse)
    
    Args:
        model: Neural operator model
        criterion: Loss function
        data_fine: Fine resolution data
        target_fine: Fine resolution targets
        data_coarse: Coarse resolution data (optional)
        target_coarse: Coarse resolution targets (optional)
        train_mean: Mean for denormalization (optional)
        train_std: Std for denormalization (optional)
        
    Returns:
        Loss tensor (scalar)
    """
    def decode(x, mean, std):
        """Denormalize predictions."""
        if mean is not None and std is not None:
            return x * (std + 1e-5) + mean
        return x
    
    if data_coarse is None:
        # Mode 1: Coarse level only
        output = model(data_fine)
        # Decode outputs when mean/std are provided
        if train_mean is not None and train_std is not None:
            output = decode(output, train_mean, train_std)
            target = decode(target_fine, train_mean, train_std)
        else:
            target = target_fine
        loss = criterion(output, target)
        return loss
    
    else:
        # Mode 2: Telescopic difference (fine - coarse)
        output_fine = model(data_fine)
        if train_mean is not None and train_std is not None:
            output_fine = decode(output_fine, train_mean, train_std)
            target_fine_decoded = decode(target_fine, train_mean, train_std)
        else:
            target_fine_decoded = target_fine
        loss_fine = criterion(output_fine, target_fine_decoded)
        
        output_coarse = model(data_coarse)
        if train_mean is not None and train_std is not None:
            output_coarse = decode(output_coarse, train_mean, train_std)
            target_coarse_decoded = decode(target_coarse, train_mean, train_std)
        else:
            target_coarse_decoded = target_coarse
        loss_coarse = criterion(output_coarse, target_coarse_decoded)
        
        # Telescopic difference
        diff_loss = loss_fine - loss_coarse
        
        return diff_loss
