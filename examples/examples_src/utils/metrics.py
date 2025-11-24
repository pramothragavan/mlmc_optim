"""Loss functions and evaluation metrics for neural operators."""

import torch
import torch.nn as nn
import numpy as np


class LpLoss:
    """Lp relative loss for neural operators.
    
    Args:
        d: Spatial dimension
        p: Lp norm order
        reduction: Whether to reduce to scalar
        size_average: Whether to average over batch
    """
    
    def __init__(self, d: int = 2, p: int = 2, reduction: str = 'mean'):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute relative Lp loss."""
        num_examples = x.size(0)
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 
            self.p, 
            1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        
        if self.reduction == 'mean':
            return torch.mean(diff_norms / y_norms)
        elif self.reduction == 'sum':
            return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)


class GINOTLoss(torch.nn.Module):
    def __init__(self, padding_value=-1000, reduction='none'):
        super().__init__()
        self.padding_value = padding_value
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, output, target):
        """Calculate loss with padding masking

        Args:
            output: Model output
            target: Target values
            reduction: 'none', 'mean', or 'sum' - how to reduce the loss

        Returns:
            If reduction='mean': scalar mean loss
            If reduction='sum': scalar sum of losses
            If reduction='none': per-example losses of shape [batch_size]
        """
        mask = (target != self.padding_value).float()
        element_loss = self.mse(output, target)
        masked_loss = element_loss * mask

        # Sum over all dimensions except batch dimension
        dims = tuple(range(1, masked_loss.dim()))
        per_example_loss = masked_loss.sum(dim=dims)
        per_example_norm = mask.sum(dim=dims) + 1
        per_example_loss = per_example_loss / per_example_norm

        # Apply reduction based on parameter
        if self.reduction == 'none':
            return per_example_loss
        elif self.reduction == 'sum':
            return per_example_loss.sum()
        else:  # 'mean' (default, backward compatible)
            return per_example_loss.mean()