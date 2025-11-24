"""
Simplified GINOT model components for JEB dataset evaluations.
This file contains the necessary model components from the GINOT repository,
adapted to run without requiring the full GINOT package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import lru_cache
from typing import Optional, List


# Position encoding functions from GINOT
def posenc_nerf(x: torch.Tensor, min_deg: int = 0, max_deg: int = 15) -> torch.Tensor:
    """Concatenate x and its positional encodings, following NeRF."""
    if min_deg == max_deg:
        return x
    scales = get_scales(min_deg, max_deg, x.dtype, x.device)
    *shape, dim = x.shape
    xb = (x.reshape(-1, 1, dim) * scales.view(1, -1, 1)).reshape(*shape, -1)
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    emb = torch.cat([xb, xb + math.pi / 2.0], axis=-1).sin()
    return torch.cat([x, emb], dim=-1)


@lru_cache
def get_scales(min_deg: int, max_deg: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return 2.0 ** torch.arange(min_deg, max_deg, device=device, dtype=dtype)


def encode_position(version: str, *, position: torch.Tensor):
    """Encode position with the specified scheme."""
    if version == "nerf":
        return posenc_nerf(position)
    raise ValueError(f"Unknown position encoding: {version}")


def position_encoding_channels(version: Optional[str] = None):
    """Return the number of channels used by a position encoding."""
    if version == "nerf":
        return 2 * 15 + 1  # Includes original coordinates
    return 1


# Simplified transformer components needed for the model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width)
        )
        
    def forward(self, x):
        return x + self.layer(x)


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln = nn.LayerNorm(width)
        
    def forward(self, x, context=None, mask=None):
        x_norm = self.ln(x)
        if context is None:
            context = x_norm
        attn_output, _ = self.attn(x_norm, context, context, key_padding_mask=mask)
        return x + attn_output


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout=0.0):
        super().__init__()
        self.attn = AttentionBlock(width, heads)
        self.mlp = ResidualBlock(width)
        
    def forward(self, x, context=None, mask=None):
        x = self.attn(x, context, mask)
        x = self.mlp(x)
        return x


# Simplified PointCloud encoder
class PointCloudPerceiverChannelsEncoder(nn.Module):
    """Simplified version of GINOT's point cloud encoder."""
    def __init__(
        self,
        config,
        in_dim=3,          # Input dimension (3D points)
        global_pooling=True,
        latent_dim=64,     # Latent dimension
        embed_dim=64,      # Embedding dimension
        num_layers=4,      # Number of layers
        num_heads=4,       # Number of attention heads
        padding_value=-1000,
        dropout=0.0,
    ):
        super().__init__()
        self.config = config
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.padding_value = padding_value
        self.global_pooling = global_pooling
        
        # Position embedding
        self.position_encoder = nn.Sequential(
            nn.Linear(in_dim * position_encoding_channels("nerf"), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Self-attention blocks
        self.attention_blocks = nn.ModuleList([
            ResidualCrossAttentionBlock(
                width=embed_dim,
                heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, latent_dim)
        
    def forward(self, points, sample_ids=None):
        """
        Args:
            points (torch.Tensor): [B, N, C] point cloud
            sample_ids: Not used in this simplified version
        """
        # Create mask for padding
        if self.padding_value is not None:
            mask = (points[:, :, 0] == self.padding_value)
        else:
            mask = None
        
        # Encode positions
        points_encoded = encode_position("nerf", position=points)
        x = self.position_encoder(points_encoded)
        
        # Apply self-attention
        for block in self.attention_blocks:
            if self.config['use_gradient_checkpointing']:
                x = torch.utils.checkpoint.checkpoint(block.forward, x, None, mask)
            else:
                x = block(x, mask=mask)
        
        # Global pooling or token output
        if self.global_pooling:
            if mask is not None:
                # Average only over non-padding tokens
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x = x.masked_fill(mask_expanded, 0)
                count = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
                x = x.sum(dim=1) / count
            else:
                x = x.mean(dim=1)  # [B, embed_dim]
            
            # Project to latent dimension
            latent = self.output_proj(x)  # [B, latent_dim]
            return latent.unsqueeze(1)  # [B, 1, latent_dim]
        else:
            # Return token embeddings
            return x  # [B, N, embed_dim]


# Main Trunk model from GINOT
class Trunk(nn.Module):
    """Simplified version of GINOT's Trunk model."""
    def __init__(
        self,
        config,
        branch, 
        embed_dim=64, 
        cross_attn_layers=4, 
        num_heads=4,
        in_channels=3, 
        out_channels=1,
        dropout=0.0, 
        emd_version="nerf", 
        padding_value=-1000
    ):
        super().__init__()
        self.config = config
        self.padding_value = padding_value
        d = position_encoding_channels(emd_version)
        
        # Query encoder for target points
        self.Q_encoder = nn.Sequential(
            nn.Linear(d*in_channels, 2*embed_dim),
            nn.ReLU(),
            nn.Linear(2*embed_dim, 3*embed_dim),
            nn.ReLU(),
            nn.Linear(3*embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.Linear(2*embed_dim, embed_dim)
        )
        
        # Point cloud encoder branch
        self.branch = branch
        
        # Cross-attention blocks
        self.resblocks = nn.ModuleList([
            ResidualCrossAttentionBlock(
                width=embed_dim,
                heads=num_heads,
                dropout=dropout
            )
            for _ in range(cross_attn_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*embed_dim, 3*embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3*embed_dim, 3*embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3*embed_dim, 2*embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*embed_dim, out_channels)
        )

    def forward(self, batch_data, sample_ids=None, return_mask=False):
        """
        Args:
            batch_data: tuple or list, where batch_data[0] is pc_batch (list of [M_i, in_channels]),
                        batch_data[1] is xy_batch (list of [N_i, in_channels])
            sample_ids: Not used in this simplified version
            return_mask: if True, returns (output, mask). Otherwise, returns output only.
        Returns:
            output: [B, max_N] tensor (padded)
            mask: [B, max_N] bool tensor (True for valid, False for padding)
        """
        pc_batch, xy_batch = batch_data[0], batch_data[1]
        device = next(self.parameters()).device
        # Move all input tensors to the model's device
        xy_batch = [xy.to(device) for xy in xy_batch]
        pc_batch = [pc.to(device) for pc in pc_batch]
        # Pad pc and xy
        max_pc_len = max(pc.shape[0] for pc in pc_batch)
        max_xy_len = max(xy.shape[0] for xy in xy_batch)
        in_channels = pc_batch[0].shape[1]
        padded_pc = []
        for pc in pc_batch:
            if pc.shape[0] < max_pc_len:
                padding = torch.full((max_pc_len - pc.shape[0], in_channels), self.padding_value, dtype=pc.dtype, device=device)
                padded_pc.append(torch.cat([pc, padding], dim=0))
            else:
                padded_pc.append(pc)
        padded_xy = []
        for xy in xy_batch:
            if xy.shape[0] < max_xy_len:
                padding = torch.full((max_xy_len - xy.shape[0], in_channels), self.padding_value, dtype=xy.dtype, device=device)
                padded_xy.append(torch.cat([xy, padding], dim=0))
            else:
                padded_xy.append(xy)
        pc_tensor = torch.stack(padded_pc)  # [B, max_M, in_channels]
        xy_tensor = torch.stack(padded_xy)  # [B, max_N, in_channels]

        # Encode point cloud
        if self.config['use_gradient_checkpointing']:
            # Need to wrap branch.forward since branch is a module
            latent = torch.utils.checkpoint.checkpoint(self.branch.forward, pc_tensor, sample_ids)  # [B, 1, latent_dim]
            def encode_and_query(xy):
                xyt_enc = encode_position('nerf', position=xy)
                return self.Q_encoder(xyt_enc)
            x = torch.utils.checkpoint.checkpoint(encode_and_query, xy_tensor)
            # Apply cross-attention
            for block in self.resblocks:
                def block_forward(x, context):
                    return block(x, context=context)
                x = torch.utils.checkpoint.checkpoint(block_forward, x, latent)
        else:
            latent = self.branch(pc_tensor, sample_ids=sample_ids)  # [B, 1, latent_dim]
            # Encode target points
            xyt_enc = encode_position('nerf', position=xy_tensor)
            x = self.Q_encoder(xyt_enc)  # [B, max_N, embed_dim]
            # Apply cross-attention
            for block in self.resblocks:
                x = block(x, latent)  # [B, max_N, embed_dim]

        # Generate output
        x = self.output_proj(x)  # [B, max_N, out_channels]
        output = x.squeeze(-1)  # [B, max_N]
        # Create mask for valid (non-padding) outputs
        mask = torch.zeros(output.shape, dtype=torch.bool, device=output.device)
        for i, xy in enumerate(xy_batch):
            mask[i, :xy.shape[0]] = True
        if return_mask:
            return output, mask
        else:
            return output

    def inverse_transform(self, output, mask, inverse_fn=None, **kwargs):
        """
        Optionally apply an inverse transform (e.g., unnormalize) to valid outputs only.
        Args:
            output: [B, max_N] tensor
            mask: [B, max_N] bool tensor
            inverse_fn: callable, e.g., lambda x: ...
            kwargs: extra arguments for inverse_fn
        Returns:
            output_inv: same shape as output, with inverse_fn applied to valid entries
        """
        if inverse_fn is None:
            return output
        output_inv = output.clone()
        output_inv[mask] = inverse_fn(output[mask], **kwargs)
        return output_inv


# Helper function to create and configure the complete GINOT model
def NOTModelDefinition(config, branch_args, trunk_args):
    """Create and return a configured GINOT model with branch and trunk components."""
    # Create the branch (point cloud encoder)
    branch = PointCloudPerceiverChannelsEncoder(config, **branch_args)
    
    # Log parameter counts
    tot_num_params = sum(p.numel() for p in branch.parameters())
    trainable_params = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    print(f"Total number of parameters of Geo encoder: {tot_num_params}, {trainable_params} of which are trainable")
    
    # Create the trunk model with the branch as input
    trunk = Trunk(config, branch, **trunk_args)
    
    # Log parameter counts for the complete model
    tot_num_params = sum(p.numel() for p in trunk.parameters())
    trainable_params = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
    print(f"Total number of parameters of NOT model: {tot_num_params}, {trainable_params} of which are trainable")
    
    return trunk
