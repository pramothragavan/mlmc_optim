"""Parabolic CNN for resolution-invariant PDE solving."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.examples_src.utils.utils import get_arg_list
from .parabolic_conv import ParabolicConv


class CNNClassifier(nn.Module):
    """Standard CNN for comparison"""

    def __init__(self, config, in_channels=1, hidden_dim=32, out_size=10, stride_config=(1, 1), input_size=28,
                 loss_type='ce', max_pool=False):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=stride_config[0], padding=1),
            # padding=1 to maintain size
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride_config[1], padding=1),
            # padding=1 to maintain size
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
        )

        # Calculate output size after convolutions and pooling
        conv1_size = input_size // stride_config[0]  # No size change with padding=1
        if max_pool:
            conv1_size = conv1_size // 2

        conv2_size = conv1_size // stride_config[1]  # No size change with padding=1
        if max_pool:
            conv2_size = conv2_size // 2

        self.max_pool = max_pool
        self.loss_type = loss_type

        # Final layer depends on loss type
        if loss_type == 'mse':  # (Darcy example)
            self.final_layer = nn.Conv2d(hidden_dim, out_size, kernel_size=1)  # 1x1 conv for regression

        elif loss_type == 'ce':  # classification (MNIST example)
            flat_size = hidden_dim * conv2_size * conv2_size
            self.final_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, out_size)
            )

    def forward(self, x):
        x = self.conv1(x)

        if self.max_pool:
            x = F.max_pool2d(x, 2)

        x = self.conv2(x)

        if self.max_pool:
            x = F.max_pool2d(x, 2)

        if self.loss_type in ['mse', 'L2', 'lp']:
            x = self.final_layer(x)  # [batch, 1, height, width]
            x = x.squeeze(1)  # Remove channel dimension
        else:  # classification
            x = self.final_layer(x)  # MLPs will handle the flattening
            x = F.log_softmax(x, dim=1)

        return x

class ParabolicCNN(nn.Module):
    """Resolution-invariant CNN using parabolic convolutions."""
    
    def __init__(self, config, in_channels=1, hidden_dim=32, out_size=10, 
                 stride_config=(1, 1), input_size=28, loss_type='ce', max_pool=False):
        super().__init__()

        # Get layer dimensions from config
        self.layer_dims = get_arg_list(config.get('layer_dims', [hidden_dim, hidden_dim]))
        self.max_pool = max_pool
        self.loss_type = loss_type

        # Create ModuleList for convolution layers
        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(nn.Sequential(
            ParabolicConv(in_channels, self.layer_dims[0], normalize=True, 
                         operator_type=config['operator_type'],
                         rank=config['rank'], advection=config['advection']),
            nn.ELU(),
            nn.BatchNorm2d(self.layer_dims[0]),
        ))
        
        # Remaining layers
        for i in range(1, len(self.layer_dims)):
            self.conv_layers.append(nn.Sequential(
                ParabolicConv(self.layer_dims[i-1], self.layer_dims[i], normalize=True, 
                             operator_type=config['operator_type'],
                             rank=config['rank'], advection=config['advection']),
                nn.ELU(),
                nn.BatchNorm2d(self.layer_dims[i]),
            ))
        
        # Calculate output size after convolutions and pooling
        output_size = input_size
        if max_pool:
            output_size = output_size // (2 ** len(self.layer_dims))
        
        # Final layer depends on loss type
        if loss_type in ['mse', 'L2', 'lp']:
            self.final_layer = nn.Conv2d(self.layer_dims[-1], out_size, kernel_size=1)
        else:  # classification
            flat_size = self.layer_dims[-1] * output_size * output_size
            self.final_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, self.layer_dims[-1]),
                nn.ReLU(),
                nn.BatchNorm1d(self.layer_dims[-1]),
                nn.Dropout(0.5),
                nn.Linear(self.layer_dims[-1], out_size)
            )
    
    def forward(self, x):
        # Pass through all convolution layers
        for conv in self.conv_layers:
            x = conv(x)
            if self.max_pool:
                x = F.max_pool2d(x, 2)
        
        if self.loss_type in ['mse', 'L2', 'lp']:
            x = self.final_layer(x)
            x = x.squeeze(1)  # Remove channel dimension
        else:  # classification
            x = self.final_layer(x)
            x = F.log_softmax(x, dim=1)
        
        return x
