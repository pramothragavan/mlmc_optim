"""Spectral convolution layers for Fourier Neural Operators."""

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """2D Fourier layer with spectral convolution.
    
    Performs FFT, linear transform in Fourier space, and inverse FFT.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes1: Number of Fourier modes in first dimension
        modes2: Number of Fourier modes in second dimension
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space.
        
        Args:
            input: (batch, in_channel, x, y)
            weights: (in_channel, out_channel, x, y)
            
        Returns:
            (batch, out_channel, x, y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral convolution.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            
        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):
    """3D Fourier layer with spectral convolution.
    
    Performs FFT, linear transform in Fourier space, and inverse FFT.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes1: Number of Fourier modes in first dimension
        modes2: Number of Fourier modes in second dimension
        modes3: Number of Fourier modes in third dimension
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 modes1: int, modes2: int, modes3: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in Fourier space.
        
        Args:
            input: (batch, in_channel, x, y, z)
            weights: (in_channel, out_channel, x, y, z)
            
        Returns:
            (batch, out_channel, x, y, z)
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral convolution.
        
        Args:
            x: Input tensor (batch, channels, depth, height, width)
            
        Returns:
            Output tensor (batch, out_channels, depth, height, width)
        """
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-3), x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2
        )
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4
        )

        # Inverse FFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
