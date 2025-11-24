"""Parabolic convolution for resolution-invariant PDEs."""

import torch
import torch.nn as nn


class ParabolicConv(nn.Module):
    """Parabolic convolution with configurable elliptic operator."""
    
    def __init__(self, in_channels, out_channels, operator_type='diagonal', rank=4, normalize=False, advection=False):
        super(ParabolicConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.operator_type = operator_type
        self.advection = advection

        # Reaction term: 1x1 convolution
        self.R = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Initialize elliptic operator parameters based on type
        if operator_type == 'diagonal':
            # Diagonal terms with rotation
            self.cx = nn.Parameter(0.5 * torch.ones(1, out_channels, 1, 1))
            self.cy = nn.Parameter(0.5 * torch.ones(1, out_channels, 1, 1))
            self.cxy = nn.Parameter(0.1 * torch.randn(1, out_channels, 1, 1))
        elif operator_type == 'factored':
            # W^T W factorization for guaranteed PSD
            self.W = nn.Parameter(0.1 * torch.randn(1, out_channels, rank, rank))
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")

        # Learnable parameters for gradients
        self.gx = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.gy = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.g0 = nn.Parameter(0.01 * torch.ones(1, out_channels, 1, 1))

        # Diffusion time
        self.t = nn.Parameter(torch.log(0.5 * torch.rand(1, out_channels, 1, 1)))

        # Optional normalization with learnable parameters
        if normalize:
            self.gamma = nn.Parameter(0.1 * torch.rand(1, out_channels, 1, 1))
            self.beta = nn.Parameter(1e-5 * torch.rand(1, out_channels, 1, 1))

        # If advection
        if self.advection:
            self.a_x = nn.Parameter(torch.ones(1, out_channels, 1, 1))
            self.a_y = nn.Parameter(torch.ones(1, out_channels, 1, 1))

            self.conv_x = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3),
                                    padding=(0, 1), groups=out_channels, bias=False)
            self.conv_x.weight.data.copy_(torch.tensor([[[[-1., 0., 1.]]]]).repeat(out_channels, 1, 1, 1))

            self.conv_y = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1),
                                    padding=(1, 0), groups=out_channels, bias=False)
            self.conv_y.weight.data.copy_(torch.tensor([[[[1.], [0.], [-1.]]]]).repeat(out_channels, 1, 1, 1))

    def clear_cache(self):
        """Safely clear GPU memory cache based on available backends."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_elliptic_operator(self, freqs_n, freqs_m):
        """Construct elliptic operator based on type."""
        if self.operator_type == 'diagonal':
            # Define elliptic operator with rotation
            A = torch.zeros(1, self.out_channels, 2, 2, device=self.cx.device)
            A[:, :, 0, 0] = torch.relu(self.cx.squeeze()) + 1e-3
            A[:, :, 1, 1] = torch.relu(self.cy.squeeze()) + 1e-3

            R = torch.zeros_like(A)
            R[:, :, 0, 0] = torch.cos(self.cxy.squeeze())
            R[:, :, 0, 1] = torch.sin(self.cxy.squeeze())
            R[:, :, 1, 0] = -torch.sin(self.cxy.squeeze())
            R[:, :, 1, 1] = torch.cos(self.cxy.squeeze())

            C = R @ A @ R.transpose(2, 3)
            cx = C[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
            cy = C[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)
            cxy = C[:, :, 0, 1].unsqueeze(-1).unsqueeze(-1)

        elif self.operator_type == 'factored':
            # Construct W^T W to ensure PSD
            W = self.W.squeeze(0)
            A = torch.matmul(W.transpose(1, 2), W)
            
            cx = A[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
            cy = A[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
            cxy = A[:, 0, 1].unsqueeze(-1).unsqueeze(-1)

        return cx * (freqs_n**2) + cy * (freqs_m**2) + 2 * cxy * freqs_n * freqs_m

    def forward(self, image):
        if image.shape[1] != self.in_channels:
            raise ValueError(f"Expected input to have {self.in_channels} channels, but got {image.shape[1]} channels")

        # Reaction term
        image = self.R(image)
        
        self.clear_cache()
            
        # Process in chunks if image is too large
        if image.shape[-1] > 10000:
            chunks = []
            chunk_size = min(4, image.shape[0])
            for i in range(0, image.shape[0], chunk_size):
                chunk = image[i:i+chunk_size]
                chunk_fft = torch.fft.rfft2(chunk)
                
                scale = 1
                device = chunk.device
                m, n = chunk.shape[-2], chunk.shape[-1]
                freqs_n = torch.fft.fftfreq(n, d=scale / n).to(device).reshape(1, 1, -1, 1)
                freqs_m = torch.fft.rfftfreq(m, d=scale / m).to(device).reshape(1, 1, 1, -1)

                freq_magnitude = self.get_elliptic_operator(freqs_n, freqs_m)

                t = torch.exp(self.t)
                Gd = torch.exp(-t * freq_magnitude.abs())
                Gu = 1j * self.gx * freqs_m + 1j * self.gy * freqs_n + self.g0
                G = Gd * Gu

                chunk_result = torch.fft.irfft2(chunk_fft * G, s=(m, n))
                chunks.append(chunk_result)

                self.clear_cache()

            image = torch.cat(chunks, dim=0)
        else:
            image_fft = torch.fft.rfft2(image)
            scale = 1
            device = image.device
            m, n = image.shape[-2], image.shape[-1]
            freqs_n = torch.fft.fftfreq(n, d=scale / n).to(device).reshape(1, 1, -1, 1)
            freqs_m = torch.fft.rfftfreq(m, d=scale / m).to(device).reshape(1, 1, 1, -1)
            freq_magnitude = self.get_elliptic_operator(freqs_n, freqs_m)
            t = torch.exp(self.t)
            Gd = torch.exp(-t * freq_magnitude.abs())
            Gu = 1j * self.gx * freqs_m + 1j * self.gy * freqs_n + self.g0
            G = Gd * Gu
            image = torch.fft.irfft2(image_fft * G, s=(m, n))

        if self.advection:
            x_adv_x = self.a_x * self.conv_x(image)
            x_adv_y = self.a_y * self.conv_y(image)
            image = image + x_adv_x + x_adv_y

        if self.normalize:
            mean = image.mean(dim=(2, 3), keepdim=True)
            std = image.std(dim=(2, 3), keepdim=True) + 1e-5
            image = (image - mean) / std
            image = self.gamma * image + self.beta

        return image
