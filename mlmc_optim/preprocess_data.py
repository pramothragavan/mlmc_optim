"""Preprocess data to create multi-resolution datasets.

This script takes fine-resolution data and creates coarser versions
by downsampling for MLMC training.

Usage:
    python scripts/preprocess_data.py --input data/fine.pt --output data/ --resolutions 30 60 120 241 --method avgpool --format ns3d
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm


def _downsample_avgpool(data, target_size):
    """Downsample using average pooling (preserves mass/energy)."""
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    
    h_factor = data.shape[2] / target_size
    w_factor = data.shape[3] / target_size
    
    if not (h_factor.is_integer() and w_factor.is_integer()):
        raise ValueError(f"Average pooling requires integer ratios. Got {data.shape[2]}/{target_size} and {data.shape[3]}/{target_size}")
    
    pool = nn.AvgPool2d(kernel_size=(int(h_factor), int(w_factor)))
    downsampled = pool(data.float())
    
    if squeeze:
        downsampled = downsampled.squeeze(1)
    return downsampled


def _downsample_bilinear(data, target_size):
    """Downsample using bilinear interpolation (smooth, works for any size)."""
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    
    downsampled = F.interpolate(
        data,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=True
    )
    
    if squeeze:
        downsampled = downsampled.squeeze(1)
    return downsampled


def _downsample_spectral(data, target_size):
    """Downsample using FFT truncation (preserves smooth features)."""
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    
    N, C, H, W = data.shape
    downsampled = torch.zeros(N, C, target_size, target_size, dtype=data.dtype)
    
    for n in range(N):
        for c in range(C):
            # FFT to frequency domain
            fft_data = np.fft.fftshift(np.fft.fft2(data[n, c].numpy()))
            
            # Crop high frequencies
            start = (H - target_size) // 2
            end = start + target_size
            fft_cropped = fft_data[start:end, start:end]
            
            # Inverse FFT
            spatial = np.fft.ifft2(np.fft.ifftshift(fft_cropped)).real
            
            # Scale to preserve energy (multiply, not divide)
            scale_factor = (target_size / H) ** 2
            downsampled[n, c] = torch.from_numpy(spatial * scale_factor)
    
    if squeeze:
        downsampled = downsampled.squeeze(1)
    return downsampled


def _downsample_subsample(data, target_size):
    """Downsample using strided subsampling (fastest, no averaging)."""
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    
    H, W = data.shape[2], data.shape[3]
    stride_h = H // target_size
    stride_w = W // target_size
    
    downsampled = data[:, :, ::stride_h, ::stride_w][:, :, :target_size, :target_size]
    
    if squeeze:
        downsampled = downsampled.squeeze(1)
    return downsampled


def downsample_2d(data, target_size, method='avgpool'):
    """Downsample 2D data to target resolution.
    
    Args:
        data: Tensor of shape (N, H, W) or (N, C, H, W)
        target_size: Target spatial resolution
        method: 'avgpool', 'bilinear', 'spectral', or 'subsample'
        
    Returns:
        Downsampled tensor
    """
    if method == 'avgpool':
        return _downsample_avgpool(data, target_size)
    elif method == 'bilinear':
        return _downsample_bilinear(data, target_size)
    elif method == 'spectral':
        return _downsample_spectral(data, target_size)
    elif method == 'subsample':
        return _downsample_subsample(data, target_size)
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'avgpool', 'bilinear', 'spectral', or 'subsample'")


def upscale_2d(data, target_size, method='bilinear'):
    """Upscale 2D data to target resolution."""
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    if method == 'bilinear':
        up = F.interpolate(data.float(), size=(target_size, target_size), mode='bilinear', align_corners=True)
    else:
        raise ValueError(f"Unknown upsampling method: {method}. Use 'bilinear'.")

    if squeeze:
        up = up.squeeze(1)
    return up


def downsample_3d(data, target_size, method='avgpool'):
    """Downsample 3D data to target resolution.
    
    Args:
        data: Tensor of shape (N, D, H, W) or (N, C, D, H, W)
        target_size: Target spatial resolution
        method: 'avgpool' or 'trilinear' (3D doesn't support spectral/subsample yet)
        
    Returns:
        Downsampled tensor
    """
    if len(data.shape) == 4:
        data = data.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    
    if method == 'avgpool':
        d_factor = data.shape[2] / target_size
        h_factor = data.shape[3] / target_size
        w_factor = data.shape[4] / target_size
        
        if not (d_factor.is_integer() and h_factor.is_integer() and w_factor.is_integer()):
            raise ValueError(f"Average pooling requires integer ratios for 3D")
        
        pool = nn.AvgPool3d(kernel_size=(int(d_factor), int(h_factor), int(w_factor)))
        downsampled = pool(data.float())
    elif method == 'trilinear':
        downsampled = F.interpolate(
            data,
            size=(target_size, target_size, target_size),
            mode='trilinear',
            align_corners=True
        )
    else:
        raise ValueError(f"3D only supports 'avgpool' or 'trilinear', got {method}")
    
    if squeeze:
        downsampled = downsampled.squeeze(1)
    
    return downsampled


def process_and_save_multiresolution(
    input_path,
    output_dir,
    resolutions,
    data_keys,
    method='avgpool',
    dimension='2d',
    split='train',
    dataset_name='data',
    max_samples=None,
):
    """Process fine data into multiple resolutions.
    
    Args:
        input_path: Path to fine resolution data (.pt file)
        output_dir: Directory to save processed data
        resolutions: List of target resolutions (coarse to fine)
        data_keys: List of keys to load, e.g. ['u', 't'] or ['x', 'y']
        method: 'avgpool', 'bilinear', 'spectral', or 'subsample'
        dimension: '2d' or '3d'
        split: 'train' or 'test'
    """
    print(f"Loading data from {input_path}...")
    data_dict = torch.load(input_path)
    
    # Extract data based on explicit keys
    loaded_data = {}
    for key in data_keys:
        loaded_data[key] = data_dict[key]  # Will fail loudly if key missing
        print(f"Loaded '{key}' with shape: {loaded_data[key].shape}")
    
    # Determine format based on keys
    if 'u' in loaded_data and 't' in loaded_data:
        # Spatiotemporal format [N, H, W, T]
        data = loaded_data['u']
        time = loaded_data['t']
        targets = None
        has_time_dim = True
    elif 'x' in loaded_data and 'y' in loaded_data:
        # Input-output format
        data = loaded_data['x']
        targets = loaded_data['y']
        time = None
        has_time_dim = False
    else:
        raise ValueError(f"Unknown key combination: {data_keys}. Expected ['u','t'] or ['x','y']")

    # Optionally restrict to the first max_samples along the sample dimension
    if max_samples is not None:
        if has_time_dim:
            data = data[:max_samples]
            # time is shared across samples; leave unchanged
        else:
            data = data[:max_samples]
            targets = targets[:max_samples]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each resolution
    for res in tqdm(resolutions, desc="Processing resolutions"):
        print(f"\nProcessing resolution {res} using {method}...")
        
        # Handle spatiotemporal format with time dimension [N, H, W, T]
        if has_time_dim and len(data.shape) == 4:
            N, H, W, T = data.shape
            # Reshape to [N*T, H, W] for spatial downsampling
            reshaped = data.permute(0, 3, 1, 2).reshape(-1, H, W)
            
            # Downsample spatially
            if dimension == '2d':
                downsampled = downsample_2d(reshaped, res, method=method)
            else:
                raise ValueError("Spatiotemporal format only supports 2D spatial downsampling")
            
            # Reshape back to [N, H', W', T]
            data_res = downsampled.reshape(N, T, res, res).permute(0, 2, 3, 1)
            print(f"  Downsampled shape: {data_res.shape}")
            
            # Save in spatiotemporal format
            output_path = output_dir / f"{dataset_name}_{split}_r{res}.pt"
            torch.save({'u': data_res, 't': time}, output_path)
            
        # Handle input_output format
        elif not has_time_dim:
            if dimension == '2d':
                data_res = downsample_2d(data, res, method=method)
                targets_res = downsample_2d(targets, res, method=method)
            elif dimension == '3d':
                data_res = downsample_3d(data, res, method=method)
                targets_res = downsample_3d(targets, res, method=method)
            else:
                raise ValueError(f"Unknown dimension: {dimension}")
            
            print(f"  Downsampled input shape: {data_res.shape}")
            print(f"  Downsampled target shape: {targets_res.shape}")
            
            # Save in input_output format
            output_path = output_dir / f"{split}_r{res}.pt"
            torch.save({'x': data_res, 'y': targets_res}, output_path)
        
        print(f"  Saved to {output_path}")
    
    print(f"\n✓ Processed {len(resolutions)} resolutions using {method}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess multi-resolution data')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing fine resolution data')
    parser.add_argument('--c2f_resolutions', type=int, nargs='+', required=True,
                        help='Target resolutions coarse-to-fine (e.g., 25 50 100)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name prefix (e.g., ns)')
    parser.add_argument('--keys', type=str, nargs='+', default=['u', 't'],
                        help='Data keys to load (e.g., u t or x y)')
    parser.add_argument('--method', type=str, default='avgpool', 
                        choices=['avgpool', 'bilinear', 'spectral', 'subsample'],
                        help='Downsampling method')
    parser.add_argument('--dimension', type=str, default='2d', choices=['2d', '3d'],
                        help='Data dimension')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Optionally limit to the first N samples before downsampling')
    parser.add_argument('--finest_res', type=int, required=True,
                        help='Finest resolution (e.g., 200)')
    
    args = parser.parse_args()
    
    # Process all splits
    splits = ['train', 'val', 'test']
    input_dir = Path(args.input_dir)
    
    for split in splits:
        input_file = input_dir / f"{args.dataset_name}_{split}_r{args.finest_res}.pt"
        
        if not input_file.exists():
            print(f"Skipping {split}: {input_file} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        
        process_and_save_multiresolution(
            str(input_file),
            str(input_dir),
            args.c2f_resolutions,
            data_keys=args.keys,
            method=args.method,
            dimension=args.dimension,
            split=split,
            dataset_name=args.dataset_name,
            max_samples=args.max_samples,
        )


if __name__ == '__main__':
    main()
