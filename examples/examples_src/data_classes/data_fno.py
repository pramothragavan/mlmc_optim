"""Multi-resolution dataset for neural operators and simple .mat->.pt converters."""

import os
from pathlib import Path

import h5py
import scipy.io
import torch
import numpy as np
from mlmc_optim.dataset import MLMCDataset


class MatReader:
    """Minimal .mat reader supporting both old and HDF5-style files.

    This is a lightweight copy of the utility used in mesh-mlmc, kept local
    here so new users can convert the official FNO .mat files into the
    finest-resolution .pt format expected by mlmc_optim.
    """

    def __init__(self, file_path, to_torch=True, to_float=True):
        self.to_torch = to_torch
        self.to_float = to_float
        self.file_path = str(file_path)
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
        return x


def convert_darcy_mat_to_pt(train_mat_path, test_mat_path, output_dir, finest_res=241):
    """Convert official Darcy .mat files into finest-resolution .pt files.

    Expects .mat files with fields: 'coeff', 'sol', and optionally
    'Kcoeff', 'Kcoeff_x', 'Kcoeff_y'. The resulting files are saved under
    ``<output_dir>/darcy2d/train_r{finest_res}.pt`` and ``.../test_r{finest_res}.pt``.
    """

    output_dir = Path(output_dir) / "darcy2d"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _process_one(mat_path, split):
        reader = MatReader(mat_path)
        data = {
            "coeff": reader.read_field("coeff"),
            "sol": reader.read_field("sol"),
        }
        for k in ["Kcoeff", "Kcoeff_x", "Kcoeff_y"]:
            try:
                data[k] = reader.read_field(k)
            except Exception:
                continue
        save_path = output_dir / f"{split}_r{finest_res}.pt"
        torch.save(data, save_path)
        print(f"Saved Darcy {split} split to {save_path}")

    _process_one(train_mat_path, "train")
    _process_one(test_mat_path, "test")


def convert_ns_mat_to_pt(ns_mat_path, output_dir, finest_res=64,
                         n_train=3000, n_test=1000):
    """Convert Navier-Stokes .mat file into finest-resolution .pt splits.

    Expects the official ``ns_V1e-3_N5000_T50.mat`` with fields 'a', 'u', 't'.
    Uses the first ``n_train`` samples for training and the next ``n_test``
    samples for testing, and writes ``ns_train_r{finest_res}.pt`` and
    ``ns_test_r{finest_res}.pt`` under ``output_dir``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = MatReader(ns_mat_path)
    a = reader.read_field("a")  # [N, H, W, T_in]
    u = reader.read_field("u")  # [N, H, W, T_total]
    t = reader.read_field("t")  # [T_total]

    N = a.shape[0]
    if N < n_train + n_test:
        raise ValueError(f"Not enough NS samples in mat file: {N} < {n_train + n_test}")

    idx_train = slice(0, n_train)
    idx_test = slice(n_train, n_train + n_test)

    for split, idx in {"train": idx_train, "test": idx_test}.items():
        split_data = {
            "u": u[idx],
            "t": t,
        }
        save_path = output_dir / f"ns_{split}_r{finest_res}.pt"
        torch.save(split_data, save_path)
        print(f"Saved Navier-Stokes {split} split to {save_path}")


class MultiResolutionDataset(MLMCDataset):
    """Dataset for loading pre-processed multi-resolution data.

    Args:
        data_dir: Directory containing the preprocessed data files
        resolution: Resolution to load
        train: If True, load training data, else load test data
        load_in_memory: If True, load all data into memory at initialization.
                       If False, load data from disk as needed (memory efficient).
    """
    def __init__(self, config, data_dir, resolution, train=True, load_in_memory=True, normalize=True, indices=None,
                 add_coords=True, use_grads=False):
        self.config = config
        self.data_dir = data_dir
        self.resolution = resolution
        self.train = train
        self.load_in_memory = load_in_memory
        if train:
            self.data_subset = config.get('train_subset', 1.0)
        else:
            self.data_subset = config.get('test_subset', 1.0)
        self.dataset = config['dataset']
        self.add_coords = add_coords
        self.use_grads = use_grads
        self.pin_memory = config['pin_memory']

        prefix = 'train' if train else 'test'
        if config['dataset'] == 'darcy':
            self.data_path = os.path.join(data_dir, 'darcy2d', f'{prefix}_r{resolution}.pt')
        elif config['dataset'] == 'adr':
            self.data_path = os.path.join(data_dir, 'adr', f'{prefix}_r{resolution}.pt')
            self.has_gradients = False
        
        print(f"Loading data from {self.data_path}...")

        # Load file once
        data = torch.load(self.data_path)
        if config['dataset'] == 'darcy':
            self.has_gradients = all(k in data for k in ["Kcoeff", "Kcoeff_x", "Kcoeff_y"])
        
        # Determine sample indices
        total_samples = len(data['coeff'])
        if indices:
            sample_indices = indices[resolution]
            n_samples = len(sample_indices)
        else:
            n_samples = int(total_samples * self.data_subset)
            sample_indices = list(range(n_samples))
        
        # Extract data based on dataset type
        if self.dataset == 'darcy':
            full_input = data['coeff']
            full_output = data['sol']
            if self.has_gradients:
                full_smooth = data['Kcoeff']
                full_gradx = data['Kcoeff_x']
                full_grady = data['Kcoeff_y']
        elif self.dataset == 'adr':
            full_input = data['coeff']
            full_output = data['sol']
        
        # Calculate normalization stats if needed
        if normalize:
            self.input_mean = torch.mean(full_input)
            self.input_std = torch.std(full_input)
            self.output_mean = torch.mean(full_output)
            self.output_std = torch.std(full_output)
            if self.dataset == 'darcy' and self.has_gradients and self.use_grads:
                self.smooth_mean = torch.mean(full_smooth)
                self.smooth_std = torch.std(full_smooth)
                self.gradx_mean = torch.mean(full_gradx)
                self.gradx_std = torch.std(full_gradx)
                self.grady_mean = torch.mean(full_grady)
                self.grady_std = torch.std(full_grady)
        
        if load_in_memory:
            # Keep data in memory
            self.input_data = full_input[sample_indices]
            self.output_data = full_output[sample_indices]
            if self.has_gradients:
                self.input_smooth = full_smooth[sample_indices]
                self.input_gradx = full_gradx[sample_indices]
                self.input_grady = full_grady[sample_indices]
            
            # Pin memory if requested and CUDA is ready (for MLMC which doesn't use DataLoader)
            if self.pin_memory:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    self.input_data = self.input_data.pin_memory()
                    self.output_data = self.output_data.pin_memory()
                    if self.has_gradients and self.use_grads:
                        self.input_smooth = self.input_smooth.pin_memory()
                        self.input_gradx = self.input_gradx.pin_memory()
                        self.input_grady = self.input_grady.pin_memory()
                    self.config['pin_memory_active'] = True
                else:
                    self.config['pin_memory_active'] = False
        else:
            # Store indices for on-demand loading
            self.data_size = n_samples
            self.sample_indices = sample_indices
        
        # Explicitly free memory
        del data
        if self.has_gradients:
            del full_input, full_output, full_smooth, full_gradx, full_grady
        else:
            del full_input, full_output

        if self.add_coords:
            # Get spatial size from loaded data or from file metadata
            if load_in_memory:
                s = self.input_data.shape[-1]
            else:
                # For on-demand loading, get size from first sample
                temp_data = torch.load(self.data_path)
                s = temp_data['coeff'].shape[-1]
                del temp_data
            
            grids = []
            grids.append(np.linspace(0, 1, s))
            grids.append(np.linspace(0, 1, s))
            grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
            grid = grid.reshape(1, s, s, 2)
            grid = torch.tensor(grid, dtype=torch.float)
            self.grid = grid  # shape [2, s, s]
            self.grid = self.grid.permute(3, 0, 1, 2)
        else:
            self.grid = None

    def __len__(self):
        if self.load_in_memory:
            return len(self.input_data)
        return self.data_size

    def __getitem__(self, idx):
        if self.load_in_memory:
            if self.has_gradients and self.use_grads:
                # print("loading data with gradients")
                x = torch.stack([
                    self.input_data[idx],
                    self.input_smooth[idx],
                    self.input_gradx[idx],
                    self.input_grady[idx]
                ], dim=0)

                # x is (4, s, s)
                if self.grid is not None:
                    grid = self.grid.squeeze(1) if self.grid.ndim == 4 else self.grid   # -> (2, s, s) ideally
                    x = torch.cat([x, grid], dim=0)                                     # (6, s, s)
                return x, self.output_data[idx]
            else:
                if self.grid is None:
                    # print("loading data without gradients")
                    x = torch.stack([self.input_data[idx]], dim=0)
                    return x, self.output_data[idx]
                else:
                    # print("loading data without gradients and with grid")
                    if isinstance(idx, slice):
                        grid_rep = self.grid.repeat(1, self.input_data[idx].shape[0], 1, 1)
                        x = torch.cat([
                            self.input_data[idx].unsqueeze(0),  # (1, 1024, 241, 241)§
                            grid_rep  # (2, 1024, 241, 241)
                        ], dim=0)
                    else:
                        grid_rep = self.grid.squeeze()
                        x = torch.cat([
                            self.input_data[idx].unsqueeze(0),  # (241, 241)
                            grid_rep  # (2, 241, 241)
                        ], dim=0)
                    return x, self.output_data[idx]
        else:
            # Load data from disk on demand
            data = torch.load(self.data_path)
            n_samples = int(len(data['coeff']) * self.data_subset)
            if idx >= n_samples:
                raise IndexError(f"Index {idx} out of range for dataset with {n_samples} samples")

            if self.dataset == 'darcy' and self.has_gradients and self.use_grads:
                x = torch.stack([
                    data['coeff'][idx],
                    data['Kcoeff'][idx],
                    data['Kcoeff_x'][idx],
                    data['Kcoeff_y'][idx]
                ], dim=0)
                # x is (4, s, s)
                if self.grid is not None:
                    grid = self.grid.squeeze(1) if self.grid.ndim == 4 else self.grid   # (2, s, s)
                    x = torch.cat([x, grid], dim=0)                                     # (6, s, s)
            else:
                if self.grid is None:
                    x = torch.stack([data['coeff'][idx]], dim=0)
                else:
                    x = torch.cat([
                        data['coeff'][idx].unsqueeze(0),
                        self.grid.squeeze()
                    ], dim=0)
            return x, data['sol'][idx]

    def get_items(self, indices):
        # Get multiple items by their indices.
        # Convert indices to tensor if not already
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)

        # Check if we have GPU cache and if requested indices are in cache
        if hasattr(self, "_gpu_input_cache") and hasattr(self, "_gpu_indices_sorted"):
            if not isinstance(indices, torch.Tensor):
                indices = torch.tensor(indices, dtype=torch.long)
            else:
                indices = indices.to(dtype=torch.long)

            idx = indices.to(self._gpu_device)

            pos = torch.searchsorted(self._gpu_indices_sorted, idx)
            n = self._gpu_indices_sorted.numel()
            in_bounds = pos < n
            in_cache = in_bounds & (self._gpu_indices_sorted[pos.clamp_max(n - 1)] == idx)

            if in_cache.all():
                if self.has_gradients and self.use_grads:
                    x = torch.stack([
                        self._gpu_input_cache[pos],
                        self._gpu_smooth_cache[pos],
                        self._gpu_gradx_cache[pos],
                        self._gpu_grady_cache[pos],
                    ], dim=1).contiguous()

                    if self.grid is not None:
                        grid = self.grid.to(self._gpu_device)
                        grid = grid.permute(1,0,2,3)            # (1,2,s,s)
                        grid_rep = grid.expand(len(indices), -1, -1, -1)  # (B,2,s,s)
                        x = torch.cat([x, grid_rep], dim=1).contiguous()
                    return x, self._gpu_output_cache[pos].contiguous()
                else:
                    x = self._gpu_input_cache[pos].unsqueeze(1).contiguous()
                    if self.grid is not None:
                        grid = self.grid.to(self._gpu_device).permute(1, 0, 2, 3)  # (1,2,s,s)
                        grid_rep = grid.expand(len(indices), -1, -1, -1)
                        x = torch.cat([x, grid_rep], dim=1).contiguous()

                    return x, self._gpu_output_cache[pos].contiguous()

        # Fall back to CPU data
        if self.load_in_memory:
            if self.has_gradients and self.use_grads:
                # Stack input data with gradients
                x = torch.stack([
                    self.input_data[indices],
                    self.input_smooth[indices],
                    self.input_gradx[indices],
                    self.input_grady[indices]
                ], dim=1)

                # x is (B, 4, s, s)
                if self.grid is not None:
                    # convert to (1, 2, s, s) then expand to (B, 2, s, s)
                    grid = self.grid
                    if grid.ndim == 4:                         # (2,1,s,s)
                        grid = grid.permute(1,0,2,3)           # (1,2,s,s)
                    else:                                      # (2,s,s)
                        grid = grid.unsqueeze(0)               # (1,2,s,s)
                    grid_rep = grid.expand(len(indices), -1, -1, -1)  # (B,2,s,s) view
                    x = torch.cat([x, grid_rep], dim=1)
                return x, self.output_data[indices]
            else:
                if self.grid is None:
                    x = self.input_data[indices].unsqueeze(1)
                    return x, self.output_data[indices]
                else:
                    # Reshape grid to match batch dimension first
                    grid_rep = self.grid.permute(1, 0, 2, 3).repeat(len(indices), 1, 1, 1)
                    x = torch.cat([
                        self.input_data[indices].unsqueeze(1),
                        grid_rep
                    ], dim=1)
                    return x, self.output_data[indices]
        else:
            # Load data from disk
            data = torch.load(self.data_path)
            if self.dataset == 'darcy':
                if self.has_gradients and self.use_grads:
                    x = torch.stack([
                        data['coeff'][indices],
                        data['Kcoeff'][indices],
                        data['Kcoeff_x'][indices],
                        data['Kcoeff_y'][indices]
                    ], dim=1)  # (B, 4, s, s)
                    if self.grid is not None:
                        grid = self.grid
                        grid = grid.permute(1, 0, 2, 3)  # (1,2,s,s) from (2,1,s,s)
                        grid_rep = grid.expand(len(indices), -1, -1, -1)  # (B,2,s,s)
                        x = torch.cat([x, grid_rep], dim=1)  # (B, 6, s, s)
                else:
                    x = data['coeff'][indices].unsqueeze(1)  # (B,1,s,s)
                    if self.grid is not None:
                        grid = self.grid.permute(1, 0, 2, 3)          # (1,2,s,s)
                        grid_rep = grid.expand(len(indices), -1, -1, -1)
                        x = torch.cat([x, grid_rep], dim=1)           # (B,3,s,s)
                return x, data['sol'][indices]
            elif self.dataset == 'adr':
                return data['coeff'][indices], data['sol'][indices]

    def to_device(self, device):
        """Move all tensors to specified device."""
        # Call parent implementation for standard tensors
        super().to_device(device)
        
        # Move gradient tensors if they exist (Darcy-specific)
        if hasattr(self, 'input_smooth'):
            self.input_smooth = self.input_smooth.to(device)
            self.input_gradx = self.input_gradx.to(device)
            self.input_grady = self.input_grady.to(device)

        # Move grid if present (FNO-specific)
        if hasattr(self, 'grid') and self.grid is not None:
            self.grid = self.grid.to(device)

        # Move gradient statistics if present (Darcy-specific)
        if hasattr(self, 'smooth_mean'):
            self.smooth_mean = self.smooth_mean.to(device)
            self.smooth_std = self.smooth_std.to(device)
            self.gradx_mean = self.gradx_mean.to(device)
            self.gradx_std = self.gradx_std.to(device)
            self.grady_mean = self.grady_mean.to(device)
            self.grady_std = self.grady_std.to(device)
        
        return self

    def load_batch_indices_to_gpu(self, indices, device):
        """Load only the data for specified indices to GPU.
        
        Extends parent to also cache gradient tensors for Darcy dataset.
        """
        # Ensure tensor on CPU for consistent indexing
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            indices = indices.to(dtype=torch.long)

        sorted_idx, _ = torch.sort(indices.cpu())

        # Call parent implementation for standard tensors
        super().load_batch_indices_to_gpu(sorted_idx, device)

        # Cache gradient tensors if present (Darcy-specific)
        if hasattr(self, 'input_smooth'):
            self._gpu_smooth_cache = self.input_smooth[sorted_idx].contiguous().to(device)
            self._gpu_gradx_cache  = self.input_gradx[sorted_idx].contiguous().to(device)
            self._gpu_grady_cache  = self.input_grady[sorted_idx].contiguous().to(device)

    def unload_from_gpu(self, indices=None):
        """Clear GPU cache to free memory.
        
        Extends parent to also clear gradient caches.
        """
        # Call parent implementation
        super().unload_from_gpu(indices)
        
        # Clear gradient caches if present (Darcy-specific)
        if hasattr(self, '_gpu_smooth_cache'):
            del self._gpu_smooth_cache
            del self._gpu_gradx_cache
            del self._gpu_grady_cache


class MultiResolutionDataset3D(MLMCDataset):
    """Dataset for loading pre-processed multi-resolution Fourier 3D data.

    Args:
        config: Configuration dictionary
        data_dir: Directory containing the preprocessed data files
        resolution: Resolution to load
        train: If True, load training data, else load test data
        load_in_memory: If True, load all data into memory at initialization
        normalize: If True, normalize the input and output data
        indices: Optional list of indices to use (for subset selection)
        add_coords: If True, add coordinate channels to input
    """

    def __init__(self, config, data_dir, resolution, train=True, load_in_memory=True,
                 normalize=True, indices=None, add_coords=True):
        self.config = config
        self.data_dir = data_dir
        self.resolution = resolution
        self.train = train
        self.load_in_memory = load_in_memory
        if train:
            self.data_subset = config.get('train_subset', 1.0)
        else:
            self.data_subset = config.get('test_subset', 1.0)
        self.dataset = config['dataset']
        self.add_coords = add_coords
        # self.add_coords = True if config['model'] == 'fno' else False
        self.pin_memory = config['pin_memory']

        prefix = 'train' if train else 'test'
        if config['dataset'] == 'navier_stokes':
            self.data_path = os.path.join(data_dir, 'ns2d_time', f'ns_{prefix}_r{resolution}.pt')
            self.has_gradients = False
        else:
            raise ValueError(f"Unknown dataset: {config['dataset']}")
        
        print(f"Loading data from {self.data_path}...")

        # Load data
        T_in = config['T_in']
        T = config['T']
        
        # Load file once
        data = torch.load(self.data_path, weights_only=True)
        full_u = data['u']
        
        # Extract metadata
        self.n_samples = full_u.shape[0]
        self.height = full_u.shape[1]
        self.width = full_u.shape[2]
        self.time_steps = T if len(full_u.shape) == 4 else 1
        
        # Determine sample indices
        if indices:
            sample_indices = indices[resolution]
            self.n_samples = len(sample_indices)
        else:
            n_samples = int(self.n_samples * self.data_subset)
            sample_indices = list(range(n_samples))
            self.n_samples = n_samples
        
        # Split into input/output
        full_a = full_u[:, :, :, :T_in]
        full_u = full_u[:, :, :, T_in: T + T_in]
        
        # Calculate normalization stats if needed
        if normalize:
            self.output_mean = torch.mean(full_u)
            self.output_std = torch.std(full_u)
            self.input_mean = torch.mean(full_a)
            self.input_std = torch.std(full_a)
        
        if load_in_memory:
            # Keep data in memory
            self.a = full_a[sample_indices]
            self.u = full_u[sample_indices]
            self.t = data['t']
            
            # Pin memory if requested and CUDA is ready (for MLMC which doesn't use DataLoader)
            if self.pin_memory:
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    self.a = self.a.pin_memory()
                    self.u = self.u.pin_memory()
                    self.t = self.t.pin_memory()
                    self.config['pin_memory_active'] = True
                else:
                    self.config['pin_memory_active'] = False
        else:
            # Store indices for on-demand loading
            self.sample_indices = sample_indices
            self.a = None
            self.u = None
        
        # Explicitly free memory
        del data, full_a, full_u

        # Create coordinate grid if needed
        if self.add_coords:
            if self.load_in_memory:
                self.a = self.a.reshape(self.a.shape[0], self.width, self.height, 1, T_in).repeat(
                    [1, 1, 1, self.time_steps, 1])  # [N, W, H, T, T_in]
            gridx = torch.tensor(np.linspace(0, 1, self.height), dtype=torch.float)
            self.gridx = gridx.reshape(1, self.height, 1, 1, 1).repeat([1, 1, self.width, self.time_steps, 1])
            gridy = torch.tensor(np.linspace(0, 1, self.width), dtype=torch.float)
            self.gridy = gridy.reshape(1, 1, self.width, 1, 1).repeat([1, self.height, 1, self.time_steps, 1])
            gridt = torch.tensor(np.linspace(0, 1, self.time_steps + 1)[1:], dtype=torch.float)
            self.gridt = gridt.reshape(1, 1, 1, self.time_steps, 1).repeat([1, self.width, self.height, 1, 1])
            
            # Pin coordinate grids if requested and CUDA is ready
            if self.pin_memory and torch.cuda.is_available() and torch.cuda.is_initialized():
                self.gridx = self.gridx.pin_memory()
                self.gridy = self.gridy.pin_memory()
                self.gridt = self.gridt.pin_memory()
        else:
            self.gridx = None
            self.gridy = None
            self.gridt = None
        
        # Create aliases for base class compatibility (after reshape)
        if self.load_in_memory:
            self.input_data = self.a
            self.output_data = self.u
        else:
            self.input_data = None
            self.output_data = None

    def __len__(self):
        if self.load_in_memory:
            return len(self.a)
        return self.n_samples

    def __getitem__(self, idx):
        if self.load_in_memory:
            # Use GPU cache if available
            if hasattr(self, '_gpu_input_cache'):
                a_data = self._gpu_input_cache
                u_data = self._gpu_output_cache
                gridx = self._gpu_gridx_cache if hasattr(self, '_gpu_gridx_cache') else self.gridx
                gridy = self._gpu_gridy_cache if hasattr(self, '_gpu_gridy_cache') else self.gridy
                gridt = self._gpu_gridt_cache if hasattr(self, '_gpu_gridt_cache') else self.gridt
                # Map idx to cache index
                if not isinstance(idx, slice):
                    cache_idx = (self._gpu_indices == idx).nonzero(as_tuple=True)[0][0]
                else:
                    cache_idx = idx
            else:
                a_data = self.a
                u_data = self.u
                gridx = self.gridx
                gridy = self.gridy
                gridt = self.gridt
                cache_idx = idx
            
            if isinstance(idx, slice):
                gridx_rep = gridx.repeat(a_data[cache_idx].shape[0], 1, 1, 1, 1)
                gridy_rep = gridy.repeat(a_data[cache_idx].shape[0], 1, 1, 1, 1)
                gridt_rep = gridt.repeat(a_data[cache_idx].shape[0], 1, 1, 1, 1)

                x = torch.cat([
                        a_data[cache_idx],
                        gridx_rep,
                        gridy_rep,
                        gridt_rep
                    ], dim=-1)
            else:
                gridx_rep = gridx
                gridy_rep = gridy
                gridt_rep = gridt

                x = torch.cat([
                        a_data[cache_idx].unsqueeze(0),
                        gridx_rep,
                        gridy_rep,
                        gridt_rep
                    ], dim=-1)
            return x.squeeze(), u_data[cache_idx]
        else:
            # Load on demand from disk
            data = torch.load(self.data_path)
            
            T_in = self.config['T_in']
            T = self.config['T']
            
            # Handle both single index and slice
            if isinstance(idx, slice):
                # Load multiple samples
                actual_indices = self.sample_indices[idx]
                a = data['u'][actual_indices, :, :, :T_in]  # Shape: [N, H, W, T_in]
                u = data['u'][actual_indices, :, :, T_in: T + T_in]  # Shape: [N, H, W, T]
                
                if self.add_coords:
                    # Reshape to [N, W, H, T_out, T_in]
                    a = a.permute(0, 2, 1, 3)  # [N, W, H, T_in]
                    a = a.unsqueeze(3).repeat(1, 1, 1, self.time_steps, 1)  # [N, W, H, T_out, T_in]
                    # Repeat grids for batch
                    gridx_rep = self.gridx.repeat(len(actual_indices), 1, 1, 1, 1)
                    gridy_rep = self.gridy.repeat(len(actual_indices), 1, 1, 1, 1)
                    gridt_rep = self.gridt.repeat(len(actual_indices), 1, 1, 1, 1)
                    x = torch.cat([a, gridx_rep, gridy_rep, gridt_rep], dim=-1)
                else:
                    x = a
                return x, u
            else:
                # Load single sample
                actual_idx = self.sample_indices[idx]
                a = data['u'][actual_idx, :, :, :T_in]  # Shape: [H, W, T_in]
                u = data['u'][actual_idx, :, :, T_in: T + T_in]  # Shape: [H, W, T]
                
                if self.add_coords:
                    # Reshape to [W, H, T_out, T_in]
                    a = a.permute(1, 0, 2)  # [W, H, T_in]
                    a = a.unsqueeze(2).repeat(1, 1, self.time_steps, 1)  # [W, H, T_out, T_in]
                    x = torch.cat([
                        a.unsqueeze(0),  # [1, W, H, T_out, T_in]
                        self.gridx,      # [1, H, W, T_out, 1]
                        self.gridy,      # [1, H, W, T_out, 1]
                        self.gridt       # [1, W, H, T_out, 1]
                    ], dim=-1)
                else:
                    x = a.unsqueeze(0)
                
                return x.squeeze(), u

    def get_items(self, indices):
        # Get multiple items by their indices.
        # Convert indices to tensor if not already
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)

        if self.load_in_memory:
            # Use GPU cache if available
            if hasattr(self, '_gpu_input_cache'):
                # Map global indices to cache indices
                cache_indices = torch.tensor([torch.where(self._gpu_indices == i)[0][0] for i in indices])
                a_data = self._gpu_input_cache[cache_indices]
                u_data = self._gpu_output_cache[cache_indices]
                gridx = self._gpu_gridx_cache if hasattr(self, '_gpu_gridx_cache') else self.gridx
                gridy = self._gpu_gridy_cache if hasattr(self, '_gpu_gridy_cache') else self.gridy
                gridt = self._gpu_gridt_cache if hasattr(self, '_gpu_gridt_cache') else self.gridt
            else:
                a_data = self.a[indices]
                u_data = self.u[indices]
                gridx = self.gridx
                gridy = self.gridy
                gridt = self.gridt
            
            # Reshape grid to match batch dimension first
            gridx_rep = gridx.repeat(len(indices), 1, 1, 1, 1)
            gridy_rep = gridy.repeat(len(indices), 1, 1, 1, 1)
            gridt_rep = gridt.repeat(len(indices), 1, 1, 1, 1)
            x = torch.cat([
                a_data,
                gridx_rep,
                gridy_rep,
                gridt_rep
            ], dim=-1)
            return x, u_data

    def to_device(self, device):
        """Move all tensors to specified device."""
        self.a = self.a.to(device)
        self.u = self.u.to(device)
        self.t = self.t.to(device)
        if hasattr(self, 'input_mean'):
            self.input_mean = self.input_mean.to(device)
            self.input_std = self.input_std.to(device)
            self.output_mean = self.output_mean.to(device)
            self.output_std = self.output_std.to(device)
        if hasattr(self, 'gridx'):
            self.gridx = self.gridx.to(device)
            self.gridy = self.gridy.to(device)
            self.gridt = self.gridt.to(device)
        return self
    
    def load_batch_indices_to_gpu(self, indices, device):
        """Load only the data for specified indices to GPU.
        
        Caches input data (a), output data (u), and coordinate grids.
        """
        if not self.load_in_memory:
            return
        
        # Convert to tensor if needed
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        
        # Store cache metadata
        self._gpu_indices = indices
        self._gpu_device = device
        
        # Clone on CPU first for contiguity, then move to GPU once
        self._gpu_input_cache = self.a[indices].clone().to(device)
        self._gpu_output_cache = self.u[indices].clone().to(device)
        
        # Cache grids if present (they're shared across all samples)
        if hasattr(self, 'gridx'):
            self._gpu_gridx_cache = self.gridx.to(device)
            self._gpu_gridy_cache = self.gridy.to(device)
            self._gpu_gridt_cache = self.gridt.to(device)
    
    def unload_from_gpu(self, indices=None):
        """Clear GPU cache to free memory."""
        if not self.load_in_memory:
            return
        
        # Clear main data caches
        if hasattr(self, '_gpu_input_cache'):
            del self._gpu_input_cache
            del self._gpu_output_cache
            del self._gpu_indices
            del self._gpu_device
        
        # Clear grid caches if present
        if hasattr(self, '_gpu_gridx_cache'):
            del self._gpu_gridx_cache
            del self._gpu_gridy_cache
            del self._gpu_gridt_cache
