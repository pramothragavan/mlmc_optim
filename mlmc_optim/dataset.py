"""Abstract base classes for MLMC-compatible datasets."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Optional
import torch
from torch.utils.data import Dataset

# Optional PyG import for graph datasets
try:
    from torch_geometric.data import InMemoryDataset as PyGInMemoryDataset
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    PyGInMemoryDataset = object  # Fallback for when PyG is not installed
    HAS_TORCH_GEOMETRIC = False


class MLMCDataset(Dataset, ABC):
    """Abstract base class for MLMC-compatible datasets.
    
    Provides default implementations for standard tensor-based datasets
    (e.g., Darcy, Navier-Stokes). Override methods for special cases
    (e.g., variable-length point clouds, graph data).
    
    Required attributes (set in __init__):
        - input_data: Input tensor [N, ...]
        - output_data: Output tensor [N, ...]
        - load_in_memory: bool
    
    Optional attributes for GPU caching:
        - _gpu_input_cache: Cached inputs on GPU
        - _gpu_output_cache: Cached outputs on GPU
        - _gpu_indices: Indices currently cached
        - _gpu_device: Device where cache is stored
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Default implementation for tensor-based datasets.
        """
        if hasattr(self, 'load_in_memory') and self.load_in_memory:
            return len(self.input_data)
        elif hasattr(self, 'data_size'):
            return self.data_size
        raise NotImplementedError("Must set input_data or data_size in __init__")
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            (input, target) tuple
            
        Note: Subclasses must implement this for dataset-specific logic
        (e.g., adding coords, handling gradients, etc.)
        """
        pass
    
    def get_items(self, indices: Union[List[int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get multiple samples by indices (batch getter).
        
        Default implementation with GPU cache support for standard tensor datasets.
        Override for special data structures (e.g., graphs, variable-length).
        
        Args:
            indices: List or tensor of sample indices
            
        Returns:
            (inputs, targets) tuple of batched tensors
        """
        # Convert to tensor if needed
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            indices = indices.to(dtype=torch.long)

        
        # Check for GPU cache
        if hasattr(self, "_gpu_input_cache") and hasattr(self, "_gpu_indices_sorted"):
            idx = indices.to(self._gpu_device)

            pos = torch.searchsorted(self._gpu_indices_sorted, idx)
            in_bounds = pos < self._gpu_indices_sorted.numel()
            in_cache = in_bounds & (self._gpu_indices_sorted[pos.clamp_max(self._gpu_indices_sorted.numel() - 1)] == idx)

            if in_cache.all():
                return (
                    self._gpu_input_cache[pos].contiguous(),
                    self._gpu_output_cache[pos].contiguous(),
                )
        
        # Fallback: use __getitem__ for each index
        batch = [self[idx.item() if isinstance(idx, torch.Tensor) else idx] for idx in indices]
        inputs = torch.stack([x for x, _ in batch])
        targets = torch.stack([y for _, y in batch])
        return inputs, targets
    
    def to_device(self, device: torch.device):
        """Move entire dataset to specified device.
        
        Default implementation for standard tensor datasets.
        Override to handle additional tensors (e.g., grids, masks).
        
        Args:
            device: Target device (cuda/mps/cpu)
            
        Returns:
            self (for chaining)
        """
        if not hasattr(self, 'load_in_memory') or not self.load_in_memory:
            return self
        
        # Move main data tensors
        if hasattr(self, 'input_data'):
            self.input_data = self.input_data.to(device)
        if hasattr(self, 'output_data'):
            self.output_data = self.output_data.to(device)
        
        # Move normalization stats if present
        if hasattr(self, 'input_mean'):
            self.input_mean = self.input_mean.to(device)
            self.input_std = self.input_std.to(device)
        if hasattr(self, 'output_mean'):
            self.output_mean = self.output_mean.to(device)
            self.output_std = self.output_std.to(device)
        
        return self
    
    def load_batch_indices_to_gpu(self, indices: Union[List[int], torch.Tensor], device: torch.device):
        """Load only specified indices to GPU (memory-efficient).
        
        Default implementation creates GPU cache for standard tensor datasets.
        Override for special data structures.
        
        Args:
            indices: Indices to load
            device: Target device
        """
        if not hasattr(self, 'load_in_memory') or not self.load_in_memory:
            return
        
        # Convert to tensor if needed
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        else:
            indices = indices.to(dtype=torch.long)

        indices_cpu = indices.cpu()
        sorted_idx, _ = torch.sort(indices_cpu)
        
        # Store cache metadata
        self._gpu_device = device
        self._gpu_indices_sorted = sorted_idx.to(device)
        
        self._gpu_input_cache = self.input_data[sorted_idx].contiguous().to(device)
        self._gpu_output_cache = self.output_data[sorted_idx].contiguous().to(device)
    
    def unload_from_gpu(self, indices: Optional[Union[List[int], torch.Tensor]] = None):
        """Free GPU memory by moving data back to CPU.
        
        Default implementation clears GPU cache.
        Override for datasets with additional cached tensors.
        
        Args:
            indices: Specific indices to unload (None = unload all/clear cache)
        """
        if not hasattr(self, 'load_in_memory') or not self.load_in_memory:
            return
        
        # Clear GPU cache
        if hasattr(self, '_gpu_input_cache'):
            del self._gpu_input_cache
            del self._gpu_output_cache
            del self._gpu_indices_sorted
            del self._gpu_device


class MLMCInMemoryDataset(PyGInMemoryDataset, ABC):
    """Abstract base class for PyG-based MLMC datasets.
    
    Dual inheritance: PyG's InMemoryDataset + MLMC interface.
    Use this for graph-based datasets (e.g., Navier-Stokes on meshes).
    
    Provides the same MLMC interface as MLMCDataset but inherits
    from PyG's InMemoryDataset for graph data handling.
    
    Required PyG methods to implement:
        - raw_file_names
        - processed_file_names
        - process()
    
    Required MLMC methods:
        - get_items(indices)
        - to_device(device)
        - load_batch_indices_to_gpu(indices, device)
        - unload_from_gpu(indices)
    """
    
    @abstractmethod
    def get_items(self, indices: Union[List[int], torch.Tensor]) -> Tuple:
        """Get multiple samples by indices (batch getter).
        
        Must be implemented by subclass - graph data requires special handling.
        
        Args:
            indices: List or tensor of sample indices
            
        Returns:
            (inputs, targets) - format depends on model (PyG Data, tensors, etc.)
        """
        pass
    
    @abstractmethod
    def to_device(self, device: torch.device):
        """Move dataset to specified device.
        
        Must handle PyG Data objects and any additional tensors.
        
        Args:
            device: Target device
            
        Returns:
            self (for chaining)
        """
        pass
    
    @abstractmethod
    def load_batch_indices_to_gpu(self, indices: Union[List[int], torch.Tensor], device: torch.device):
        """Load only specified indices to GPU.
        
        Must handle PyG Data objects.
        
        Args:
            indices: Indices to load
            device: Target device
        """
        pass
    
    @abstractmethod
    def unload_from_gpu(self, indices: Optional[Union[List[int], torch.Tensor]] = None):
        """Free GPU memory.
        
        Must handle PyG Data objects.
        
        Args:
            indices: Specific indices to unload (None = unload all)
        """
        pass
