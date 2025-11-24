import torch
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import time
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from typing import Union, List, Tuple
from torch_geometric.nn import fps
import random

from mlmc_optim.dataset import MLMCDataset

# Configuration parameters
PADDING_VALUE = -1000

# Batch wrapper for MLMC and GINOT forward compatibility
class JEBBatch:
    def __init__(self, pc_padded, xyt_padded):
        self.pc_padded = pc_padded
        self.xyt_padded = xyt_padded

    @property
    def shape(self):
        return self.pc_padded.shape

    def to(self, device):
        return JEBBatch(self.pc_padded.to(device), self.xyt_padded.to(device))

    def __iter__(self):
        return iter((self.pc_padded, self.xyt_padded))

    def __getitem__(self, idx):
        return (self.pc_padded, self.xyt_padded)[idx]

    def __len__(self):
        return self.pc_padded.shape[0]

# Define inverse transform functions at module level so they can be pickled
class InverseTransforms:
    @staticmethod
    def sigma_inverse(x, sigma_scale, sigma_shift):
        return x * sigma_scale + sigma_shift
    
    @staticmethod
    def pc_inverse(x, pc_scale, pc_shift):
        pc_scale_ = pc_scale.to(x.device)
        pc_shift_ = pc_shift.to(x.device)
        return x * pc_scale_ + pc_shift_
    
    @staticmethod
    def vert_inverse(x, vert_scale, vert_shift):
        vert_scale_ = vert_scale.to(x.device)
        vert_shift_ = vert_shift.to(x.device)
        return x * vert_scale_ + vert_shift_

def pad_collate_fn(batch):
    """Collate function for JEB batches.

    Returns generic (data, targets) tuples expected by the baseline
    training/eval loops, where:

        data    = (pc_padded, xyt_padded)
        targets = S_padded

    Any extra per-sample fields (e.g. sample_ids) are ignored.
    """
    pc_batch = [item[0] for item in batch]  # Extract pc (variable-length)
    xyt_batch = [item[1] for item in batch]  # Extract xyt (variable-length)
    S = [item[2] for item in batch]  # Extract S (variable-length)
    padding_value = PADDING_VALUE
    pc_padded = pad_sequence(pc_batch, batch_first=True, padding_value=padding_value)
    xyt_padded = pad_sequence(xyt_batch, batch_first=True, padding_value=padding_value)
    S_padded = pad_sequence(S, batch_first=True, padding_value=padding_value)

    data = (pc_padded, xyt_padded)
    targets = S_padded
    return data, targets


# Extracted LoadDataJEBGeo function from GINOT
def LoadDataJEBGeo(config, test_size=0.1, seed=42, padding_value=PADDING_VALUE):
    start = time.time()
    # load data
    data_file = os.path.join(config['data_dir'], "GEJetEngineBracket", "GE-JEB.pkl")
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    vertices = data['vertices']
    cells = data['cells']
    nodal_stress = data['nodal_stress']
    points_cloud = data['points_cloud']
    # normalize data
    vert_concat = np.concatenate(vertices, axis=0, dtype=np.float32)
    vert_shift, vert_scale = np.mean(
        vert_concat, axis=0), np.std(vert_concat, axis=0)
    vert_shift = vert_shift[None, :]  # (1,3)
    vert_scale = vert_scale[None, :]

    pc_concat = np.concatenate(points_cloud, axis=0, dtype=np.float32)
    pc_shift, pc_scale = np.mean(pc_concat, axis=0), np.std(pc_concat, axis=0)
    pc_shift = pc_shift[None, :]
    pc_scale = pc_scale[None, :]

    sigma_concat = np.concatenate(nodal_stress, axis=0, dtype=np.float32)
    sigma_shift, sigma_scale = np.mean(
        sigma_concat), np.std(sigma_concat)

    vertices_norm = [torch.tensor(
        (v.astype(np.float32)-vert_shift)/vert_scale) for v in vertices]
    pc_norm = [torch.tensor((pc.astype(np.float32)-pc_shift)/pc_scale)
               for pc in points_cloud]
    sigma_norm = [torch.tensor((s.astype(np.float32)-sigma_shift)/sigma_scale)
                  for s in nodal_stress]
    pc_shift = torch.tensor(pc_shift)[None, :]
    pc_scale = torch.tensor(pc_scale)[None, :]
    vert_shift = torch.tensor(vert_shift)[None, :]  # (1, 1,3)
    vert_scale = torch.tensor(vert_scale)[None, :]
    # split test and train data
    train_ids, test_ids = train_test_split(
        np.arange(len(cells)), test_size=test_size, random_state=seed, shuffle=False)
    train_pc = [pc_norm[i] for i in train_ids]
    test_pc = [pc_norm[i] for i in test_ids]
    train_xyt = [vertices_norm[i] for i in train_ids]
    test_xyt = [vertices_norm[i] for i in test_ids]
    train_S = [sigma_norm[i] for i in train_ids]
    test_S = [sigma_norm[i] for i in test_ids]
    train_dataset = ListDataset(
        (train_pc, train_xyt, train_S, torch.tensor(train_ids)))
    test_dataset = ListDataset(
        (test_pc, test_xyt, test_S, torch.tensor(test_ids)))

    # Create inverse transforms using the module-level class for picklability
    s_inverse_fn = lambda x: InverseTransforms.sigma_inverse(x, sigma_scale, sigma_shift)
    pc_inverse_fn = lambda x: InverseTransforms.pc_inverse(x, pc_scale, pc_shift)
    vert_inverse_fn = lambda x: InverseTransforms.vert_inverse(x, vert_scale, vert_shift)
    
    # Store the parameters needed for inverse transforms
    inverse_params = {
        'sigma_scale': sigma_scale,
        'sigma_shift': sigma_shift,
        'pc_scale': pc_scale,
        'pc_shift': pc_shift,
        'vert_scale': vert_scale,
        'vert_shift': vert_shift
    }
    
    print(f"Data loading time: {time.time()-start:.2f} s")
    return train_dataset, test_dataset, cells, s_inverse_fn, pc_inverse_fn, vert_inverse_fn, inverse_params

# ListDataset class extracted from GINOT
class ListDataset(Dataset):
    """For list of tensors"""
    
    def __init__(self, data: Union[list, tuple]):
        """
        args:
            data: list of data, each element is a list of tensors
            e.g. [(pc1, xyt1, S1), (pc2, xyt2, S2), ...]
        """
        self.data = data
    
    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        one_data = [d[idx] for d in self.data]
        return one_data


class JEBDataset(MLMCDataset):
    """Dataset for loading Jet Engine Bracket (JEB) data.

    This dataset is designed to work with the GINOT model and to be
    compatible with the MLMC training pipeline.
    
    Inherits from MLMCDataset but overrides get_items for variable-length
    point cloud data with padding.
    """

    def __init__(self, config, root="./jeb_processed", data_dir="./jeb_data", resolution=0, pc_resolution=None, xy_resolution=None, 
                 train=True, use_coarse_pc=False, load_in_memory=True, normalize=True, indices=None):
        use_coarse_pc = config.get('ginot_coarse_pc', use_coarse_pc)
        self.config = config
        self.resolution = resolution
        # Use base resolution as default if specific ones aren't provided
        self.pc_resolution = pc_resolution if pc_resolution is not None else resolution
        self.xy_resolution = xy_resolution if xy_resolution is not None else resolution
        self.train = train
        self.load_in_memory = load_in_memory
        self.normalize = normalize
        self.data_dir = data_dir  # Source data directory
        self.root = root  # Processed data save directory
        self.use_coarse_pc = use_coarse_pc

        # Load or process data
        if not self._load_preprocessed_data():
            self._load_data()

        # Handle indices for MLMC compatibility
        if indices is not None and resolution in indices:
            self.indices = indices[resolution]
        else:
            self.indices = list(range(len(self.pc_data)))

        # --- Compute stress normalization stats on all targets (after data loaded) ---
        all_targets = [t.flatten() for t in self.target_data]
        all_targets_concat = torch.cat(all_targets)
        self.target_mean = all_targets_concat.mean()
        self.target_std = all_targets_concat.std()
        
        # Create aliases for base class compatibility (lists, not tensors)
        self.input_data = self.pc_data
        self.output_data = self.target_data
    
    def _load_preprocessed_data(self):
        """Try to load preprocessed data for this resolution"""
        data_path = os.path.join(self.root, f"preprocessed_r{self.resolution}_data.pt")
        
        if os.path.exists(data_path):
            print(f"Loading preprocessed data from {data_path}")
            data = torch.load(data_path)
            # Convert to tensors if they're numpy arrays
            self.pc_data = [torch.tensor(pc) if not torch.is_tensor(pc) else pc for pc in data['pc']]
            self.xy_data = [torch.tensor(xy) if not torch.is_tensor(xy) else xy for xy in data['xy']]
            self.target_data = [torch.tensor(target) if not torch.is_tensor(target) else target for target in data['targets']]
            
            # Always load coarse point clouds if available
            if 'pc_coarse' in data:
                self.pc_coarse = [torch.tensor(pc) if not torch.is_tensor(pc) else pc for pc in data['pc_coarse']]
            else:
                self.pc_coarse = None
            
            # Load metadata and inverse transform parameters
            meta_path = os.path.join(self.root, f"metadata_r{self.resolution}.pt")
            if os.path.exists(meta_path):
                meta = torch.load(meta_path)
                params = meta['inverse_params']
                self.inverse_params = params
                
                # Recreate the inverse transform functions
                self.s_inverse = lambda x: InverseTransforms.sigma_inverse(x, params['sigma_scale'], params['sigma_shift'])
                self.pc_inverse = lambda x: InverseTransforms.pc_inverse(x, params['pc_scale'], params['pc_shift'])
                self.vert_inverse = lambda x: InverseTransforms.vert_inverse(x, params['vert_scale'], params['vert_shift'])
            
            return True
        return False


    def _load_data(self):
        """Load data - either preprocessed or by generating it on the fly. Handles missing metadata by computing stats from data if needed."""
        # Paths for preprocessed data and meta
        data_path = os.path.join(self.root, f"preprocessed_r{self.resolution}_data.pt")
        meta_path = os.path.join(self.root, f"metadata_r{self.resolution}.pt")

        # Try to load preprocessed data first
        if self._load_preprocessed_data():
            print(f"Successfully loaded preprocessed data for resolution {self.resolution}")
            # Try to load metadata
            if os.path.exists(meta_path):
                meta = torch.load(meta_path)
                print(f"Loaded normalization metadata from {meta_path}")
            else:
                print(f"Metadata file {meta_path} not found. Computing stats from preprocessed data...")
                # Compute mean and std for each
                def compute_stats(data_list):
                    arrs = [x.numpy() if hasattr(x, 'numpy') else x for x in data_list]
                    arrs = [x.flatten() for x in arrs]
                    arr = np.concatenate(arrs)
                    return float(arr.mean()), float(arr.std())
                sigma_mean, sigma_std = compute_stats(self.target_data)
                pc_mean, pc_std = compute_stats(self.pc_data)
                vert_mean, vert_std = compute_stats(self.vert_data)
                meta = {
                    'sigma_scale': sigma_std,
                    'sigma_shift': sigma_mean,
                    'pc_scale': pc_std,
                    'pc_shift': pc_mean,
                    'vert_scale': vert_std,
                    'vert_shift': vert_mean,
                }
                torch.save(meta, meta_path)
                print(f"Saved computed normalization metadata to {meta_path}")
            # Set up inverse transform functions
            self.s_inverse = lambda x: x * meta['inverse_params']['sigma_scale'] + meta['inverse_params']['sigma_shift']
            self.pc_inverse = lambda x: x * meta['inverse_params']['pc_scale'] + meta['inverse_params']['pc_shift']
            self.vert_inverse = lambda x: x * meta['inverse_params']['vert_scale'] + meta['inverse_params']['vert_shift']
            self.normalization_meta = meta
            return True
        
        print(f"Loading data for resolution {self.resolution}...")
        try:
            # Call our extracted LoadDataJEBGeo function directly
            train_dataset, test_dataset, _, self.s_inverse, self.pc_inverse, self.vert_inverse, self.inverse_params = LoadDataJEBGeo(
                config=self.config,
                test_size=self.config.get('test_split', 0.1),
                seed=self.config.get('seed', 42),
            )

            # Select the appropriate dataset
            dataset = train_dataset if self.train else test_dataset

            # Extract raw data
            pc_list = []
            pc_coarse_list = []
            xy_list = []
            target_list = []

            for i in tqdm(range(len(dataset)), desc=f"Processing {('train' if self.train else 'test')} data"):
                sample = dataset[i]
                pc = sample[0]
                xy = sample[1]
                target = sample[2]

                # Apply coarsening based on resolution
                if self.resolution > 0:
                    xy_n_points = len(xy)
                    target_n_points = max(10, xy_n_points // (2 ** self.resolution))
                    print(f"Using XY size ({xy_n_points}) as basis for coarsening to {target_n_points} points at resolution {self.resolution}")

                    original_pc_points = len(pc) if not torch.is_tensor(pc) else pc.shape[0]
                    target_pc_points = max(10, original_pc_points // (2 ** self.resolution))
                    print(f"Using PC size ({original_pc_points}) as basis for coarsening to {target_pc_points} points at resolution {self.resolution}")

                    # Convert to numpy if it's a tensor
                    xy_np = xy.numpy() if torch.is_tensor(xy) else xy
                    pc_np = pc.numpy() if torch.is_tensor(pc) else pc

                    xy_downsampled, indices = self._coarsen_point_cloud(xy_np, target_n_points, return_indices=True)
                    target = target[indices] if torch.is_tensor(target) else target[indices]
                    pc_downsampled = self._coarsen_point_cloud(pc_np, target_pc_points)

                    xy = torch.from_numpy(xy_downsampled).float() if isinstance(xy, np.ndarray) else xy_downsampled
                    pc = torch.from_numpy(pc_downsampled).float() if isinstance(pc, np.ndarray) else pc_downsampled

                pc_list.append(pc)
                xy_list.append(xy)
                target_list.append(target)

            # Store as lists since point clouds have variable sizes
            # Don't attempt to stack variable-sized tensors
            self.pc_data = pc_list
            self.xy_data = xy_list
            self.target_data = target_list

            print(f"Loaded {len(self.pc_data)} samples with variable sizes")
            print(f"Sample point cloud shapes: {[pc.shape for pc in self.pc_data[:3]]}")

            # Create destination directory if it doesn't exist
            os.makedirs(self.root, exist_ok=True)

            # Save preprocessed data with resolution in filename
            data_path = os.path.join(self.root, f"preprocessed_r{self.resolution}_data.pt")

            data = {
                'pc': self.pc_data,
                'xy': self.xy_data,
                'targets': self.target_data
            }

            print(f"Saving preprocessed data to {data_path}")
            torch.save(data, data_path)

            # Save inverse transforms and metadata
            meta_path = os.path.join(self.root, f"metadata_r{self.resolution}.pt")

            # Store the parameters for inverse transforms instead of the functions
            meta = {
                'resolution': self.resolution,
                'train': self.train,
                'inverse_params': self.inverse_params
            }
            torch.save(meta, meta_path)

        except Exception as e:
            print(f"Error loading JEB data: {e}")
            raise


    def _coarsen_point_cloud(self, points, n_samples, return_indices=False):
        """Try coarsening with FPS -> KMeans -> Random cascade"""
        if len(points) <= n_samples:
            indices = np.arange(len(points))
            return (points, indices) if return_indices else points
        
        try:
            # Try FPS first (best quality)
            points_tensor = torch.tensor(points, dtype=torch.float32) if not torch.is_tensor(points) else points
            batch = torch.zeros(len(points_tensor), dtype=torch.long)
            idx = fps(points_tensor, batch, ratio=n_samples/len(points))
            coarsened = points[idx.numpy() if torch.is_tensor(idx) else idx]
            return (coarsened, idx.numpy() if torch.is_tensor(idx) else idx) if return_indices else coarsened
        except Exception as e:
            print(f"FPS failed: {e}, trying KMeans")
            
            try:
                # Try KMeans (good quality, slower)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init='auto')
                kmeans.fit(points)
                # Find closest points to centroids
                distances = np.linalg.norm(points[:, None] - kmeans.cluster_centers_, axis=2)
                indices = np.argmin(distances, axis=0)
                coarsened = points[indices]
                return (coarsened, indices) if return_indices else coarsened
            except Exception as e:
                print(f"KMeans failed: {e}, using random sampling")

                # Fall back to random sampling (fast but lower quality)
                indices = np.random.choice(len(points), n_samples, replace=False)
                coarsened = points[indices]
                return (coarsened, indices) if return_indices else coarsened

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Map idx to the actual index if using subset
        real_idx = self.indices[idx]

        # Select between original and coarse point cloud
        if self.use_coarse_pc:
            if self.pc_coarse is None:
                raise RuntimeError("Coarse point cloud not loaded. Call _load_preprocessed_pc_coarse() or set use_coarse_pc=False.")
            pc = self.pc_coarse[real_idx]
        else:
            pc = self.pc_data[real_idx]
        xy = self.xy_data[real_idx]
        target = self.target_data[real_idx]

        # --- Normalize the target on the fly ---
        target_norm = (target - self.target_mean) / (self.target_std + 1e-8)
        return pc, xy, target_norm



    def get_items(self, indices):
        """Get multiple items at once - for compatibility with MLMC pipeline
        Uses padding to handle variable length point clouds and features
        Returns:
            (inputs, targets):
                inputs: tuple (pc_padded, xyt_padded)
                targets: target_padded
        """
        pc_batch = []
        xyt_batch = []
        target_batch = []
        
        for idx in indices:
            real_idx = self.indices[idx]
            if self.use_coarse_pc:
                if self.pc_coarse is None:
                    raise RuntimeError("Coarse point cloud not loaded. Either load a dataset that has coarse point clouds or set use_coarse_pc=False.")
                pc_batch.append(self.pc_coarse[real_idx])
            else:
                pc_batch.append(self.pc_data[real_idx])
            xyt_batch.append(self.xy_data[real_idx])
            # --- Normalize each target on the fly ---
            target = self.target_data[real_idx]
            target_norm = (target - self.target_mean) / (self.target_std + 1e-8)
            target_batch.append(target_norm)
        
        # Apply padding for variable-length tensors
        pc_padded = pad_sequence(pc_batch, batch_first=True, padding_value=PADDING_VALUE)
        xyt_padded = pad_sequence(xyt_batch, batch_first=True, padding_value=PADDING_VALUE)
        target_padded = pad_sequence(target_batch, batch_first=True, padding_value=PADDING_VALUE)
        return JEBBatch(pc_padded, xyt_padded), target_padded

        
    def to_device(self, device):
        """Move data to device - for compatibility with MLMC pipeline"""
        if self.load_in_memory:# and torch.is_tensor(self.pc_data):
            self.pc_data = self.pc_data.to(device)
            self.xy_data = self.xy_data.to(device)
            self.target_data = self.target_data.to(device)
            
    def get_inverse_transforms(self):
        """Return the inverse transforms for post-processing"""
        return {
            's_inverse': self.s_inverse,
            'pc_inverse': self.pc_inverse,
            'vert_inverse': self.vert_inverse
        }
        
    def load_batch_indices_to_gpu(self, indices, device):
        """Load only the data for specified indices to GPU.
        Optimized for bulk transfers to minimize device operations.
        """
        # Convert indices to Python list if needed
        idx_list = indices.tolist() if hasattr(indices, 'tolist') else indices
        
        # Check if data is stored as tensor or list of tensors
        if isinstance(self.pc_data, torch.Tensor):
            # Tensor mode - use direct indexing
            self.pc_data[indices] = self.pc_data[indices].to(device)
            self.xy_data[indices] = self.xy_data[indices].to(device)
            self.target_data[indices] = self.target_data[indices].to(device)
        else:
            # List mode - faster with operator[] access and single list comprehension
            # This creates new lists with GPU tensors for the selected indices
            for idx in idx_list:
                self.pc_data[idx] = self.pc_data[idx].to(device)
                self.xy_data[idx] = self.xy_data[idx].to(device)
                self.target_data[idx] = self.target_data[idx].to(device)
        
    def unload_from_gpu(self, indices=None):
        """Explicitly move data from GPU back to CPU to free up memory."""
        if isinstance(self.pc_data, torch.Tensor):
            self.pc_data[indices] = self.pc_data[indices].cpu()
            self.xy_data[indices] = self.xy_data[indices].cpu()
            self.target_data[indices] = self.target_data[indices].cpu()
        else:
            for idx in indices:
                self.pc_data[idx] = self.pc_data[idx].cpu()
                self.xy_data[idx] = self.xy_data[idx].cpu()
                self.target_data[idx] = self.target_data[idx].cpu()


def create_multi_resolution_jeb_data(config, output_dir="./jeb_data", resolutions=None):
    """
    Create and save multi-resolution datasets for the Jet Engine Bracket.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save the processed data
        resolutions: List of resolution levels (0=finest, higher=coarser)
    """
    # Make the data_dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process source dir from config or use default
    source_dir = config['data_dir']
    
    # Loop through resolutions and initialize datasets
    # This will trigger data processing and saving
    for res in resolutions:
        print(f"\nProcessing resolution {res}...")
        
        # Create processed directory paths based on resolution
        processed_dir = os.path.join(output_dir, f"level_{res}")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Create train dataset
        train_dataset = JEBDataset(
            config=config,
            root=os.path.join(processed_dir, "train"), 
            data_dir=source_dir,
            resolution=res, 
            train=True
        )
        
        # Create test dataset
        test_dataset = JEBDataset(
            config=config,
            root=os.path.join(processed_dir, "test"), 
            data_dir=source_dir,
            resolution=res, 
            train=False
        )
        
        print(f"Resolution {res} datasets created and saved successfully.")
        
    print(f"Multi-resolution JEB data processing complete!")


def collect_dataset_statistics(config, resolutions):
    """Collect statistics for each resolution level of the dataset."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Configure pandas to show all columns in tabular format with good readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)  # Limit decimal places
    pd.set_option('display.expand_frame_repr', False)  # Don't wrap to multiple lines
    
    stats = []
    
    # Use the same processed directory where data was originally saved
    processed_dir = config['data_dir']
    source_dir = config['data_dir']
    
    for res in resolutions:
        print(f"Loading resolution {res} datasets...")
        # Load train dataset
        try:
            level_dir = os.path.join(processed_dir, f"level_{res}")
            train_dataset = JEBDataset(
                config=config,
                root=os.path.join(level_dir, "train"),
                data_dir=source_dir,
                resolution=res, 
                train=True
            )
            # Load test dataset
            test_dataset = JEBDataset(
                config=config,
                root=os.path.join(level_dir, "test"),
                data_dir=source_dir,
                resolution=res, 
                train=False
            )
            
            # Collect XY point count and point cloud statistics
            train_point_counts = []
            train_pc_counts = []
            train_solution_values = []
            for i in tqdm(range(len(train_dataset)), desc=f"Processing train data res={res}"):
                # JEBDataset returns a tuple (pc, xy, target)
                pc, xy, target = train_dataset[i]
                train_point_counts.append(len(xy))
                train_pc_counts.append(len(pc))  # Count points in point cloud
                if isinstance(target, torch.Tensor):
                    train_solution_values.extend(target.tolist())
                else:
                    train_solution_values.extend(target)
            
            test_point_counts = []
            test_pc_counts = []
            test_solution_values = []
            for i in tqdm(range(len(test_dataset)), desc=f"Processing test data res={res}"):
                # JEBDataset returns a tuple (pc, xy, target)
                pc, xy, target = test_dataset[i]
                test_point_counts.append(len(xy))
                test_pc_counts.append(len(pc))  # Count points in point cloud
                if isinstance(target, torch.Tensor):
                    test_solution_values.extend(target.tolist())
                else:
                    test_solution_values.extend(target)
            
            # Compute statistics
            train_points_mean = np.mean(train_point_counts)
            train_points_std = np.std(train_point_counts)
            train_points_min = np.min(train_point_counts)
            train_points_max = np.max(train_point_counts)
            
            # Calculate point cloud statistics
            train_pc_mean = np.mean(train_pc_counts)
            train_pc_std = np.std(train_pc_counts)
            train_pc_min = np.min(train_pc_counts)
            train_pc_max = np.max(train_pc_counts)
            
            test_points_mean = np.mean(test_point_counts)
            test_points_std = np.std(test_point_counts)
            test_points_min = np.min(test_point_counts)
            test_points_max = np.max(test_point_counts)
            
            # Calculate point cloud statistics
            test_pc_mean = np.mean(test_pc_counts)
            test_pc_std = np.std(test_pc_counts)
            test_pc_min = np.min(test_pc_counts)
            test_pc_max = np.max(test_pc_counts)
            
            train_values_mean = np.mean(train_solution_values)
            train_values_std = np.std(train_solution_values)
            train_values_min = np.min(train_solution_values)
            train_values_max = np.max(train_solution_values)
            
            test_values_mean = np.mean(test_solution_values)
            test_values_std = np.std(test_solution_values)
            test_values_min = np.min(test_solution_values)
            test_values_max = np.max(test_solution_values)
            
            stats.append({
                'resolution': res,
                'train_samples': len(train_dataset),
                'test_samples': len(test_dataset),
                'train_xy_points_mean': train_points_mean,
                'train_xy_points_std': train_points_std,
                'train_xy_points_min': train_points_min,
                'train_xy_points_max': train_points_max,
                'test_xy_points_mean': test_points_mean,
                'test_xy_points_std': test_points_std,
                'test_xy_points_min': test_points_min,
                'test_xy_points_max': test_points_max,
                # Add point cloud statistics
                'train_pc_points_mean': train_pc_mean,
                'train_pc_points_std': train_pc_std,
                'train_pc_points_min': train_pc_min,
                'train_pc_points_max': train_pc_max,
                'test_pc_points_mean': test_pc_mean,
                'test_pc_points_std': test_pc_std,
                'test_pc_points_min': test_pc_min,
                'test_pc_points_max': test_pc_max,
                'train_values_mean': train_values_mean,
                'train_values_std': train_values_std,
                'train_values_min': train_values_min,
                'train_values_max': train_values_max,
                'test_values_mean': test_values_mean,
                'test_values_std': test_values_std,
                'test_values_min': test_values_min,
                'test_values_max': test_values_max,
            })
            print(f"Completed statistics for resolution {res}")
        except Exception as e:
            print(f"Error processing resolution {res}: {e}")

    return pd.DataFrame(stats)


def plot_statistics(stats_df):
    """Plot the statistics for each resolution level."""
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Plot XY point count statistics
    plt.figure(figsize=(12, 8))
    plt.errorbar(stats_df['resolution'], stats_df['train_xy_points_mean'], 
                yerr=stats_df['train_xy_points_std'], 
                fmt='o-', label='Train')
    plt.errorbar(stats_df['resolution'], stats_df['test_xy_points_mean'], 
                yerr=stats_df['test_xy_points_std'], 
                fmt='s-', label='Test')
    plt.xlabel('Resolution Level')
    plt.ylabel('Average Number of XY Points')
    plt.title('Average Number of XY Points vs Resolution Level')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/xy_point_count_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot solution value statistics
    plt.figure(figsize=(12, 8))
    plt.errorbar(stats_df['resolution'], stats_df['train_values_mean'], 
                yerr=stats_df['train_values_std'], 
                fmt='o-', label='Train')
    plt.errorbar(stats_df['resolution'], stats_df['test_values_mean'], 
                yerr=stats_df['test_values_std'], 
                fmt='s-', label='Test')
    plt.xlabel('Resolution Level')
    plt.ylabel('Average Solution Value')
    plt.title('Solution Value Statistics vs Resolution Level')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/solution_value_stats.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more detailed tabular view of the statistics
    plt.figure(figsize=(14, 8))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    col_labels = ['Resolution', 'Train PC Points (Mean±Std)', 'Test PC Points (Mean±Std)', 
                'Train Values (Mean±Std)', 'Test Values (Mean±Std)']
    table_data = []
    
    for _, row in stats_df.iterrows():
        # For backward compatibility: if no pc points stats, use xy points
        train_pc_mean = row.get('train_pc_points_mean', row['train_xy_points_mean'])
        train_pc_std = row.get('train_pc_points_std', row['train_xy_points_std'])
        test_pc_mean = row.get('test_pc_points_mean', row['test_xy_points_mean'])
        test_pc_std = row.get('test_pc_points_std', row['test_xy_points_std'])
        
        table_data.append([
            str(int(row['resolution'])),
            f"{train_pc_mean:.1f} ± {train_pc_std:.1f}",
            f"{test_pc_mean:.1f} ± {test_pc_std:.1f}",
            f"{row['train_values_mean']:.4f} ± {row['train_values_std']:.4f}",
            f"{row['test_values_mean']:.4f} ± {row['test_values_std']:.4f}"
        ])
    
    table = plt.table(cellText=table_data, colLabels=col_labels, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.savefig('figures/dataset_stats_table.png', dpi=300, bbox_inches='tight')
    
    return


def plot_prediction_jeb(config, model, data_loader, epoch, device, N=4, show=False, save_path=None):
    """
    Plot up to N examples of JEB predictions vs ground truth from the data_loader.
    Each row: [Ground Truth | Prediction | Error] for a single sample.
    Uses 3D scatter on (x, y, z) plane. All device/model logic is internal.
    """

    plotted = 0
    fig = plt.figure(figsize=(15, 4 * N))
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'Eval ({config["dataset"]}-{config["loss_type"]})', leave=False)
        for batch in pbar:
            if config['dataset'] == 'jeb':
                # Move each tensor in batch to device
                pc = batch[0].to(device)
                xy = batch[1].to(device)
                target = batch[2].to(device)
                pred = model((pc, xy)).detach().cpu()
                pc, xy, target = pc.cpu(), xy.cpu(), target.cpu()
                # If single sample, unsqueeze
                if pc.ndim == 2:
                    pc = pc.unsqueeze(0)
                    xy = xy.unsqueeze(0)
                    target = target.unsqueeze(0)
                    pred = pred.unsqueeze(0)
                batch_size = pc.shape[0]
                for i in range(batch_size):
                    if plotted >= N:
                        break
                    
                    # JEBDataset returns a tuple (pc, xy, target)
                    pc_i = pc[i].numpy() if isinstance(pc[i], torch.Tensor) else pc[i]
                    xy_i = xy[i].numpy() if isinstance(xy[i], torch.Tensor) else xy[i]
                    tgt_i = target[i].numpy() if isinstance(target[i], torch.Tensor) else target[i]
                    pred_i = pred[i].numpy() if isinstance(pred[i], torch.Tensor) else pred_i
                    
                    # Prepare data for plotting
                    # Fetch s_inverse from dataset if available
                    s_inverse = None
                    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 's_inverse'):
                        s_inverse = data_loader.dataset.s_inverse
                    # Denormalize if s_inverse is available
                    if s_inverse is not None:
                        try:
                            values_tgt = s_inverse(tgt_i)
                            values_pred = s_inverse(pred_i)
                        except Exception as e:
                            print(f"Denormalization failed: {e}. Plotting raw values.")
                            values_tgt = tgt_i
                            values_pred = pred_i
                    else:
                        values_tgt = tgt_i
                        values_pred = pred_i
                    
                    # Check which components match in length
                    if isinstance(tgt_i, np.ndarray) or isinstance(tgt_i, torch.Tensor):
                        if len(tgt_i) == len(pc_i):
                            points = pc_i
                            values_tgt = values_tgt
                            values_pred = values_pred
                        elif len(tgt_i) == len(xy_i):
                            points = xy_i
                            values_tgt = values_tgt
                            values_pred = values_pred
                        else:
                            print("  No matching dimensions - using PC with uniform color")
                            points = pc_i
                            values_tgt = np.ones(len(pc_i)) * (float(i) / batch_size)
                            values_pred = np.ones(len(pc_i)) * (float(i) / batch_size)
                    else:
                        # Default to PC with uniform colors
                        points = pc_i
                        values_tgt = np.ones(len(pc_i)) * (float(i) / batch_size)
                        values_pred = np.ones(len(pc_i)) * (float(i) / batch_size)
                    
                    # Normalize values for coloring
                    if isinstance(values_tgt, torch.Tensor):
                        values_tgt = values_tgt.numpy()
                    if isinstance(values_pred, torch.Tensor):
                        values_pred = values_pred.numpy()
                    
                    # Ensure we're working with numpy arrays
                    if isinstance(points, torch.Tensor):
                        points = points.numpy()
                    
                    # Mask out padded points
                    valid_mask = ~np.any(points == PADDING_VALUE, axis=1)
                    # Create a subplot for ground truth
                    ax = fig.add_subplot(N, 3, plotted*3 + 1, projection='3d')
                    scatter_tgt = ax.scatter(
                        points[valid_mask, 0], points[valid_mask, 1], points[valid_mask, 2],
                        c=values_tgt[valid_mask], 
                        cmap='viridis', 
                        s=5,
                        alpha=0.8
                    )
                    ax.set_title(f"Ground truth {plotted}")
                    cbar_tgt = plt.colorbar(scatter_tgt, ax=ax, pad=0.1)
                    cbar_tgt.set_label('Stress field')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                    # Create a subplot for prediction
                    ax = fig.add_subplot(N, 3, plotted*3 + 2, projection='3d')
                    scatter_pred = ax.scatter(
                        points[valid_mask, 0], points[valid_mask, 1], points[valid_mask, 2], 
                        c=values_pred[valid_mask], 
                        cmap='viridis', 
                        s=5,
                        alpha=0.8
                    )
                    ax.set_title(f"Prediction {plotted}")
                    cbar_pred = plt.colorbar(scatter_pred, ax=ax, pad=0.1)
                    cbar_pred.set_label('Stress field')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                    # Create a subplot for error
                    ax = fig.add_subplot(N, 3, plotted*3 + 3, projection='3d')
                    scatter_err = ax.scatter(
                        points[valid_mask, 0], points[valid_mask, 1], points[valid_mask, 2], 
                        c=np.abs((values_pred - values_tgt)[valid_mask])/np.abs(values_tgt[valid_mask]),
                        cmap='viridis', 
                        s=5,
                        alpha=0.8,
                        vmin=0,
                        vmax=1
                    )
                    ax.set_title(f"Relative error {plotted}")
                    cbar_err = plt.colorbar(scatter_err, ax=ax, pad=0.1)
                    cbar_err.set_label('Error')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                    plotted += 1
                    if plotted >= N:
                        break
            if plotted >= N:
                break
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    elif show:
        plt.show()
    return fig


def ginot_forward_pass(config, resolutions):
    """Test forward pass with GINOT model on JEB data at different resolutions."""
    import sys
    import torch.nn as nn
    
    # Use our local GINOT model implementation instead
    try:
        print("Using local GINOT model implementation...")
        from ginot_model import Trunk, PointCloudPerceiverChannelsEncoder
        print("Successfully imported local GINOT model components.")
    except ImportError as e:
        print(f"Could not import local GINOT model components: {e}")
        print("\nMake sure ginot_model.py is in the same directory as data_jeb.py")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n\nRunning GINOT model forward pass test on device: {device}\n" + "-"*50)
    
    # Define branch encoder parameters
    branch_args = {
        'in_dim': 3,             # Input dimension (3D points)
        'global_pooling': True,   # Use global pooling
        'latent_dim': 64,         # Latent dimension
        'embed_dim': 64,          # Embedding dimension
        'num_layers': 4,          # Number of layers
        'num_heads': 4,           # Number of attention heads
        'padding_value': -1000,   # Padding value
        'dropout': 0.0,           # Dropout rate
    }
    
    # Define trunk parameters
    trunk_args = {
        'embed_dim': 64,          # Embedding dimension
        'cross_attn_layers': 4,   # Cross attention layers
        'num_heads': 4,           # Number of attention heads
        'in_channels': 3,         # Input channels (3D points)
        'out_channels': 1,        # Output channels (scalar stress)
        'dropout': 0.0,           # Dropout rate
        'emd_version': 'nerf',    # Position encoding version
        'padding_value': -1000,   # Padding value
    }
    
    # Create branch and trunk models
    branch = PointCloudPerceiverChannelsEncoder(config, **branch_args).to(device)
    model = Trunk(config, branch, **trunk_args).to(device)
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Test forward pass at each resolution
    for res in resolutions:
        print(f"\nTesting resolution {res}:")
        
        # Create dataset with correct path structure
        level_dir = os.path.join(config['data_dir'], f"level_{res}")
        dataset = JEBDataset(
            config=config,
            root=os.path.join(level_dir, "train"),
            data_dir=config['data_dir'],
            resolution=res,
            train=True,
            load_in_memory=True,
            normalize=True
        )
        
        # Get a single batch
        batch_size = min(4, len(dataset))
        indices = torch.randint(0, len(dataset), (batch_size,))
        (inputs, targets) = dataset.get_items(indices)
        (pc_padded, xyt_padded) = inputs
        
        try:
            with torch.no_grad():
                output, mask = model(pc_padded, xyt_padded)
                print(f"  Output shape: {output.shape}")
                print(f"  Mask shape: {mask.shape}")
            
            # Calculate basic statistics for outputs
            valid_outputs = output[mask]
            print(f"  Valid output shape: {valid_outputs.shape}")
            print(f"  Output stats - Min: {valid_outputs.min().item():.4f}, Max: {valid_outputs.max().item():.4f}, Mean: {valid_outputs.mean().item():.4f}")

            # Example: Apply inverse transform (uncomment and adapt as needed)
            # output_inv = model.inverse_transform(output, mask, inverse_fn=InverseTransforms.sigma_inverse, sigma_scale=..., sigma_shift=...)
            # valid_outputs_inv = output_inv[mask]
        
        except Exception as e:
            print(f"  Forward pass failed: {e}")
    
    print("-"*50)


def fig1_plot(config, resolutions, num_examples=3, output_dir='../figures/fig1_components'):
    """
    Create individual component plots for Figure 1 assembly.
    Generates separate files for each geometry at each resolution.
    
    Outputs:
    - geometry_only_res{res}_example{idx}.pdf - Just geometry, no color
    - geometry_colored_res{res}_example{idx}.pdf - Geometry with stress field, no axis/colorbar
    
    Args:
        config: Configuration dictionary
        resolutions: List of resolution levels to plot
        num_examples: Number of example geometries to show
        output_dir: Directory to save individual component plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processed_dir = config['data_dir']
    source_dir = config['data_dir']
    
    # Load the datasets
    datasets = {}
    for res in resolutions:
        level_dir = os.path.join(processed_dir, f"level_{res}")
        datasets[res] = JEBDataset(
            config=config,
            root=os.path.join(level_dir, "train"),
            data_dir=source_dir,
            resolution=res, 
            train=True
        )
    
    # Select random examples (same indices across resolutions)
    dataset_size = len(datasets[resolutions[0]])
    example_indices = random.sample(range(dataset_size), num_examples)
    
    for res in resolutions:
        dataset = datasets[res]
        
        for idx in example_indices:
            pc, xy, target = dataset[idx]
            
            points = xy
            if isinstance(points, torch.Tensor):
                points = points.numpy()
            if isinstance(target, torch.Tensor):
                target = target.numpy()
            
            # 1. Geometry only (no color)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c='gray', s=1, alpha=0.6, edgecolors='none')
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(False)
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            
            # Equal aspect
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            z_lim = ax.get_zlim()
            x_center = np.mean(x_lim)
            y_center = np.mean(y_lim)
            z_center = np.mean(z_lim)
            max_radius = max(
                max(abs(x_lim[0] - x_center), abs(x_lim[1] - x_center)),
                max(abs(y_lim[0] - y_center), abs(y_lim[1] - y_center)),
                max(abs(z_lim[0] - z_center), abs(z_lim[1] - z_center))
            )
            ax.set_xlim3d([x_center - max_radius, x_center + max_radius])
            ax.set_ylim3d([y_center - max_radius, y_center + max_radius])
            ax.set_zlim3d([z_center - max_radius, z_center + max_radius])
            
            plt.savefig(f"{output_dir}/geometry_only_res{res}_example{idx}.pdf", 
                       bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close()
            
            # 2. Geometry with stress field (no axis, no colorbar)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=target, cmap='viridis', s=1, alpha=0.8, edgecolors='none')
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(False)
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            
            ax.set_xlim3d([x_center - max_radius, x_center + max_radius])
            ax.set_ylim3d([y_center - max_radius, y_center + max_radius])
            ax.set_zlim3d([z_center - max_radius, z_center + max_radius])
            
            plt.savefig(f"{output_dir}/geometry_colored_res{res}_example{idx}.pdf", 
                       bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close()
            
            print(f"Saved plots for resolution {res}, example {idx}")
    
    print(f"\nAll component plots saved to {output_dir}/")
    print(f"Files: geometry_only_res{{res}}_example{{idx}}.pdf")
    print(f"       geometry_colored_res{{res}}_example{{idx}}.pdf")


def visualize_examples(config, resolutions, num_examples=3):
    """Visualize random examples across different resolutions."""
    
    # Create output directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Use the same processed directory where data was originally saved
    processed_dir = config['data_dir']
    source_dir = config['data_dir']
    
    # Load the datasets
    datasets = {}
    for res in resolutions:
        try:
            level_dir = os.path.join(processed_dir, f"level_{res}")
            datasets[res] = JEBDataset(
                config=config,
                root=os.path.join(level_dir, "train"),
                data_dir=source_dir,
                resolution=res, 
                train=True
            )
        except Exception as e:
            print(f"Error loading resolution {res}: {e}")
            return
    
    # Select random examples (same indices across resolutions)
    dataset_size = len(datasets[resolutions[0]])
    example_indices = random.sample(range(dataset_size), num_examples)
    
    # Create a grid of 3D plots
    fig = plt.figure(figsize=(16, 4 * len(resolutions)))
    
    for i, res in enumerate(resolutions):
        dataset = datasets[res]
        
        for j, idx in enumerate(example_indices):
            # JEBDataset returns a tuple (pc, xy, target)
            pc, xy, target = dataset[idx]
            
            # Debug info
            print(f"\nResolution {res}, Example {idx}:")
            print(f"  PC shape: {pc.shape if hasattr(pc, 'shape') else 'scalar'}")
            print(f"  XY shape: {xy.shape if hasattr(xy, 'shape') else 'scalar'}")
            print(f"  Target shape: {target.shape if hasattr(target, 'shape') else 'scalar'}")
            
            # Check which components match in length
            if isinstance(target, np.ndarray) or isinstance(target, torch.Tensor):
                if len(target) == len(pc):
                    print("  Using PC and target for visualization (dimensions match)")
                    points = pc
                    values = target
                elif len(target) == len(xy):
                    print("  Using XY and target for visualization (dimensions match)")
                    points = xy
                    values = target
                else:
                    print("  No matching dimensions - using PC with uniform color")
                    points = pc
                    values = np.ones(len(pc)) * (float(idx) / num_examples)
            else:
                # Default to PC with uniform colors
                points = pc
                values = np.ones(len(pc)) * (float(idx) / num_examples)
            
            # Normalize values for coloring
            if isinstance(values, torch.Tensor):
                values = values.numpy()
            
            # Ensure we're working with numpy arrays
            if isinstance(points, torch.Tensor):
                points = points.numpy()
            
            # Create a subplot
            ax = fig.add_subplot(len(resolutions), num_examples, i*num_examples + j + 1, projection='3d')
            
            # Check if values dimension matches points
            if isinstance(values, np.ndarray) and len(values) != len(points):
                print(f"Resolution {res}, Example {idx}: Value shape {values.shape} doesn't match point cloud size {points.shape}")
                # Use a fixed color scheme based on example index
                color_value = float(idx) / num_examples
                scatter = ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    c=[color_value] * len(points),  # Use uniform color 
                    cmap='viridis',
                    s=5,
                    alpha=0.8
                )
            else:
                # Mask out padded points
                valid_mask = ~(np.any(points == -1000, axis=1))
                scatter = ax.scatter(
                    points[valid_mask, 0], points[valid_mask, 1], points[valid_mask, 2],
                    c=values[valid_mask], 
                    cmap='viridis', 
                    s=5,
                    alpha=0.8
                )
            
            # Add a colorbar
            if j == num_examples - 1:
                cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
                cbar.set_label('Solution')
            
            # ax.set_title(f"Res {res}, Example {idx}")
            title_map ={0: "R", 2: "R_2", 6: "R_1"}
            ax.set_title(f"{title_map.get(res, 'Unknown')}, Example {idx}")
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            
            # Make the axes equal for better proportions
            try:
                # Get current axis limits
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                z_lim = ax.get_zlim()
                
                # Calculate center and radius
                x_center = np.mean(x_lim)
                y_center = np.mean(y_lim)
                z_center = np.mean(z_lim)
                
                # Find the furthest point from the center
                x_radius = max(abs(x_lim[0] - x_center), abs(x_lim[1] - x_center))
                y_radius = max(abs(y_lim[0] - y_center), abs(y_lim[1] - y_center))
                z_radius = max(abs(z_lim[0] - z_center), abs(z_lim[1] - z_center))
                
                # Use the largest radius for all dimensions
                max_radius = max(x_radius, y_radius, z_radius)
                
                # Set equal aspect ratio
                ax.set_xlim3d([x_center - max_radius, x_center + max_radius])
                ax.set_ylim3d([y_center - max_radius, y_center + max_radius])
                ax.set_zlim3d([z_center - max_radius, z_center + max_radius])
            except Exception as e:
                print(f"Warning: Couldn't set equal axes: {e}")
                # Continue without equal axes if there's an issue
    
    plt.tight_layout()
    
    # Save as PNG with compression (much smaller file size than PDF)
    print("Saving visualization as compressed PNG instead of PDF to reduce file size")
    plt.savefig('figures/example_point_clouds.png', bbox_inches='tight')
    plt.savefig('figures/example_point_clouds.pdf', bbox_inches='tight')
    plt.close()
    
    # Note about PDF conversion if needed
    print("For PDF conversion, you can use ImageMagick:")
    print("  convert figures/example_point_clouds.png figures/example_point_clouds.pdf")
    
    print(f"Saved visualization of {num_examples} examples across {len(resolutions)} resolutions")
    return


# Main script
if __name__ == "__main__":
    # Minimal config for testing
    config = {
        'data_dir': "examples/pdes/jeb/data/GEJetEngineBracket/",
        # 'c2f_resolutions': [0, 2, 6],
        'c2f_resolutions': [0, 1, 2, 3, 4, 5, 6],
    }

    # Create multi-resolution data
    print(f"Processing JEB data for resolutions: {config['c2f_resolutions']}")
    # create_multi_resolution_jeb_data(
    #     config=config,
    #     output_dir=config['data_dir'],
    #     resolutions=config['c2f_resolutions']
    # )

    print("Multi-resolution dataset creation completed")

    stats = collect_dataset_statistics(config, config['c2f_resolutions'])
    print("\nDataset statistics:")
    print(stats)

    print("\nVisualizing example point clouds...")
    # visualize_examples(config, config['c2f_resolutions'], num_examples=3)
    
    # Test GINOT model forward pass
    print("\nTesting GINOT model forward pass...")
    # ginot_forward_pass(config, config['c2f_resolutions'])
    
    # Fig1 - Individual component plots for paper assembly
    print("\nGenerating Fig1 component plots...")
    # fig1_plot(config, config['c2f_resolutions'], num_examples=3, output_dir='../figures/fig1_components')
