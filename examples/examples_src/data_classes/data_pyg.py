import os
import sys
import shutil
from tqdm import tqdm
import copy
import json
import traceback
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch, DataLoader
from collections import namedtuple

from mlmc_optim.dataset import MLMCInMemoryDataset

class NavierStokesDataset(MLMCInMemoryDataset):
    """Dataset for loading pre-processed Navier-Stokes data.
    
    Inherits from MLMCInMemoryDataset for dual PyG + MLMC compatibility.
    """
    def __init__(self, config, root, data_dir, mesh_level=1, mesh_prefix="mesh", Delta_t=0.25,
                 n_timesteps=50, input_window=30, output_window=20, transform=None, pre_transform=None,
                 train=True, data_split=slice(None)):
        """Initialize NavierStokesDataset.
        
        Args:
            root (str): Root directory where the dataset should be saved
            data_dir (str): Directory containing raw data
            mesh_level (int): Level of mesh to use (default 1)
            mesh_prefix (str): Prefix for mesh files (default "mesh")
            Delta_t (float): Time step size (default 0.25)
            n_timesteps (int): Number of timesteps per sample (default 50)
            input_window (int): Size of input time window (default 30)
            output_window (int): Size of output time window (default 20)
            train (bool): Whether this is training data (affects file path)
            data_split (slice): Slice of data to use, e.g. slice(0,150) for first 150 samples
            direct_loading (bool): If True, load data directly from simulation at the specified mesh level
                                   rather than projecting from the finest mesh (default False)
        """
        print(f"\nInitializing Navier-Stokes dataset from {root}")
        self.config = config
        self.data_dir = data_dir
        self.mesh_level = mesh_level
        self.Delta_t = Delta_t
        self.n_timesteps = n_timesteps
        
        # Handle both slice and list for data_split
        if isinstance(data_split, slice):
            self.N = data_split.stop - data_split.start
        else:
            # Assume data_split is a list of indices
            self.N = len(data_split)
            
        self.mesh_prefix = mesh_prefix
        self.train = train
        self.data_split = data_split  # Can be slice or list of indices
        self.input_window = input_window
        self.output_window = output_window
        self.total_timesteps = n_timesteps
        
        # Determine direct loading mode from config
        self.direct_loading = self.config.get('FPC_direct', False)
        
        # Base name used for Flow Past Cylinder dataset directories
        self.dataset_suffix = "FlowPastCylinder"
        direct_suffix = "_direct" if self.direct_loading else ""
        self.dataset_suffix = f"{self.dataset_suffix}{direct_suffix}_{self.mesh_level}"

        assert input_window + output_window <= n_timesteps, \
            f"Total window size {input_window + output_window} must be <= total timesteps {n_timesteps}"
        

        super().__init__(root, transform, pre_transform)
        # Load the tuple as before
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        mode = "train" if self.train else "test"
        return [f'data_{mode}_level_{self.mesh_level}_N{self.N}.pt']

    def process(self):
        """NavierStokesDataset now assumes preprocessed .pt files are provided.

        This method is intentionally minimal: it raises a clear error if the
        expected processed file is missing, rather than attempting to build it
        via Firedrake/Gmsh.
        """
        processed_path = self.processed_paths[0]
        raise RuntimeError(
            f"NavierStokesDataset.process is disabled. Provide a preprocessed "
            f"file at '{processed_path}' (typically generated offline) before "
            f"instantiating this dataset."
        )



    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        data = super().__getitem__(idx)
        
        # Split velocity into input/output windows
        velocity_sequence = data.x  # [timesteps, n_nodes, 2]
        pressure_sequence = data.pressure  # [timesteps, n_nodes]
        
        # Original version
        # data.x = velocity_sequence[:self.input_window]  # [input_window, n_nodes, 2]
        # data.y = velocity_sequence[self.input_window:self.input_window + self.output_window]  # [output_window, n_nodes, 2]
        
        # New version - transpose to [n_nodes, timesteps, 2] for proper batching
        data.x = velocity_sequence[:self.input_window].transpose(0, 1)  # [n_nodes, input_window, 2]
        data.y = velocity_sequence[self.input_window:self.input_window + self.output_window].transpose(0, 1)  # [n_nodes, output_window, 2]
        data.pressure = pressure_sequence.transpose(0, 1)  # [n_nodes, timesteps]
        
        return data

    def get_items(self, indices):
        """Get multiple items from the dataset and split into input/targets.
        
        Args:
            indices (list): List of indices to get
            
        Returns:
            tuple: (batch_data, batch_targets) where:
                - batch_data is a Data object with the input sequences
                - batch_targets is a tensor of shape [batch_size, output_window, n_nodes, 2]
        """
        # Get the items
        batch_list = [self[i] for i in indices]
        
        # Collate the batch
        batch_data = Batch.from_data_list(batch_list)

        # Get targets (output window velocities)
        batch_targets = batch_data.y  # [batch_size, output_window, n_nodes, 2]
        
        return batch_data, batch_targets

    def __len__(self):
        """Return the length of the dataset."""
        return self.N  # Number of source terms in data_split
        
    def to_device(self, device):
        """Move dataset to specified device."""
        # Move main data tensors
        self.data = self.data.to(device)
        self.pos = self.pos.to(device)
        self.edge_index = self.edge_index.to(device)

        # Move boundary masks
        self.inlet_mask = self.inlet_mask.to(device)
        self.sides_mask = self.sides_mask.to(device)
        self.obstacle_mask = self.obstacle_mask.to(device)
        self.outlet_mask = self.outlet_mask.to(device)
        
        # Move slices if they exist
        if hasattr(self, 'slices') and self.slices is not None:
            self.slices = {k: v.to(device) for k, v in self.slices.items()}
        
        return self
        
    def load_batch_indices_to_gpu(self, indices, device):
        """Move all main tensors and masks to GPU, matching to_device and get_items conventions."""
        self.data[indices] = self.data[indices].to(device)
        self.pos[indices] = self.pos[indices].to(device)
        self.edge_index = self.edge_index.to(device)
        # Move boundary masks to GPU
        self.inlet_mask = self.inlet_mask.to(device)
        self.sides_mask = self.sides_mask.to(device)
        self.obstacle_mask = self.obstacle_mask.to(device)
        self.outlet_mask = self.outlet_mask.to(device)
        
        print(f"NavierStokesDataset: Finished loading data to {device}")
        
    def unload_from_gpu(self, indices=None):
        """Explicitly move data from GPU back to CPU to free memory.
        Selectively unloads specified indices when possible.
        
        Args:
            indices (list, optional): Indices to unload. If None, unloads nothing.
        """
        if not torch.cuda.is_available() or not indices:
            return
        
        self.data = self.data.cpu()
        self.pos = self.pos.cpu()
        self.edge_index = self.edge_index.cpu()
        self.inlet_mask = self.inlet_mask.cpu()
        self.sides_mask = self.sides_mask.cpu()
        self.obstacle_mask = self.obstacle_mask.cpu()
        self.outlet_mask = self.outlet_mask.cpu()
        if hasattr(self, 'slices') and self.slices is not None:
            self.slices = {k: v.cpu() for k, v in self.slices.items()}


def plot_solution(data, level, save_dir):
    """Plot velocity field and pressure for a given solution."""
    pos = data.pos.numpy()
    vel = data.x.numpy()  # [n_nodes, time_steps, 2]
    pressure = data.pressure.numpy()  # [n_nodes, time_steps]
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot velocity field as quiver plot
    vel_t0 = vel[:, -1, :]  # [n_nodes, 2] - get last timestep
    print(f"Velocity at t=0 shape: {vel_t0.shape}, range: [{vel_t0.min():.3f}, {vel_t0.max():.3f}]")
    print(f"Pressure at t=0 shape: {pressure[:, -1].shape}")  # Debug print
    
    q = ax1.quiver(pos[:, 0], pos[:, 1], vel_t0[:, 0], vel_t0[:, 1], scale=50)
        
    ax1.set_title(f'Velocity Field (Level {level})', fontsize=14)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_aspect('equal')
    
    # Plot pressure field as scatter plot
    pressure_t0 = pressure[:, -1]  # [n_nodes] - get last timestep
    scatter = ax2.scatter(pos[:, 0], pos[:, 1], c=pressure_t0, s=10, cmap='coolwarm')
    plt.colorbar(scatter, ax=ax2)
        
    ax2.set_title(f'Pressure Field (Level {level})', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'solution_level_{level}_t_0.00_source_0.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def _build_fpc_config(dataset_name, base_res, FPC_direct):
    """Construct minimal config dict for FPC1/FPC2."""
    return {
        'dataset': dataset_name,
        'base_res': base_res,
        'n_timesteps': 50,
        'input_window': 30,
        'output_window': 20,
        'Delta_t': 0.25,
        'FPC_direct': FPC_direct,
    }

def run_generate_mode(data_dir, dataset_name, base_res, FPC_direct, train_indices, test_indices, levels=None):
    """Generate processed .pt datasets (train + test) for the given levels."""
    config = _build_fpc_config(dataset_name, base_res, FPC_direct)
    N_train = len(train_indices)
    N_test = len(test_indices)

    source_dir = data_dir
    direct_suffix = "_direct" if FPC_direct else ""
    processed_dir = os.path.join(data_dir, f"{dataset_name}{direct_suffix}")
    os.makedirs(processed_dir, exist_ok=True)

    if levels is None:
        levels = [base_res]

    for level in levels:
        print(f"\n[generate] Processing train data for level {level}...")
        level_root_train = os.path.join(processed_dir, f"level_{level}_N{N_train}_train")
        NavierStokesDataset(
            config=config,
            root=level_root_train,
            data_dir=source_dir,
            mesh_level=level,
            mesh_prefix='mesh',
            Delta_t=config['Delta_t'],
            n_timesteps=config['n_timesteps'],
            input_window=config['input_window'],
            output_window=config['output_window'],
            train=True,
            data_split=train_indices,
        )

        print(f"[generate] Processing test data for level {level}...")
        level_root_test = os.path.join(processed_dir, f"level_{level}_N{N_test}_test")
        NavierStokesDataset(
            config=config,
            root=level_root_test,
            data_dir=source_dir,
            mesh_level=level,
            mesh_prefix='mesh',
            Delta_t=config['Delta_t'],
            n_timesteps=config['n_timesteps'],
            input_window=config['input_window'],
            output_window=config['output_window'],
            train=False,
            data_split=test_indices,
        )


def run_data_shapes_mode(data_dir, dataset_name, base_res, FPC_direct, train_indices):
    """Print tensor shapes for debugging."""
    config = _build_fpc_config(dataset_name, base_res, FPC_direct)
    N_train = len(train_indices)

    source_dir = data_dir
    direct_suffix = "_direct" if FPC_direct else ""
    processed_dir = os.path.join(data_dir, f"{dataset_name}{direct_suffix}")

    mesh_level = base_res
    print(f"\n[data_shapes] Processing mesh level {mesh_level}...")
    level_root_dir = os.path.join(processed_dir, f"level_{mesh_level}_N{N_train}_train")
    print(f"[data_shapes] Data directory: {data_dir}, root_dir: {level_root_dir}")

    ds = NavierStokesDataset(
        config=config,
        root=level_root_dir,
        data_dir=source_dir,
        mesh_level=mesh_level,
        mesh_prefix='mesh',
        Delta_t=config['Delta_t'],
        n_timesteps=config['n_timesteps'],
        input_window=config['input_window'],
        output_window=config['output_window'],
        train=True,
        data_split=train_indices,
    )

    for i in range(len(ds)):
        sample = ds[i]
        print(f"\n[data_shapes] Sample {i}")
        print(
            f"X: {sample.x.shape}, Y: {sample.y.shape}, "
            f"Pressure: {sample.pressure.shape}, Pos: {sample.pos.shape}, "
            f"Edge Index: {sample.edge_index.shape}"
        )


def run_plot_data_mode(data_dir, dataset_name, base_res, FPC_direct, train_indices, sample_idx=0):
    """Quick visualisation of a single sample."""
    config = _build_fpc_config(dataset_name, base_res, FPC_direct)
    N_train = len(train_indices)

    source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "navier_stokes_data")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    direct_suffix = "_direct" if FPC_direct else ""
    processed_dir = os.path.join(data_dir, f"{dataset_name}{direct_suffix}")
    vis_dir = os.path.join(processed_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    level_root_dir = os.path.join(processed_dir, f"level_{base_res}_N{N_train}_train")
    ds = NavierStokesDataset(
        config=config,
        root=level_root_dir,
        data_dir=source_dir,
        mesh_level=base_res,
        mesh_prefix="mesh",
        Delta_t=config['Delta_t'],
        n_timesteps=config['n_timesteps'],
        input_window=config['input_window'],
        output_window=config['output_window'],
        train=True,
        data_split=train_indices,
    )

    if len(ds) == 0:
        print("[plot_data] Dataset empty.")
        return

    idx = min(sample_idx, len(ds) - 1)
    plot_solution(ds[idx], base_res, vis_dir)



if __name__ == "__main__":
    # Modes:
    #   'generate', 'data_shapes', 'analyze',
    #   'plot_data', 'test_forward_pass'
    mode = "data_shapes"

    dataset_name = "FlowPastCylinder"
    base_res = 4
    FPC_direct = True

    # Single source of truth for data root
    data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "pdes",
    "flow_past_cylinder",
    "data",
    )

    train_indices = [i for i in range(150)]
    test_indices = [i for i in range(150, 200)]

    if mode == "generate":
        run_generate_mode(data_dir, dataset_name, base_res, FPC_direct, train_indices, test_indices)
    elif mode == "data_shapes":
        run_data_shapes_mode(data_dir, dataset_name, base_res, FPC_direct, train_indices)
    elif mode == "plot_data":
        run_plot_data_mode(data_dir, dataset_name, base_res, FPC_direct, train_indices, sample_idx=0)