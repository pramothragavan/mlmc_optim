"""Configuration utilities."""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
from typing import Dict, Any
from examples.examples_src.utils.utils import set_seed, get_device


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_int_list(v):
    """Parse integer list from string or return as-is if already int."""
    if isinstance(v, int):
        return v
    # If it's a string that looks like a list, return it as-is for later eval
    if isinstance(v, str) and v.startswith('['):
        return v
    # Otherwise try to parse as int
    return int(v)


def nullable_float(value):
    """Convert string to float or None (for sweeps that pass 'None'/'null')."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null"):
        return None
    return float(value)

def nullable_int(value):
    """Convert string to int or None (for wandb sweeps that pass 'None' as string)"""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("none", "null"):
        return None
    return int(value)

def _build_parser():


    """Build the base argument parser (no parsing yet)."""
    parser = argparse.ArgumentParser(
        description='Train neural operator with MLMC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (optional when running sweeps)
    parser.add_argument('--config', type=str, required=False, default=None,
                        help='Path to config file (e.g., examples/pdes/darcy_flow/darcy_fno.yaml)')

    # Model
    parser.add_argument('--model', type=str, default='fno', choices=['parabolic', 'cnn', 'fno', 'fno3d', 'mp_pde', 'ginot'], help='Type of model to use')
    parser.add_argument('--in_channels', type=int, help='Input channels')
    parser.add_argument('--out_channels', type=int, help='Output channels')
    parser.add_argument('--num_layers', type=int, help='Number of layers')
    # FNO/CNN specific arguments
    parser.add_argument('--rank', type=int, default=2, help='Rank for factored operator')
    parser.add_argument('--input_size', type=int, help='input image size')
    parser.add_argument('--hidden_dim', type=int, default=16, help='input image size')
    parser.add_argument('--stride1', type=int, help='stride for first conv layer')
    parser.add_argument('--stride2', type=int, help='stride for second conv layer')
    parser.add_argument('--base_channels', type=int, help='base number of channels for UNet')
    parser.add_argument('--max_pool', action='store_true', help='Whether to use max pooling in the CNN')
    parser.add_argument('--layer_dims', nargs='+', default=[16, 16, 16, 16], help='List of hidden dimensions for each layer (e.g., --layer_dims 32 64 128)')
    parser.add_argument('--fno_modes', type=int, default=8, help='Number of Fourier modes for FNO')
    parser.add_argument('--fno_width', type=int, default=32, help='Width of FNO layers')

    # parabolic CNN specific arguments
    parser.add_argument('--operator_type', type=str, default='diagonal', choices=['diagonal', 'factored'], help='Type of elliptic operator for parabolic convolutions')
    parser.add_argument('--advection', type=bool, default=False, help='To add advection term to parabollic CNN')
    # MP-PDE specific arguments
    parser.add_argument('--L', type=float, default=1.0, help='Domain length')
    parser.add_argument('--tmax', type=int, default=50, help='Maximum simulation time')
    parser.add_argument('--dt', type=float, default=0.25, help='Time step size')
    parser.add_argument('--mp_pde_input_window', type=int, default=30, help='Input window size for MP-PDE model')
    parser.add_argument('--mp_pde_output_window', type=int, default=20, help='Size of output time window')
    parser.add_argument('--mp_pde_hidden_dim', type=int, default=128, help='Hidden dimension in MP-PDE')
    parser.add_argument('--mp_pde_layers', type=int, default=6, help='Number of layers in MP-PDE')
    parser.add_argument('--mp_pde_autoregressive', type=bool, default=False, help='Whether to use autoregressive mode for MP-PDE')
    parser.add_argument('--mp_pde_eq_vars', default=None, help='Equation variables for MP-PDE (as JSON string)')
    # GINOT model hyperparameters
    parser.add_argument('--ginot_in_dim', type=int, default=3, help='Input dimension for GINOT branch encoder')
    parser.add_argument('--ginot_global_pooling', type=bool, default=True, help='Global pooling for GINOT branch encoder')
    parser.add_argument('--ginot_latent_dim', type=int, default=64, help='Latent dimension for GINOT branch encoder')
    parser.add_argument('--ginot_embed_dim', type=int, default=64, help='Embedding dimension for GINOT branch/trunk')
    parser.add_argument('--ginot_num_layers', type=int, default=4, help='Number of layers for GINOT branch encoder')
    parser.add_argument('--ginot_num_heads', type=int, default=4, help='Number of attention heads for GINOT')
    parser.add_argument('--ginot_padding_value', type=float, default=-1000, help='Padding value for GINOT')
    parser.add_argument('--ginot_dropout', type=float, default=0.0, help='Dropout rate for GINOT')
    parser.add_argument('--ginot_cross_attn_layers', type=int, default=4, help='Number of cross attention layers for GINOT trunk')
    parser.add_argument('--ginot_in_channels', type=int, default=3, help='Input channels for GINOT trunk')
    parser.add_argument('--ginot_out_channels', type=int, default=1, help='Output channels for GINOT trunk')
    parser.add_argument('--ginot_emd_version', type=str, default='nerf', help='Embedding version for GINOT trunk')
    parser.add_argument('--padding_value', type=float, default=-1000, help='Padding value for GINOT trunk')
    parser.add_argument('--ginot_coarse_pc', type=bool, default=False, help='Whether to use coarse point clouds for GINOT')
    parser.add_argument('--ginot_both_dim', type=int, default=64, help='Embedding and latent dimension for GINOT')

    # Data
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--dataset', type=str, default='darcy', choices=['darcy', 'navier_stokes', 'adr', 'FlowPastCylinder', 'jeb'], help='dataset')
    parser.add_argument('--c2f_resolutions', type=parse_int_list, nargs='+', help='Coarse-to-fine resolutions list')
    parser.add_argument('--base_res', type=int, help='Base resolution')
    parser.add_argument('--override_res', type=int, default=None, help='Override resolution for single-resolution baseline training')
    parser.add_argument('--add_coords', type=str2bool, default=False, help='Add coordinate channels')
    parser.add_argument('--train_subset', type=float, default=1., help='Train subset fraction')
    parser.add_argument('--test_subset', type=float, default=1., help='Test subset fraction')
    parser.add_argument('--normalize', type=str2bool, default=True, help='Whether to normalize the data')
    parser.add_argument('--use_grads', default=False, type=str2bool, help='Use gradient data')
    parser.add_argument('--mixed_res_training', default=False, type=str2bool, help='Use mixed resolution training')
    parser.add_argument('--mixed_res_mode', type=str, default='uniform', choices=['uniform', 'mlmc_schedule'], help='Mixed-res sampling mode')
    
    # Time series parameters (for FNO3D on temporal data)
    parser.add_argument('--T', type=int, help='Total number of timesteps')
    parser.add_argument('--T_in', type=int, help='Number of input timesteps')

    # Memory management
    parser.add_argument('--load_in_memory', default=True, type=str2bool, help='Load entire dataset into RAM at startup (faster but requires more memory)')
    parser.add_argument('--device', type=str, choices=['gpu', 'cuda', 'mps', 'cpu'], help='Device to run training on')
    parser.add_argument('--pin_memory', default=False, type=str2bool, help='Pin CPU memory for faster data transfer to GPU (requires extra RAM)')
    parser.add_argument('--load_gpu', default=True, type=str2bool, help='Load entire dataset into GPU memory at startup (fastest but requires large GPU memory)')
    parser.add_argument('--load_gpu_epoch', default=False, type=str2bool, help='Load only current epoch batches to GPU (slower but uses less GPU memory)')

    # FlowPastCylinder specific parameters
    parser.add_argument('--mp_pde_inlet', type=float, default=1.0, help='Inlet boundary condition (u=0,0)')
    parser.add_argument('--mp_pde_sides', type=float, default=1.0, help='Sides boundary condition (u=1,0)')
    parser.add_argument('--mp_pde_obstacle', type=float, default=1.0, help='Obstacle boundary condition (u=0,0)')
    parser.add_argument('--mp_pde_outlet', type=float, default=1.0, help='Outlet boundary condition (p=0)')

    # MLMC parameters
    parser.add_argument('--mlmc_pairing', type=str, default='hierarchy', choices=['hierarchy', 'random'], help='Pairing strategy for MLMC')
    parser.add_argument('--mlmc_sampling_style', type=str, default=None, choices=['geom_prog', 'prescribed', 'random', 'random_disjoint', 'mixed'], help='MLMC sampling style')
    # for MLMC sampling style: prescribed
    parser.add_argument('--mlmc_samples_per_level', nargs='+', default=[1024], help='Prescribed number of samples per resolution level')
    parser.add_argument('--total_samples', type=int, default=1024, help='Total dataset size / sample pool size')
    parser.add_argument('--mlmc_batch_sizes', nargs='+', default=None, help='Batch sizes for each level')
    # for MLMC sampling style: geom_prog
    parser.add_argument('--mlmc_data_size_multiplier', type=float, default=2, help='Factor k where each finer level sees 1/k of the data of the next coarser level')
    parser.add_argument('--mlmc_batch_style', type=str, default='geom_prog_batch', choices=['geom_prog_batch','fixed', 'prescribed_batch'], help='Batch size style for MLMC')
    parser.add_argument('--mlmc_batch_size_multiplier', type=float, default=2, help='Factor where each finer level uses 1/k of the batch size')
    parser.add_argument('--mlmc_both_multiplier', type=float, default=None, help='Factor k for both batch size and data size (convenience param that sets both multipliers)')
    parser.add_argument('--mlmc_max_level', type=int, default=None, help='Maximum MLMC level (overrides instead of all resolutions)')
    parser.add_argument('--mlmc_deterministic_batching', type=str2bool, default=False, help='Use deterministic (sequential) batching instead of random sampling')
    parser.add_argument('--mlmc_hierarchy_cache', type=str2bool, default=False, help='Whether to cache the hierarchy for MLMC optimisation')

    # Training
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--loss_type', type=str, help='Loss type')
    parser.add_argument('--loss_reduction', type=str, help='Loss reduction method', choices=['mean', 'sum'])
    parser.add_argument('--h1_alpha', type=float, default=0.01, help='H1 loss gradient penalty weight (default: 0.01)')
    parser.add_argument('--h1_base_loss', type=str, default='mse', choices=['mse', 'l2', 'lp'], help='H1 base loss function (default: mse)')
    parser.add_argument('--pinn_nu', type=float, default=0.001, help='PINN viscosity (1/Reynolds, default: 0.001 for R=1000)')
    parser.add_argument('--pinn_lambda', type=float, default=1.0, help='PINN PDE residual weight (default: 1.0)')
    parser.add_argument('--pinn_data_loss', type=str, default='mse', choices=['mse', 'l2', 'lp'], help='PINN data loss type (default: mse)')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')

    # Optimizer and Scheduler
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--optimizer', type=str, help='Optimizer')
    parser.add_argument('--use_scheduler', type=str2bool, default=False, help='Use LR scheduler')
    parser.add_argument('--scheduler', type=str, help='Scheduler type')
    parser.add_argument('--step_size', type=int, help='Scheduler step size')
    parser.add_argument('--gamma', type=float, help='Scheduler gamma')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Scheduler lr_decay')

    # Logging
    parser.add_argument('--eval_every', type=int, help='Evaluation frequency')
    parser.add_argument('--eval_plot_every', type=int, help='Plotting frequency (default: same as eval_every)')
    parser.add_argument('--save_model', type=str2bool, default=False, help='Save model locally')
    parser.add_argument('--save_wandb_artifact', type=str2bool, default=False, help='Save model to wandb artifacts')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')

    parser.add_argument('--eval_grad_every', type=nullable_int, default=None, help='evaluate grad every N epochs')
    parser.add_argument('--eval_grad_rand', type=str2bool, default=False, help='use random sampling during gradient analysis')
    parser.add_argument('--eval_grad_mode', type=str, default='raw', choices=['raw', 'scaled'], help='Gradient analysis mode')
    parser.add_argument('--eval_grad_batches', type=nullable_int, default=None, help='Number of batches to use for gradient analysis. None means use all batches.')


    # Wandb parameters
    parser.add_argument('--use_wandb', help='Use Weights & Biases logging', action='store_true')
    parser.add_argument('--wandb_sweep', action='store_true', help="flag if sweeping, ignore base config")
    parser.add_argument('--wandb_entity', default="mlmc-optim", type=str)
    parser.add_argument('--wandb_project', default="mlmc-optim", type=str)
    parser.add_argument('--wandb_group', default="model_dev", type=str, choices=["model_dev","model_comparison"])
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_exp_idx', type=int, default=0, help="experiment index")

    return parser


def parse_args():


    """Parse command line arguments with comprehensive coverage."""
    parser = _build_parser()
    return parser.parse_args()


def get_config() -> Dict[str, Any]:
    """Get complete configuration with proper precedence.
    
    Precedence (highest to lowest):
    1. Explicit command-line arguments (those differing from argparse defaults)
    2. YAML config file
    3. Argparse defaults
    
    Returns:
        Flattened configuration dictionary with all settings
    """
    parser = _build_parser()

    # Argparse defaults only (no CLI)
    defaults_ns = parser.parse_args([])
    defaults = vars(defaults_ns)

    # Actual CLI (defaults + any overrides)
    args = parser.parse_args()
    cli = vars(args)

    # Start from pure argparse defaults
    config: Dict[str, Any] = defaults.copy()

    # Override with YAML config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        yaml_config = load_config(str(config_path))
        config.update(yaml_config)

    # Finally override with explicitly-set CLI arguments
    for k, v in cli.items():
        if k == 'config':
            continue
        if v != defaults.get(k):
            config[k] = v

    # Handle list arguments that may come as strings from sweeps
    list_args = ['c2f_resolutions']
    for arg in list_args:
        if arg in config and config[arg] is not None:
            # If it's a string that looks like a list, eval it
            if isinstance(config[arg], str) and config[arg].startswith('['):
                config[arg] = eval(config[arg])
            # If it's a list with a single string element, eval it
            elif isinstance(config[arg], list) and len(config[arg]) > 0:
                if isinstance(config[arg][0], str) and config[arg][0].startswith('['):
                    config[arg] = eval(config[arg][0])
    
    # Handle mlmc_both_multiplier convenience parameter
    if config.get('mlmc_both_multiplier') is not None:
        config['mlmc_batch_size_multiplier'] = config['mlmc_both_multiplier']
        config['mlmc_data_size_multiplier'] = config['mlmc_both_multiplier']
    
    # Handle ginot_both_dim convenience parameter
    if config.get('ginot_both_dim'):
        config['ginot_embed_dim'] = config['ginot_both_dim']
        config['ginot_latent_dim'] = config['ginot_both_dim']
    
    # Handle MP-PDE FlowPastCylinder boundary conditions
    if config.get('model') == 'mp_pde' and config.get('dataset') == 'FlowPastCylinder':
        config['mp_pde_eq_vars'] = {'inlet': 1, 'sides': 1, 'obstacle': 0, 'outlet': 1}
    
    # Trim c2f_resolutions to match max_level if needed
    if 'c2f_resolutions' in config and config['c2f_resolutions'] is not None:
        max_level = config.get('mlmc_max_level')
        if max_level is not None and max_level != 1 and len(config['c2f_resolutions']) != max_level:
            # Take the finest max_level resolutions (from the end)
            config['c2f_resolutions'] = config['c2f_resolutions'][-max_level:]
            print(f"Trimming c2f_resolutions to match mlmc_max_level: {config['c2f_resolutions']}")
        
        # Set base_res to finest resolution if not already set
        if 'base_res' not in config or config['base_res'] is None:
            config['base_res'] = config['c2f_resolutions'][-1]
            print(f"Setting base_res to finest resolution: {config['base_res']}")
    
    # Auto-detect device if not set
    config['device'] = get_device(config)
    if config['model'] == 'fno3d' and config['device'] == 'mps':
        config['device'] = 'cpu'
        print("FNO3d model is not supported on MPS. Switching to CPU device.")

    if config.get('load_gpu'):
        if config.get('pin_memory'):
            print("Disabling pin_memory because load_gpu is True (pinning only applies to CPU tensors).")
        config['pin_memory'] = False

    # Set seed
    if config.get('seed'):
        set_seed(config['seed'])

    # Print configuration
    print("=" * 80)
    if config.get('mlmc_sampling_style'):
        print("MLMC Optim - Neural Operator Training")
    else:
        print("Neural Operator Training")
    print("=" * 80)
    print(f"Dataset: {config.get('dataset', 'N/A')}")
    print(f"Model: {config.get('model', 'N/A')}")
    print(f"Device: {config.get('device', 'cpu')}")
    print(f"Epochs: {config.get('epochs', 'N/A')}")
    print(f"Learning rate: {config.get('lr', 'N/A')}")
    if config.get('override_res'):
        print(f"Training resolution: {config.get('override_res')}")
        print(f"Test resolution: {config.get('base_res')}")
    elif config.get('c2f_resolutions'):
        print(f"C2F Resolutions: {config.get('c2f_resolutions')}")
        if config.get('sample_sizes'):
            print(f"Sample sizes: {config.get('sample_sizes')}")
    print("=" * 80)
    
    return config