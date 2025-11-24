
import os
from typing import Dict, Tuple, Optional, Callable, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader


from examples.examples_src.utils.metrics import LpLoss, GINOTLoss
from examples.examples_src.data_classes.data_fno import MultiResolutionDataset
from examples.examples_src.data_classes.data_jeb import pad_collate_fn


def get_criterion(config, for_eval: bool = False):
    """Get loss criterion based on config.

    Returns the loss criterion module based on the provided configuration.

    Args:
        config: Configuration dictionary with 'loss_type' key
        for_eval: Whether the criterion is for evaluation (affects reduction)

    Returns:
        Loss criterion module
    """

    if for_eval:
        reduction = config['loss_reduction']
    else:
        if config.get('mlmc_hierarchy_cache', False):
            reduction = 'none'
        elif config['loss_reduction'] is None:
            reduction = 'none'
        else:
            reduction = config['loss_reduction']

    if config['loss_type'] == 'lp':
        criterion = LpLoss(reduction=reduction)
    elif config['loss_type'] == 'mse':
        return nn.MSELoss(reduction=reduction)
    elif config['loss_type'] == 'ginot':
        # GiN OT loss with padding mask; honor reduction from config
        criterion = GINOTLoss(padding_value=config['padding_value'], reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {config['loss_type']}")

    print(f"✓ Criterion: {config.get('criterion', 'mse').upper()}(reduction={reduction})")

    return criterion


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on config.

    Args:
        model: Neural network model
        config: Configuration dictionary

    Returns:
        Optimizer
    """
    optimizer_type = config['optimizer'].lower()
    lr = config['lr']
    weight_decay = config['weight_decay']

    if optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=config['momentum'], weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def setup_optimizers(model, config):
    # Print gradient accumulation info if using more than 1 step
    if config['gradient_accumulation_steps'] > 1:
        print(f"Using gradient accumulation with {config['gradient_accumulation_steps']} steps")
        print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")

    # Always use a single optimizer and (optionally) a single scheduler
    optimizer = get_optimizer(model, config)

    scheduler = None
    if config['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )
    return optimizer, scheduler


def get_denormalizer(train_datasets: Dict[int, MultiResolutionDataset],
                     config: Dict[str, Any]) -> Optional[Callable]:
    """Create denormalizer function if normalization is enabled.

    Uses statistics from the finest resolution training dataset.

    Args:
        train_datasets: Dictionary of training datasets keyed by resolution
        config: Configuration dictionary

    Returns:
        Denormalizer function or None if normalization is disabled
    """
    if not config['normalize']:
        return None

    # Get stats from finest resolution
    finest_res = config['c2f_resolutions'][-1]
    device = config['device']

    train_mean = train_datasets[finest_res].output_mean.to(device)
    train_std = train_datasets[finest_res].output_std.to(device)

    def denormalizer(x):
        """Denormalize: x * std + mean"""
        return x * (train_std + 1e-5) + train_mean

    return denormalizer



def get_data_loader(dataset, config, shuffle=False):
    """Create appropriate DataLoader based on dataset type.
    
    Args:
        dataset: Dataset to create loader for
        config: Configuration dictionary
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader (PyTorch or PyG depending on dataset type)
    """
    
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 0)
    pin_memory = config.get('pin_memory', False)
    
    # PyG datasets (graph-based)
    if config['dataset'] == 'FlowPastCylinder':
        return PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    # JEB dataset (variable-sized point clouds)
    elif config['dataset'] == 'jeb':
        # Collate into generic (data, targets):
        #   data    = (pc_padded, xyt_padded)
        #   targets = S_padded
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=pad_collate_fn,
        )
    
    # Standard tensor datasets (Darcy, ADR, Navier-Stokes)
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )


def save_model(model, optimizer, epoch, config, metrics, is_best=False):
    """Save model checkpoint with wandb artifact support
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        config: Config dictionary with model parameters
        metrics: Dictionary of metrics to save
        is_best: If True, this is the best model so far
    
    Returns:
        checkpoint_path: Path where checkpoint was saved
    """
    # Create checkpoint directory
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics
    }
    
    checkpoint_path = None
    
    # Save locally if requested
    if config.get('save_model', False):
        checkpoint_name = config.get('wandb_project', 'model')
        checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_name}_best.pth' if is_best else f'{checkpoint_name}_last.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved {'best' if is_best else 'last'} checkpoint to {checkpoint_path} (epoch {epoch})")
    
    # Save to wandb artifact if requested
    if config.get('save_wandb_artifact', False) and config.get('use_wandb', False):
        import wandb
        
        # Need local file for artifact
        if checkpoint_path is None:
            checkpoint_name = config.get('wandb_project', 'model')
            checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_name}_temp.pth')
            torch.save(checkpoint, checkpoint_path)
        
        artifact = wandb.Artifact(
            name=f"{config.get('wandb_project', 'model')}-{wandb.run.id}",
            type='model',
            description=f"Model checkpoint at epoch {epoch}",
            metadata={'epoch': epoch, 'is_best': is_best, **metrics}
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact, aliases=['best' if is_best else 'last'])
        print(f"Logged {'best' if is_best else 'last'} model to wandb artifacts (epoch {epoch})")
    
    return checkpoint_path
