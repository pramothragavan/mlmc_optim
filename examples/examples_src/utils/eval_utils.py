"""Evaluation utilities for MLMC training."""

from torch.utils.data import Dataset
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn

def evaluate_model(model, test_loader, criterion, device, denormalizer=None, config=None):
    """Evaluate model on test dataset.
    
    Matches training loss calculation exactly to maintain loss magnitude consistency.

    Args:
        model: Neural network model
        test_loader: Test DataLoader
        criterion: Loss criterion
        device: Device to run on
        denormalizer: Optional denormalizer function
        config: Configuration dictionary

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    test_loss = 0.0
    total_rel_l2 = 0.0
    n_test = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle different dataset types
            if config['dataset'] == 'FlowPastCylinder':
                # PyG datasets: batch is a single Data/Batch object
                data = batch.to(device)
                target = data.y
                data_shape = data.batch_size
            else:
                # Standard datasets (including JEB baseline): batch is (data, target)
                data, target = batch
                # data may be a tensor or a tuple (e.g., JEB: (pc_padded, xyt_padded))
                if isinstance(data, tuple):
                    data = tuple(d.to(device) for d in data)
                else:
                    data = data.to(device)
                target = target.to(device)
                data_shape = target.shape[0]
            
            # Forward pass
            output = model(data)
            
            # Apply denormalizer if provided
            if denormalizer is not None:
                output = denormalizer(output)
                target = denormalizer(target)
            
            loss = criterion(output, target)

            if config['loss_reduction'] == 'mean':
                test_loss += loss.item() * data_shape
            else:
                test_loss += loss.item()
            
            # Compute relative L2 error
            diff_norm = torch.norm(output.reshape(data_shape, -1) - target.reshape(data_shape, -1), dim=1)
            target_norm = torch.norm(target.reshape(data_shape, -1), dim=1)
            rel_l2 = (diff_norm / (target_norm + 1e-8)).sum().item()
            
            total_rel_l2 += rel_l2
            n_test += data_shape
            n_batches += 1
    
    # Compute averages matching training behavior
    # Both mean and sum reduction report per-sample error
    avg_loss = test_loss / n_test if n_test > 0 else 0.0
    
    avg_rel_l2 = total_rel_l2 / n_test if n_test > 0 else 0.0
    
    model.train()
    
    return {
        'test_loss': avg_loss,
        'rel_l2_error': avg_rel_l2,
        'n_samples': n_test
    }


def get_plot_fn(config: Dict[str, Any]) -> Optional[Callable]:
    """Create plotting function based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Plotting function or None if plotting disabled
    """
    if not config.get('eval_plot_every'):
        return None
    
    # Check if JEB dataset
    if config['dataset'] == 'jeb':
        # Import JEB plotting function
        from examples.examples_src.data_classes.data_jeb import plot_prediction_jeb
        
        def plot_predictions_jeb(model, dataset, device, epoch, denormalizer=None):
            """Generate JEB prediction plots.
            
            Args:
                model: Neural network model
                dataset: JEB dataset to plot from
                device: Device to run on
                epoch: Current epoch number
                denormalizer: Optional denormalizer function (unused for JEB)
                
            Returns:
                matplotlib figure
            """
            from torch.utils.data import DataLoader
            
            # Create a small dataloader for plotting
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            fig = plot_prediction_jeb(
                config=config,
                model=model,
                data_loader=data_loader,
                epoch=epoch,
                device=device,
                N=4,
                show=False,
                save_path=None
            )
            return fig
        
        return plot_predictions_jeb
    else:
        # Standard plotting for regular datasets
        def plot_predictions(model, dataset, device, epoch, denormalizer=None):
            """Generate prediction plots.
            
            Args:
                model: Neural network model
                dataset: Dataset to plot from
                device: Device to run on
                epoch: Current epoch number
                denormalizer: Optional denormalizer function
                
            Returns:
                matplotlib figure
            """
            # Use existing plot_prediction function
            fig = plot_prediction(
                model=model,
                dataset=dataset,
                device=device,
                n_samples=4,
                save_path=None,
                title=f"Epoch {epoch+1} - Predictions"
            )
            return fig
        
        return plot_predictions


"""Plotting utilities for visualizing predictions."""

def plot_prediction(
        model,
        dataset,
        device,
        n_samples=4,
        save_path=None,
        title="Predictions"
):
    """Plot predictions vs targets for visualization.
    
    For 3D spatiotemporal data (Navier-Stokes), plots 3 timesteps:
    start (t=0), middle (t=T//2), and end (t=T-1).

    Args:
        model: Trained model
        dataset: Dataset to sample from
        device: Device for computation
        n_samples: Number of samples to plot
        save_path: Path to save figure (optional)
        title: Figure title
    """
    model.eval()

    # Sample random indices
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    # Get first sample to check if data is temporal
    with torch.no_grad():
        sample_data, sample_target = dataset[indices[0]]
        is_temporal = len(sample_target.shape) == 3  # [H, W, T]
    
    # Set up figure layout based on data type
    if is_temporal:
        # For temporal data: rows = samples, cols = 3 timesteps × 2 (target, pred) = 6
        fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
    else:
        # For static data: rows = samples, cols = 3 (input, target, pred)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            data, target = dataset[idx]
            data = data.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            # Predict
            pred = model(data)

            # Move to CPU for plotting
            data = data.cpu().squeeze().numpy()
            target = target.cpu().squeeze().numpy()
            pred = pred.cpu().squeeze().numpy()

            # Handle input data format
            if len(data.shape) == 3:
                data = data[0]  # Take first channel for multi-channel 2D
                resolution = data.shape[0]
            
            # Plot based on data type
            if is_temporal:  # [H, W, T]
                T = target.shape[-1]
                timesteps = [0, T//2, T-1]
                timestep_labels = ['Start', 'Mid', 'End']
                
                for t_idx, (t, label) in enumerate(zip(timesteps, timestep_labels)):
                    # Get shared vmin/vmax for this timestep's target-pred pair
                    vmin = min(target[:, :, t].min(), pred[:, :, t].min())
                    vmax = max(target[:, :, t].max(), pred[:, :, t].max())
                    
                    # Plot target at this timestep
                    im_target = axes[i, t_idx*2].imshow(target[:, :, t], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[i, t_idx*2].set_title(f'Target {idx}\n{label} (t={t})')
                    axes[i, t_idx*2].axis('off')
                    
                    # Plot prediction at this timestep
                    im_pred = axes[i, t_idx*2 + 1].imshow(pred[:, :, t], cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[i, t_idx*2 + 1].set_title(f'Pred {idx}\n{label} (t={t})')
                    axes[i, t_idx*2 + 1].axis('off')
                    
                    # Add shared colorbar between the pair with proper spacing
                    divider = make_axes_locatable(axes[i, t_idx*2 + 1])
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    fig.colorbar(im_pred, cax=cax)
            else:
                # Static 2D data
                im1 = axes[i, 0].imshow(data, cmap='viridis')
                axes[i, 0].set_title(f'Input {idx}')
                axes[i, 0].axis('off')
                plt.colorbar(im1, ax=axes[i, 0])

                im2 = axes[i, 1].imshow(target, cmap='viridis')
                axes[i, 1].set_title(f'Target {idx}')
                axes[i, 1].axis('off')
                plt.colorbar(im2, ax=axes[i, 1])

                im3 = axes[i, 2].imshow(pred, cmap='viridis')
                axes[i, 2].set_title(f'Prediction {idx}')
                axes[i, 2].axis('off')
                plt.colorbar(im3, ax=axes[i, 2])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    model.train()
    return fig


def plot_error_distribution(errors, save_path=None):
    """Plot distribution of prediction errors.

    Args:
        errors: Array of relative errors
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Relative L2 Error')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    axes[0].axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel('Relative L2 Error')
    axes[1].set_title('Error Statistics')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(train_losses, test_losses=None, save_path=None):
    """Plot training and test loss curves.

    Args:
        train_losses: List of training losses
        test_losses: List of test losses (optional)
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)

    if test_losses is not None:
        test_epochs = np.linspace(1, len(train_losses), len(test_losses))
        ax.plot(test_epochs, test_losses, 'r-', label='Test Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()

