import sys
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from examples.examples_src.models.model_utils import get_model
from examples.examples_src.utils.config import get_config
from examples.examples_src.utils.data_utils import get_datasets
from examples.examples_src.utils.train_utils import get_optimizer, get_criterion, get_denormalizer, get_data_loader, save_model
from examples.examples_src.utils.eval_utils import evaluate_model, get_plot_fn
from examples.examples_src.utils.mixed_res_loader import MixedResolutionDataLoader

def train_epoch(model, train_loader, optimizer, criterion, device, denormalizer, config):
    """Train for one epoch."""
    model.train()
    loss_sum = 0.0
    sample_count = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    # Dataset-type flags
    is_pyg = config['dataset'] == 'FlowPastCylinder'
    
    for batch in pbar:
        if is_pyg:
            # PyG datasets: batch is a single Data/Batch object
            batch = batch.to(device)
            data = batch
            targets = batch.y
            batch_size = batch.num_graphs
        else:
            # Standard datasets: batch is (data, targets) tuple
            data, targets = batch
            # data may be a tensor or a tuple (e.g., JEB: (pc_padded, xyt_padded))
            if isinstance(data, tuple):
                data = tuple(d.to(device) for d in data)
            else:
                data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[0]
        
        optimizer.zero_grad()
        output = model(data)
        
        # Denormalize if needed
        if denormalizer:
            output = denormalizer(output)
            targets = denormalizer(targets)
        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        # Aggregate loss based on loss_reduction config
        if config['loss_reduction'] == 'mean':
            # Mean reduction: weight by batch size
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
        else:
            # Sum reduction: sum over batch, divide by total samples
            loss_sum += loss.item()
            sample_count += batch_size
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return loss_sum / sample_count if sample_count > 0 else 0.0


def main():
    """Main training function for single-resolution baseline."""
    
    # Get config (loads YAML, applies CL overrides, sets seed)
    config = get_config()

    # Load datasets
    print("\nLoading datasets...")
    train_res = config['override_res'] if config['override_res'] else config['base_res']
    test_res = config['base_res']
    
    # Load both training and test resolutions
    # resolutions = [train_res] if train_res == test_res else [train_res, test_res]
    if config.get('mixed_res_training', False):
        resolutions = list(set(config['c2f_resolutions'] + [test_res]))
    elif config['override_res']:
        resolutions = [config['override_res']]
    elif train_res == test_res:
        resolutions = [train_res]
    else:
        resolutions = [train_res, test_res]
    config['c2f_resolutions'] = resolutions

    train_datasets, test_datasets, input_channels, input_size, dataset_specific = get_datasets(
        config, resolutions, config['device']
    )
    
    # Check for mixed-resolution training
    if config.get('mixed_res_training', False):
        train_resolutions = config['c2f_resolutions']
        train_datasets_by_res = {res: train_datasets[res] for res in train_resolutions}

        mixed_mode = config.get('mixed_res_mode', 'uniform')
        train_loader = MixedResolutionDataLoader(
            datasets_by_res=train_datasets_by_res,
            batch_size=config['batch_size'],
            total_samples=config['total_samples'],
            mode=mixed_mode,
            config=config if mixed_mode == 'mlmc_schedule' else None,
        )
        print(f"✓ Mixed-resolution training enabled ({mixed_mode})")
        print(f"  Train resolutions: {train_resolutions}")
    else:
        train_dataset = train_datasets[train_res]
        train_loader = get_data_loader(train_dataset, config, shuffle=True)
        print(f"✓ Loaded datasets")
        print(f"  Train resolution: {train_res} ({len(train_dataset)} samples)")
    
    test_dataset = test_datasets[test_res]
    print(f"  Test resolution: {test_res} ({len(test_dataset)} samples)")
    test_loader = get_data_loader(test_dataset, config, shuffle=False)
    print("✓ Test DataLoader created")

    # Create model
    print("\nInitializing model...")
    model = get_model(config, input_channels=input_channels, input_size=input_size)
    model = model.to(config['device'])

    # Create optimizer
    print(f"✓ Optimizer: {config['optimizer']}")
    optimizer = get_optimizer(model, config)

    # Create criterion
    criterion = get_criterion(config)
    print(f"✓ Criterion: {config['loss_type']}")

    # Create denormalizer
    denormalizer = get_denormalizer(train_datasets, config)
    print(f"✓ Denormalizer: {'enabled' if denormalizer else 'disabled'}")
    print(f"✓ Loss reduction: {config['loss_reduction']}")
    
    # Create plot function
    plot_fn = get_plot_fn(config)
    if plot_fn and config.get('eval_plot_every'):
        print(f"✓ Plotting enabled (every {config['eval_plot_every']} epochs)")
    
    # Create scheduler
    scheduler = None
    if config['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['step_size'], 
            gamma=config['gamma']
        )
        print(f"✓ Scheduler: StepLR(step_size={config['step_size']}, gamma={config['gamma']})")
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            config=config,
            name=config.get('wandb_run_name', None)
        )
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    # Training loop
    total_train_time = 0
    total_eval_time = 0
    best_test_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            config['device'], denormalizer, config
        )
        epoch_train_time = time.time() - epoch_start
        total_train_time += epoch_train_time
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Evaluate
        if (epoch + 1) % config['eval_every'] == 0 or epoch == config['epochs'] - 1:
            eval_start = time.time()
            eval_metrics = evaluate_model(
                model,
                test_loader,
                criterion,
                config['device'],
                denormalizer,
                config
            )
            epoch_eval_time = time.time() - eval_start
            total_eval_time += epoch_eval_time
            
            test_loss = eval_metrics['test_loss']
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Train Time: {epoch_train_time:.2f}s | Eval Time: {epoch_eval_time:.2f}s")
            
            # Save model (best or last epoch)
            if (config.get('save_model', False) or config.get('save_wandb_artifact', False)) and (test_loss < best_test_loss or epoch == config['epochs'] - 1):
                best_test_loss = test_loss
                metrics = {
                    'test_loss': test_loss,
                    'train_loss': train_loss,
                }
                save_model(model, optimizer, epoch + 1, config, metrics, is_best=True)
                print(f"  ✓ Saved best model (test_loss: {test_loss:.6f})")
            
            # Generate plots if enabled
            eval_plot_every = config.get('eval_plot_every', None)
            if eval_plot_every and plot_fn and ((epoch + 1) % eval_plot_every == 0 or epoch == config['epochs'] - 1):
                fig = plot_fn(
                    model=model,
                    dataset=test_dataset,
                    device=config['device'],
                    epoch=epoch + 1,
                    denormalizer=denormalizer
                )
                if config['use_wandb'] and fig:
                    wandb.log({"prediction_plot": wandb.Image(fig)}, step=epoch + 1)
                import matplotlib.pyplot as plt
                plt.close(fig)
            
            if config['use_wandb']:
                wandb.log({
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "epoch_train_time": epoch_train_time,
                    "epoch_eval_time": epoch_eval_time,
                    "total_train_time": total_train_time,
                    "total_eval_time": total_eval_time,
                    "epoch": epoch + 1
                })
        else:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | Train Loss: {train_loss:.6f} | Train Time: {epoch_train_time:.2f}s")
            if config['use_wandb']:
                wandb.log({
                    "train_loss": train_loss,
                    "epoch_train_time": epoch_train_time,
                    "total_train_time": total_train_time,
                    "epoch": epoch + 1
                })
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    final_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        config['device'],
        denormalizer,
        config
    )
    print(f"Test Loss: {final_metrics['test_loss']:.6f}")
    if 'rel_l2_error' in final_metrics:
        print(f"Relative L2 Error: {final_metrics['rel_l2_error']:.6f}")
    
    if config['use_wandb']:
        wandb.finish()
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print(f"Final test loss: {final_metrics['test_loss']:.6f}")
    print("="*80)


if __name__ == '__main__':
    sys.exit(main())