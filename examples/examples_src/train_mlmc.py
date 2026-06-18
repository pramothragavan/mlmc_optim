"""Generic training script for MLMC optimization.

Usage:
    python examples/examples_src/train_mlmc.py --config examples/pdes/darcy_flow/darcy_fno.yaml
    python examples/examples_src/train_mlmc.py --config examples/pdes/darcy_flow/darcy_fno.yaml --epochs 100 --lr 0.001
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlmc_optim import MLMCTrainer
from examples.examples_src.utils.config import get_config
from examples.examples_src.utils.data_utils import get_datasets
from examples.examples_src.utils.train_utils import setup_optimizers, get_criterion, get_denormalizer, get_data_loader
from examples.examples_src.utils.eval_utils import evaluate_model, get_plot_fn
from examples.examples_src.models.model_utils import get_model
from examples.examples_src.utils.gradient_analysis import evaluate_gradient_differences

def main():
    """Main training function."""
    
    # Get config (loads YAML, applies CL overrides, sets seed)
    config = get_config()

    # Load datasets
    print("\nLoading datasets...")
    c2f_resolutions = config['c2f_resolutions']
    train_datasets, test_datasets, input_channels, input_size, dataset_specific = get_datasets(config, c2f_resolutions, config['device'])
    print(f"✓ Loaded {len(train_datasets)} resolution levels")
    for res, dataset in train_datasets.items():
        print(f"  Resolution {res}: {len(dataset)} samples")
    
    # Create model
    print("\nInitializing model...")
    model = get_model(config, input_channels=input_channels, input_size=input_size)

    # Create optimizer
    optimizer, scheduler = setup_optimizers(model, config)

    # Create training and evaluation criteria
    criterion = get_criterion(config)
    eval_criterion = get_criterion(config, for_eval=True)
    print(f"✓ Criterion: {config['loss_type']}, with reduction {criterion.reduction}")

    # Create per-resolution denormalizers for MLMC training
    denormalizers = None
    eval_norm_stats = None
    if config.get('normalize_output', True) and config.get('model') in ['fno', 'fno3d']:
        denormalizers = {}
        device = config['device']

        def make_denorm(m, s):
            def denorm(x):
                return x * (s + 1e-5) + m
            return denorm

        for res, dataset in train_datasets.items():
            if hasattr(dataset, 'output_mean') and hasattr(dataset, 'output_std'):
                mean = dataset.output_mean.to(device)
                std = dataset.output_std.to(device)
                denormalizers[res] = make_denorm(mean, std)

        eval_mean = dataset_specific.get('eval_output_mean')
        eval_std = dataset_specific.get('eval_output_std')
        if eval_mean is not None and eval_std is not None:
            eval_norm_stats = (eval_mean, eval_std)
            denormalizers[config['base_res']] = make_denorm(
                eval_mean.to(device), eval_std.to(device))

    print(f"✓ Denormalizers: {'enabled' if denormalizers else 'disabled'}")
    print(f"✓ Loss reduction: {config['loss_reduction']}")

    # Create a single eval DataLoader at the base resolution (the only one with a test dataset)
    eval_dataset = test_datasets[config['base_res']]
    eval_loader = get_data_loader(eval_dataset, config, shuffle=False)
    print("✓ Eval DataLoader created")

    # Create plot function
    plot_fn = get_plot_fn(config)
    if plot_fn:
        print("✓ Plotting function created")
    
    # Create trainer
    print("\nInitializing MLMC trainer...")
    trainer = MLMCTrainer(
        model=model,
        c2f_resolutions=config['c2f_resolutions'],
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        eval_criterion=eval_criterion,
        eval_fn=evaluate_model,
        plot_fn=plot_fn,
        eval_loader=eval_loader,
        grad_eval_fn=evaluate_gradient_differences if config['eval_grad_every'] else None,
        denormalizers=denormalizers,
        eval_norm_stats=eval_norm_stats,
        device=config['device'],
        sample_sizes=config.get('sample_sizes', None),
        batch_sizes=config.get('batch_sizes', None),
        seed=config['seed'],
        epochs=config['epochs'],
        eval_every=config['eval_every'],
        use_wandb=config['use_wandb'],
        config=config
    )
    print("✓ Trainer initialized")
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    test_loss, train_losses = trainer.fit()
    
    print("\n" + "=" * 80)
    print("✓ Training complete!")
    print(f"Final test loss: {test_loss:.6f}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
