"""MLMC Trainer for Neural Operators.

Implements Multi-Level Monte Carlo training with variance reduction.
"""

import json
import os
from collections import defaultdict
import time
from typing import Dict, Optional, List
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import gc
import wandb
import matplotlib.pyplot as plt
import copy

from mlmc_optim.batcher import MLMCBatcher
from mlmc_optim.spectral_diagnostics import SpectralDiagnostics
from mlmc_optim.kernel_drift import KernelDriftDiagnostics


def sync_device(device):
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev.startswith("xpu"):
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and hasattr(xpu, "is_available") and xpu.is_available():
            xpu.synchronize()


def json_ready(obj):
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        obj = obj.detach().cpu()
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [json_ready(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    return str(obj)


def write_local_outputs(config, history_rows, eval_rows):
    out_dir = config.get("out_dir")
    if not out_dir:
        return

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(json_ready(config), f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(json_ready(history_rows), f, indent=2)

    if eval_rows:
        arrays = {}
        keys = sorted({k for row in eval_rows for k in row.keys()})
        for key in keys:
            vals = []
            for row in eval_rows:
                val = row.get(key, np.nan)
                if val is None:
                    vals.append(np.nan)
                elif isinstance(val, (int, float, bool, np.integer, np.floating, np.bool_)):
                    vals.append(float(val))
                else:
                    vals = None
                    break
            if vals is not None:
                arrays[key] = np.asarray(vals, dtype=float)
        if arrays:
            np.savez(os.path.join(out_dir, "metrics_eval.npz"), **arrays)

    final_eval = eval_rows[-1] if eval_rows else {}
    final_history = history_rows[-1] if history_rows else {}
    best_eval = None
    for row in eval_rows:
        val = row.get("test_loss")
        if val is None:
            continue
        if best_eval is None or float(val) < float(best_eval.get("test_loss")):
            best_eval = row
    summary = {
        "n_logged_rows": len(history_rows),
        "n_eval_rows": len(eval_rows),
        "final_test_loss": final_eval.get("test_loss"),
        "best_test_loss": best_eval.get("test_loss") if best_eval else None,
        "final_epoch_completed": (
            int(final_history.get("epoch", -1)) + 1 if final_history else 0
        ),
        "final_eval_cum_train_time": final_eval.get(
            "eval_cum_train_time", final_eval.get("cum_train_time")
        ),
    }
    if final_history:
        summary.update({
            "final_cum_train_time": final_history.get("cum_train_time"),
            "final_epoch_train_time": final_history.get("epoch_train_time"),
            "final_time_per_sample": final_history.get("time_per_sample"),
            "final_epoch_eval_block_time": final_history.get("epoch_eval_block_time"),
            "final_epoch_total_time": final_history.get("epoch_total_time"),
        })
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(json_ready(summary), f, indent=2)


class MLMCTrainer:
    """MLMC Trainer for neural operators.
    
    Implements hierarchical sampling and telescopic gradient estimation
    for efficient training on multi-resolution PDE data.
    
    Args:
        model: PyTorch model (e.g., FNO2d)
        data_dir: Path to data directory (e.g., 'data/darcy2d')
        resolutions: List of resolutions from coarse to fine [30, 60, 120, 241]
        datasets: Pre-loaded datasets dict (alternative to data_dir)
        test_datasets: Pre-loaded test datasets dict (optional)
        dataset_name: Dataset name for path construction (default: inferred from data_dir)
        lr: Learning rate
        device: Device to train on
        epochs: Number of training epochs
        sample_sizes: Samples per resolution (auto-computed if None)
        batch_sizes: Batch size per resolution (auto-computed if None)
        seed: Random seed
        eval_every: Evaluate every N epochs
        use_wandb: Log to Weights & Biases
        fraction: Fraction of data to use (for fast testing)
    """
    
    def __init__(
        self,
        model: nn.Module,
        c2f_resolutions: List[int],
        train_datasets: Dict[int, Dataset],
        test_datasets: Dict[int, Dataset],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: nn.Module,
        eval_criterion: Optional[nn.Module] = None,
        eval_fn: Optional[callable] = None,
        eval_loader: Optional["DataLoader"] = None,
        grad_eval_fn: Optional[callable] = None,
        plot_fn: Optional[callable] = None,
        sample_sizes: Optional[List[int]] = None,
        batch_sizes: Optional[List[int]] = None,
        normalizer: Optional[callable] = None,
        denormalizers: Optional[Dict[int, callable]] = None,
        eval_norm_stats: Optional[tuple] = None,
        device: str = 'cuda',
        seed: int = None,
        epochs: int = 10,
        lr: float = 0.001,
        eval_every: int = 1,
        use_wandb: bool = False,
        fraction: float = 1.0,
        config: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.c2f_resolutions = c2f_resolutions
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.device = device

        # Store config first so it can be used for optimizer/scheduler setup
        self.config = config if config is not None else {}

        # Always use a single optimizer / scheduler (standard training path)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        # Use a separate criterion for evaluation if provided
        self.eval_criterion = eval_criterion if eval_criterion is not None else criterion
        self.normalizer = normalizer
        self.denormalizers = denormalizers
        self.eval_norm_stats = eval_norm_stats
        self.eval_fn = eval_fn
        self.eval_loader = eval_loader
        self.grad_eval_fn = grad_eval_fn
        self.plot_fn = plot_fn
        # Parameter management: config overrides defaults
        self.epochs = self.config.get('epochs', epochs)
        self.lr = self.config.get('lr', lr)
        self.eval_every = self.config.get('eval_every', eval_every)
        # Gradient analysis parameters
        self.eval_grad_every = self.config.get('eval_grad_every', None)
        self.eval_grad_rand = self.config.get('eval_grad_rand', False)
        self.use_wandb = self.config.get('use_wandb', use_wandb)
        self.fraction = self.config.get('fraction', fraction)

        # MLMC-specific parameters with defaults
        self.mlmc_sampling_style = self.config.get('mlmc_sampling_style', 'geom_prog')
        self.mlmc_batch_style = self.config.get('mlmc_batch_style', 'geom_prog_batch')
        self.mlmc_data_size_multiplier = self.config.get('mlmc_data_size_multiplier', 2)
        self.mlmc_batch_size_multiplier = self.config.get('mlmc_batch_size_multiplier', 2)
        # self.max_level = len(c2f_resolutions)
        self.max_level = config['mlmc_max_level'] if config['mlmc_max_level'] is not None else len(c2f_resolutions)
        
        # Use actual coarsest dataset size (accounts for train_subset)
        coarsest_res = c2f_resolutions[0]
        self.total_samples = len(train_datasets[coarsest_res])
        self.batch_size = self.config.get('batch_size', 32)  # Base batch size
        self.use_optimized_cache = self.config.get('mlmc_hierarchy_cache', False)
        self.load_gpu_epoch = self.config.get('load_gpu_epoch', False)

        # Prescribed mode parameters (required if using prescribed modes)
        self.mlmc_samples_per_level = self.config.get('mlmc_samples_per_level', None)  # Required for 'prescribed' sampling
        self.mlmc_batch_sizes = self.config.get('mlmc_batch_sizes', None)  # Required for 'prescribed_batch'

        # Auto-compute MLMC parameters
        if sample_sizes is None or batch_sizes is None:
            sample_sizes, batch_sizes = self._compute_mlmc_parameters()
        self.sample_sizes = sample_sizes
        self.batch_sizes = batch_sizes
        
        # Basic diagnostics
        print(f"Sample_sizes: {sample_sizes}, batch_sizes: {batch_sizes}")
        print(f"MLMC configuration:")
        print(f"  Pairing style: {self.config['mlmc_pairing']}")
        print(f"  Sampling style: {self.mlmc_sampling_style}")
        print(f"  Resolutions: {c2f_resolutions}")
        print(f"  Sample sizes: {sample_sizes}")
        print(f"  Batch sizes: {batch_sizes}")
        # Map each resolution to its sample size and batch size
        res_level_info = []
        for res, n_samples, bsz in zip(c2f_resolutions, sample_sizes, batch_sizes):
            n_batches = int(np.ceil(n_samples / bsz)) if bsz > 0 else 0
            res_level_info.append((res, n_samples, bsz, n_batches))
        print("  Per-level summary (res, samples, batch_size, num_batches):")
        for res, n_samples, bsz, n_batches in res_level_info:
            print(f"    res={res}: samples={n_samples}, batch_size={bsz}, num_batches={n_batches}")
        
        self.batcher = MLMCBatcher(
            config,
            c2f_resolutions, 
            sample_sizes,
            batch_sizes,
            self.total_samples,
            seed, 
        )
        
        wandb.init(
            project=self.config['wandb_project'],
            entity=self.config['wandb_entity'],
            group=self.config['wandb_group'],
            name=self.config['wandb_run_name'],
            config=self.config,
            mode="online" if self.use_wandb else "disabled"
        )

    def has_phase_schedule(self):
        return bool(self.config.get("epochs_per_phase"))

    def phase_total_epochs(self):
        if self.has_phase_schedule():
            total_epochs = int(sum(int(x) for x in self.config["epochs_per_phase"]))
            self.config["epochs"] = total_epochs
            self.config["total_epochs"] = total_epochs
            return total_epochs
        total_epochs = int(self.config.get("epochs", self.epochs))
        self.config["total_epochs"] = total_epochs
        return total_epochs

    def phase_index(self, epoch, cumulative_epochs):
        return int(np.searchsorted(cumulative_epochs, epoch, side="right") - 1)

    def phase_params(self, phase=None):
        if phase is None or not self.has_phase_schedule():
            return self.c2f_resolutions, self.sample_sizes, self.batch_sizes

        c2f_res_per_phase = self.config.get("c2f_res_per_phase")
        subset_size_per_phase = self.config.get("subset_size_per_phase")
        batch_size_per_phase = self.config.get("batch_size_per_phase")
        if c2f_res_per_phase is None or subset_size_per_phase is None:
            raise ValueError(
                "Phase schedules require c2f_res_per_phase and subset_size_per_phase."
            )

        active_resolutions = [int(r) for r in c2f_res_per_phase[phase]]
        active_sample_sizes = [int(n) for n in subset_size_per_phase[phase]]
        if batch_size_per_phase is not None:
            active_batch_sizes = [int(n) for n in batch_size_per_phase[phase]]
        else:
            active_batch_sizes = [int(self.batch_size)] * len(active_resolutions)

        if not len(active_resolutions) == len(active_sample_sizes) == len(active_batch_sizes):
            raise ValueError(
                "Each phase must have matching resolution, sample-size, and "
                "batch-size lengths."
            )

        return active_resolutions, active_sample_sizes, active_batch_sizes

    def make_batcher(self, active_resolutions, sample_sizes, batch_sizes):
        total_samples = len(self.train_datasets[active_resolutions[0]])
        return MLMCBatcher(
            self.config,
            active_resolutions,
            sample_sizes,
            batch_sizes,
            total_samples,
            self.config.get("seed"),
        )

    def set_phase_lr(self, phase):
        lr_per_phase = self.config.get("lr_per_phase")
        if lr_per_phase is None:
            return
        new_lr = float(lr_per_phase[phase])
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        print(f"[lr_per_phase] entering phase {phase} with lr={new_lr:.3e}")

    def eval_norm_stats_for_logging(self):
        if not self.config.get("normalize_output", True):
            return None, None
        if self.eval_norm_stats is not None:
            mean, std = self.eval_norm_stats
            return mean.to(self.device), std.to(self.device)
        base_res = self.config.get("base_res", self.c2f_resolutions[-1])
        dataset = self.train_datasets.get(base_res)
        if dataset is None:
            dataset = self.train_datasets[self.c2f_resolutions[-1]]
        if hasattr(dataset, "output_mean") and hasattr(dataset, "output_std"):
            return dataset.output_mean.to(self.device), dataset.output_std.to(self.device)
        return None, None

    def get_sample_sizes(self):
        """Get sample sizes for each level based on config."""
        if self.mlmc_sampling_style == 'geom_prog':
            # Calculate base sample size by working backwards from total samples
            total_mult = sum(self.mlmc_data_size_multiplier ** l for l in range(self.max_level))
            base_samples = self.total_samples // total_mult
            # Now multiply down from base samples - least samples at level 0 (finest)
            samples = [int(base_samples * (self.mlmc_data_size_multiplier ** (self.max_level - l))) for l in
                       range(1, self.max_level + 1)]
            return samples
        elif self.mlmc_sampling_style == 'prescribed':
            return self.mlmc_samples_per_level
        else:
            raise ValueError(f"Unknown MLMC sampling style: {self.mlmc_sampling_style}")

    def _compute_mlmc_parameters(self) -> tuple:
        """Compute optimal sample_sizes and batch_sizes based on MLMC theory."""

        # Get number of samples to use per resolution based on sampling style
        if self.mlmc_sampling_style in ['geom_prog', 'prescribed']:
            c2f_sample_sizes = self.get_sample_sizes()
        elif self.mlmc_sampling_style == 'random':
            raise NotImplementedError("Random sampling is not currently supported")
        else:
            c2f_sample_sizes = self.mlmc_samples_per_level

        if self.mlmc_batch_style == 'geom_prog_batch':
            c2f_batch_sizes = [int(self.batch_size * (self.mlmc_batch_size_multiplier ** i)) for i in
                               range(self.max_level)][::-1]
        elif self.mlmc_batch_style == 'prescribed_batch':
            c2f_batch_sizes = self.mlmc_batch_sizes
        else:
            c2f_batch_sizes = [self.batch_size] * self.max_level

        return c2f_sample_sizes, c2f_batch_sizes

    def load_epoch_indices_to_gpu(self, datasets, idxs_per_res, device):
        """
        Load only the necessary data to GPU for the current epoch.

        Args:
            datasets: Dictionary of datasets keyed by resolution
            idxs_per_res: Dictionary of indices keyed by resolution (from batcher.create_batches())
            device: Device to load the data to ('cuda')
        """
        print("Loading epoch data to GPU...")

        # Load only the data needed for this epoch
        for res, dataset in datasets.items():
            if res in idxs_per_res and len(idxs_per_res[res]) > 0:
                if hasattr(dataset, 'load_batch_indices_to_gpu'):
                    dataset.load_batch_indices_to_gpu(idxs_per_res[res], device)
                    print(f"Finished loading data res {res} to {device}")
                else:
                    print(f"Warning: Dataset for resolution {res} doesn't have load_batch_indices_to_gpu method")

        print("Done loading epoch data to GPU.")

    def unload_epoch_indices_from_gpu(self, datasets, idxs_per_res):
        """
        Unload data from GPU that was previously loaded for an epoch.
        Should be called at the end of an epoch with the same indices used for loading.

        Args:
            datasets: Dictionary of datasets keyed by resolution
            idxs_per_res: Dictionary of indices keyed by resolution (used in load_epoch_indices_to_gpu)
        """
        print("Unloading epoch data from GPU...")
        for res, dataset in datasets.items():
            if res in idxs_per_res and len(idxs_per_res[res]) > 0:
                if hasattr(dataset, 'unload_from_gpu'):
                    dataset.unload_from_gpu()
                else:
                    print(f"Warning: Dataset for resolution {res} doesn't have unload_from_gpu method")

        # Force garbage collection to clean up any dangling tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done unloading epoch data from GPU.")


    def mlmc_loss_handler(self, model, data_batch, targets, mode='single', device=None, timing_dict=None):
        """
        Compute MLMC gradient estimate using configured model, criterion, and denormalizer.
        
        Uses self.criterion which is already configured with the correct reduction mode
        (set via get_criterion with using_optimized_cache flag).
        
        Two modes:
        1. 'single': Compute loss for coarsest level (standalone term)
        2. 'pair': Compute loss difference between fine and coarse levels

        Args:
            model: Model to evaluate (allows flexibility for different models)
            data_batch: Dictionary of data batches keyed by resolution
            targets: Dictionary of targets keyed by resolution
            mode: 'single' for coarse level, 'pair' for resolution pair
            device: Computation device (for empty batch handling)
            timing_dict: Dictionary for timing statistics

        Returns:
            mode='single': loss tensor (per-example if criterion has reduction='none', else scalar)
            mode='pair': (diff_loss, loss_fine, loss_coarse) tuple
        """
        config = self.config
        if config['mlmc_hierarchy_cache']:
            reduction = 'none'
        elif config['loss_reduction'] == None:
            reduction = 'none'
        else:
            reduction = config['loss_reduction']

        if mode == 'single':
            # Single resolution forward pass (coarsest level)
            res = list(data_batch.keys())[0]  # Only one resolution in this case

            # Time forward pass
            forward_start = time.time()
            output = model(data_batch[res])

            # Apply per-resolution denormalizer (required)
            if self.denormalizers:
                denorm = self.denormalizers[res]
                output = denorm(output)
                target = denorm(targets[res])
            else:
                target = targets[res]

            # Compute loss using configured criterion
            loss = self.criterion(output, target)

            timing_dict['ts_forward'][str(res)] += time.time() - forward_start
            return loss

        elif mode == 'pair':
                # Callers construct data_batch as {fine_res: ..., coarse_res: ...},
            # so keys()[0] is fine and keys()[1] is coarse.
            fine_res, coarse_res = list(data_batch.keys())
            pair_key = f"{fine_res}_{coarse_res}"

            # Skip if either batch is empty
            if len(data_batch[fine_res]) == 0 or len(data_batch[coarse_res]) == 0:
                zero_tensor = torch.tensor(0.0, device=device)
                if reduction == 'none':
                    # For per-example case, return empty tensors
                    return (torch.tensor([], device=device), torch.tensor([], device=device))
                else:
                    return zero_tensor

            # Time forward pass
            forward_start = time.time()

            denorm_f = self.denormalizers.get(fine_res) if self.denormalizers else None
            denorm_c = self.denormalizers.get(coarse_res) if self.denormalizers else None

            # Forward pass at fine resolution
            output_fine = model(data_batch[fine_res])
            target_fine = targets[fine_res]
            if denorm_f is not None:
                output_fine = denorm_f(output_fine)
                target_fine = denorm_f(target_fine)

            # Forward pass at coarse resolution
            output_coarse = model(data_batch[coarse_res])
            target_coarse = targets[coarse_res]
            if denorm_c is not None:
                output_coarse = denorm_c(output_coarse)
                target_coarse = denorm_c(target_coarse)

            # Compute losses using configured criterion
            loss_fine = self.criterion(output_fine, target_fine)
            loss_coarse = self.criterion(output_coarse, target_coarse)

            # Handle based on reduction mode
            if reduction == 'none':
                # For per-example mode, return the raw losses for further processing
                timing_dict['ts_forward'][pair_key] += time.time() - forward_start
                return (loss_fine, loss_coarse)
            else:
                # Calculate difference loss
                diff_loss = loss_fine - loss_coarse
                timing_dict['ts_forward'][pair_key] += time.time() - forward_start
                return (diff_loss, loss_fine, loss_coarse)


    def fit(self):
        config = self.config
        device = self.device
        # Initialize metrics
        training_times = []
        eval_times = []
        total_train_time = 0
        total_eval_time = 0
        train_losses = []
        cum_train_time = 0
        best_test_loss = float('inf')
        history_rows = []
        eval_rows = []
        last_test_loss = None

        timing_stats = {
            'ts_batch_creation': 0,
            'ts_coarse': defaultdict(float),
            'ts_correction': defaultdict(float),
            'ts_forward': defaultdict(float),
            'ts_backward': 0  # Single backward pass time per batch
        }

        total_epochs = self.phase_total_epochs()
        cumulative_epochs = np.cumsum([0] + config.get("epochs_per_phase", [total_epochs]))
        prev_phase = None

        spectral_diag = None
        if config.get("spectral_diagnostics", False):
            if self.eval_loader is None:
                raise ValueError("spectral_diagnostics requires eval_loader.")
            spectral_out_dir = config.get("spectral_out_dir")
            if spectral_out_dir is None:
                spectral_out_dir = os.path.join(config.get("out_dir", "."), "spectral")
                config["spectral_out_dir"] = spectral_out_dir
            spectral_diag = SpectralDiagnostics(
                modes=config.get("fno_modes", 8),
                eval_loader=self.eval_loader,
                device=device,
                out_dir=spectral_out_dir,
            )
            print(f"Spectral diagnostics enabled, saving to {spectral_out_dir}")

        kernel_drift_diag = None
        if config.get("kernel_drift_diagnostics", False):
            probe_split = str(config.get("kernel_drift_probe_split", "train"))
            probe_res_config = config.get("kernel_drift_probe_res")
            if probe_res_config is None:
                base_res = int(config.get("base_res", self.c2f_resolutions[-1]))
                probe_res = base_res if base_res in self.train_datasets else int(self.c2f_resolutions[-1])
            else:
                probe_res = int(probe_res_config)
            if probe_split == "train":
                if probe_res not in self.train_datasets:
                    raise ValueError(
                        f"kernel_drift_probe_res={probe_res} is not in train_datasets "
                        f"{sorted(self.train_datasets)}.")
                probe_dataset = self.train_datasets[probe_res]
            elif probe_split == "test":
                base_res = config.get("base_res", self.c2f_resolutions[-1])
                probe_dataset = self.test_datasets[base_res]
                probe_res = int(base_res)
            else:
                raise ValueError("kernel_drift_probe_split must be 'train' or 'test'.")

            denormalizer = None
            if self.denormalizers is not None:
                denormalizer = self.denormalizers.get(probe_res)
            kernel_out_dir = config.get("kernel_drift_out_dir")
            if kernel_out_dir is None:
                kernel_out_dir = os.path.join(config.get("out_dir", "."), "kernel_drift")
                config["kernel_drift_out_dir"] = kernel_out_dir
            kernel_drift_diag = KernelDriftDiagnostics(
                config=config,
                dataset=probe_dataset,
                device=device,
                denormalizer=denormalizer,
                out_dir=kernel_out_dir,
            )
            print(
                "Kernel drift diagnostics enabled: "
                f"split={probe_split}, res={probe_res}, out={kernel_out_dir}"
            )

        if kernel_drift_diag is not None and kernel_drift_diag.should_log(0):
            init_phase = self.phase_index(0, cumulative_epochs) if self.has_phase_schedule() else None
            init_resolutions, _, _ = self.phase_params(init_phase)
            kernel_drift_diag.log_snapshot(
                self.model,
                epoch_completed=0,
                phase=init_phase,
                phase_res=init_resolutions[-1] if init_resolutions else None,
                cum_train_time=0.0,
            )

        for epoch in range(total_epochs):
            print(f"\nEpoch {epoch + 1}/{total_epochs}")
            self.model.train()
            sync_device(device)
            epoch_start = time.time()
            eval_time_total = 0.0
            batch_creation_start = time.time()

            if self.has_phase_schedule():
                phase = self.phase_index(epoch, cumulative_epochs)
                if phase != prev_phase:
                    self.set_phase_lr(phase)
                    prev_phase = phase
                active_resolutions, active_sample_sizes, active_batch_sizes = self.phase_params(phase)
                batcher = self.make_batcher(
                    active_resolutions, active_sample_sizes, active_batch_sizes
                )
            else:
                phase = None
                active_resolutions = self.c2f_resolutions
                batcher = self.batcher

            idxs_per_res, batches = batcher.get_epoch_indices()
            # Batch diagnostics: print explicit shapes for coarsest-level indices
            # try:
            #     if batches:
            #         # Coarsest resolution and total samples there
            #         coarsest_res = batches[0][0][0][-1]
            #         total_coarse_samples = sum(len(batch[0][1]) for batch in batches)
            #         print(
            #             f"Total coarse samples this epoch: {total_coarse_samples}, "
            #             f"num_batches: {len(batches)} (res {coarsest_res})"
            #         )

            #         # Show shapes for the first few batches at the coarsest level
            #         max_show = min(5, len(batches))
            #         for b_idx in range(max_show):
            #             res_info, idxs = batches[b_idx][0]
            #             print(
            #                 f"  batch {b_idx}: res={res_info[-1]}, "
            #                 f"idxs.shape={tuple(idxs.shape)}, size={len(idxs)}"
            #             )
            # except Exception as e:
            #     print(f"Warning: failed to compute batch summary: {e}")
            # print(f"Total samples: {sum([len(b[1]) for b in batches])}, Batch summary: {[(b[0],len(b[1])) for b in batches]}")

            if config['load_gpu_epoch']:
                self.load_epoch_indices_to_gpu(self.train_datasets, idxs_per_res, self.device)

            # Initialize loss and gradient tracking
            coarse_loss = 0
            pair_losses = defaultdict(float)
            total_steps = 0
            pair_samples = defaultdict(int)
            total_samples = 0

            # Initialize cache for retained losses across levels
            coarse_samples = 0
            timing_stats['ts_batch_creation'] += time.time() - batch_creation_start

            # Training loop
            pbar = tqdm(total=len(batches), desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(batches):

                # 1. Coarsest resolution update
                # Get available resolutions for this batch
                available_resolutions = [res_data[0][-1] for res_data in batch]

                coarse_start = time.time()
                coarsest_res = available_resolutions[0]
                batch_indices = batch[0][1]
                coarsest_data, coarsest_targets = self.train_datasets[coarsest_res].get_items(batch_indices)
                coarsest_data = coarsest_data.to(device)
                coarsest_targets = coarsest_targets.to(device)

                # Skip if coarsest batch is empty
                if len(coarsest_data) == 0:
                    continue

                # Create data dictionary
                data_dict = {coarsest_res: coarsest_data}

                # Compute coarse loss (criterion already has correct reduction mode)
                coarse_result = self.mlmc_loss_handler(
                    self.model,
                    data_dict,
                    {coarsest_res: coarsest_targets},
                    mode='single',
                    device=device,
                    timing_dict=timing_stats
                )

                # Handle caching logic if enabled
                if self.use_optimized_cache:
                    coarse_losses = coarse_result
                    
                    # Cache losses for retained indices (used in next level)
                    if len(batch) > 1:  # Make sure there are correction terms
                        retained_indices = batch[1][1]
                        retained_mask = np.isin(batch_indices, retained_indices)
                        retain_mask = torch.tensor(retained_mask, dtype=torch.bool, device=device)

                        if retain_mask.any():
                            # For PyG datasets, convert batch-level mask to node-level
                            is_pyg_dataset = config['dataset'] == 'FlowPastCylinder'
                            if is_pyg_dataset and hasattr(coarsest_data, 'batch'):
                                batch_mask = torch.zeros(coarsest_data.batch.max().item() + 1, dtype=torch.bool, device=device)
                                batch_mask[retain_mask.nonzero().squeeze()] = True
                                retain_mask = batch_mask[coarsest_data.batch]

                        # Aggregate retained losses for caching
                        if config['loss_reduction'] == 'sum':
                            cached_coarse_loss = coarse_losses[retain_mask].sum()
                        else:
                            cached_coarse_loss = coarse_losses[retain_mask].mean()

                    # Aggregate all coarse losses for this batch
                    if config['loss_reduction'] == 'sum':
                        coarse_loss_batch = coarse_losses.sum()
                    else:
                        coarse_loss_batch = coarse_losses.mean()
                else:
                    # Non-caching: loss is already aggregated
                    coarse_loss_batch = coarse_result

                timing_stats['ts_coarse'][str(coarsest_res)] += time.time() - coarse_start

                # Initialize the total batch loss with the coarse loss
                total_batch_loss = coarse_loss_batch

                # Update metrics for the coarsest level
                if config['loss_reduction'] == 'mean':
                    coarse_loss += coarse_loss_batch.item() * len(batch_indices)
                    coarse_samples += len(batch_indices)
                else:
                    coarse_loss += coarse_loss_batch.item()
                    coarse_samples += len(batch_indices)
                total_samples += len(batch_indices)  # Count coarsest resolution samples

                # 2. Resolution pair updates - process from coarsest to finest
                batch_corrections = batch[1:]

                for level, correction_item in enumerate(batch_corrections):
                    # Unpack the correction item
                    (coarse_pair_res, fine_pair_res), correction_indices = correction_item
                    pair_key = f"{fine_pair_res}_{coarse_pair_res}"
                    correction_start = time.time()

                    # Skip if batch is empty
                    if len(correction_indices) == 0:
                        continue

                    # Get fine resolution data
                    fine_pair_data, fine_pair_targets = self.train_datasets[fine_pair_res].get_items(correction_indices)
                    fine_pair_data = fine_pair_data.to(self.device)
                    fine_pair_targets = fine_pair_targets.to(self.device)

                    if self.use_optimized_cache:
                        # Caching: compute only fine loss, reuse cached coarse loss
                        fine_losses = self.mlmc_loss_handler(
                            self.model,
                            {fine_pair_res: fine_pair_data},
                            {fine_pair_res: fine_pair_targets},
                            mode='single',
                            device=self.device,
                            timing_dict=timing_stats
                        )

                        # Aggregate fine losses
                        if config['loss_reduction'] == 'sum':
                            fine_loss = fine_losses.sum()
                        else:
                            fine_loss = fine_losses.mean()

                        # Use cached coarse loss from previous level
                        batch_coarse_loss = cached_coarse_loss
                        diff_loss = fine_loss - batch_coarse_loss

                        # Cache fine losses for next level (if it exists)
                        if level + 1 < len(batch_corrections):
                            retained_indices = batch_corrections[level + 1][1]
                            retained_mask = np.isin(correction_indices, retained_indices)
                            retain_mask = torch.tensor(retained_mask, dtype=torch.bool, device=self.device)

                            if retain_mask.any():
                                # For PyG datasets, convert batch-level mask to node-level
                                is_pyg_dataset = config['dataset'] == 'FlowPastCylinder'
                                if is_pyg_dataset and hasattr(fine_pair_data, 'batch'):
                                    batch_mask = torch.zeros(fine_pair_data.batch.max().item() + 1, dtype=torch.bool, device=self.device)
                                    batch_mask[retain_mask.nonzero().squeeze()] = True
                                    retain_mask = batch_mask[fine_pair_data.batch]

                                # Aggregate retained losses for caching
                                if config['loss_reduction'] == 'sum':
                                    cached_coarse_loss = fine_losses[retain_mask].sum()
                                else:
                                    cached_coarse_loss = fine_losses[retain_mask].mean()

                            else:
                                # No retained samples for next level; default to zero on correct device
                                cached_coarse_loss = torch.tensor(0.0, device=self.device)

                            # # Aggregate retained losses for caching
                            # if config['loss_reduction'] == 'sum':
                            #     cached_coarse_loss = fine_losses[retain_mask].sum()
                            # else:
                            #     cached_coarse_loss = fine_losses[retain_mask].mean()



                        pair_loss_batch = (diff_loss, fine_loss, batch_coarse_loss)

                    else:
                        # Non-caching: compute both fine and coarse losses
                        coarse_pair_data, coarse_pair_targets = self.train_datasets[coarse_pair_res].get_items(correction_indices)
                        coarse_pair_data = coarse_pair_data.to(self.device)
                        coarse_pair_targets = coarse_pair_targets.to(self.device)

                        # Skip if either batch is empty
                        if len(fine_pair_data) == 0 or len(coarse_pair_data) == 0:
                            continue

                        # Compute pair loss (returns diff, fine, coarse)
                        pair_loss_batch = self.mlmc_loss_handler(
                            self.model,
                            {fine_pair_res: fine_pair_data, coarse_pair_res: coarse_pair_data},
                            {fine_pair_res: fine_pair_targets, coarse_pair_res: coarse_pair_targets},
                            mode='pair',
                            device=self.device,
                            timing_dict=timing_stats
                        )

                    timing_stats['ts_correction'][pair_key] += time.time() - correction_start

                    # Extract components from pair_loss_batch
                    diff_loss, pair_fine_loss, pair_coarse_loss = pair_loss_batch

                    # Accumulate diff_loss to total_batch_loss without immediate backward
                    total_batch_loss += diff_loss

                    if config['dataset'] == 'FlowPastCylinder':
                        data_size = getattr(fine_pair_data, 'batch_size', len(fine_pair_data))
                    else:
                        data_size = fine_pair_data.shape[0]

                    # Accumulate pair loss
                    if config['loss_reduction'] == 'mean':
                        pair_losses[pair_key] += diff_loss.item() * data_size
                        pair_samples[pair_key] += data_size
                    else:
                        pair_losses[pair_key] += diff_loss.item()
                        pair_samples[pair_key] += data_size
                    total_samples += data_size

                # Backward pass
                backward_start = time.time()
                accum = config['gradient_accumulation_steps']
                if batch_idx % accum == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                # Scale loss if gradient accumulation is used
                scaled_batch_loss = total_batch_loss / accum
                scaled_batch_loss.backward()

                timing_stats['ts_backward'] += time.time() - backward_start

                # Take optimizer step after accumulating gradients
                if (((batch_idx + 1) % accum == 0) or ((batch_idx + 1) == len(batches))):
                    # Standard single-optimizer step
                    self.optimizer.step()
                    total_steps += 1
                    self.optimizer.zero_grad()

                if batch_idx % 10 == 0:  # Update every 10 batches
                    pbar.update(10)
                pbar.set_postfix({
                    'coarse_loss': coarse_loss / (total_steps if total_steps > 0 else 1),
                    **{f'pair_loss_{k}': v / (total_steps if total_steps > 0 else 1) for k, v in pair_losses.items()}
                })

            pbar.close()

            # Step LR scheduler if present. Phase-specific LR schedules own the LR.
            if hasattr(self, 'scheduler') and self.scheduler is not None and config.get("lr_per_phase") is None:
                self.scheduler.step()


            epoch_train_time = time.time() - epoch_start
            training_times.append(epoch_train_time)
            total_train_time += epoch_train_time
            cum_train_time += epoch_train_time
            train_losses.append(coarse_loss)

            # Compute average losses and metrics
            coarse_loss = coarse_loss / coarse_samples if coarse_samples > 0 else 0

            # Avoid division by zero for pair losses
            pair_losses = {k: v / pair_samples[k] if pair_samples[k] > 0 else 0.0 for k, v in pair_losses.items()}
            train_time_per_sample = epoch_train_time / total_samples if total_samples > 0 else 0

            # Calculate total loss as sum of coarse and pair losses
            total_loss = coarse_loss + sum(pair_losses.values())
            current_lr = self.optimizer.param_groups[0].get("lr", self.lr)
            log_metrics = {
                'epoch': epoch,
                'epoch_completed': epoch + 1,
                'coarse_loss': coarse_loss,
                'total_loss': total_loss,
                'epoch_train_time': epoch_train_time,
                'cum_train_time': cum_train_time,
                'cumulative_train_time': cum_train_time,
                'time_per_sample': train_time_per_sample,
                'n_train_samples': total_samples,
                'n_optimizer_steps': total_steps,
                'lr': current_lr,
                'phase': -1 if phase is None else phase,
                'phase_resolution': (
                    active_resolutions[-1] if active_resolutions else None
                ),
                **{f'pair_loss_{k}': v for k, v in pair_losses.items()},
            }

            # Evaluation phase (every eval_every epochs or at the end)
            if self.eval_every and (epoch % self.eval_every == 0 or epoch == total_epochs - 1):
                eval_start = time.time()
                eval_denorm = None
                
                if self.eval_fn is not None:
                    # Use provided eval function - it returns metrics dict.
                    # Expect a pre-built eval_loader to be provided at init.
                    if self.eval_loader is None:
                        raise ValueError("eval_loader must be provided when eval_fn is not None")

                    # For evaluation, use the finest-resolution denormalizer if available
                    if self.denormalizers is not None and isinstance(self.denormalizers, dict):
                        eval_res = config.get("base_res", self.c2f_resolutions[-1])
                        eval_denorm = self.denormalizers.get(eval_res, None)
                    else:
                        eval_denorm = None

                    eval_metrics = self.eval_fn(
                        self.model,
                        self.eval_loader,
                        self.eval_criterion,
                        device,
                        eval_denorm,
                        self.config,
                    )
                    test_loss = eval_metrics.get('test_loss', 0.0)
                elif self.test_datasets is not None:
                    # Minimal evaluation - just compute test loss
                    self.model.eval()
                    test_loss = 0.0
                    eval_res = config.get("base_res", self.c2f_resolutions[-1])
                    test_dataset = self.test_datasets[eval_res]
                    with torch.no_grad():
                        for i in range(len(test_dataset)):
                            data, target = test_dataset[i]
                            data, target = data.to(device).unsqueeze(0), target.to(device).unsqueeze(0)
                            output = self.model(data)
                            if self.denormalizers:
                                denorm = self.denormalizers.get(eval_res)
                                if denorm is not None:
                                    output = denorm(output)
                                    target = denorm(target)
                            test_loss += self.criterion(output, target).item()
                    test_loss /= len(test_dataset)
                    self.model.train()
                    eval_metrics = {'test_loss': test_loss}
                else:
                    test_loss = 0.0
                    eval_metrics = {'test_loss': test_loss}
                
                epoch_eval_time = time.time() - eval_start
                eval_time_total = epoch_eval_time
                last_test_loss = test_loss
                
                # Track best loss and save model
                improved = test_loss < best_test_loss
                if improved:
                    best_test_loss = test_loss

                if (config.get('save_model', False) or config.get('save_wandb_artifact', False)) and (improved or epoch == total_epochs - 1):
                    metrics = {
                        'test_loss': test_loss,
                        'coarse_loss': coarse_loss,
                        'total_loss': total_loss,
                    }
                    self.save_model(self.model, self.optimizer,
                                    epoch, config, metrics, is_best=True)
                    print(f"  ✓ Saved best model (test_loss: {test_loss:.6f})")
                
                # Prepare metrics for logging
                log_metrics.update({
                    'epoch_eval_time': epoch_eval_time,
                    'eval_cum_train_time': cum_train_time,
                    **eval_metrics,  # Include all eval metrics
                })

                # Generate prediction plots if enabled
                eval_plot_every = config.get('eval_plot_every', None)
                if eval_plot_every and self.plot_fn and (epoch % eval_plot_every == 0 or epoch == total_epochs - 1):
                    try:
                        fig = self.plot_fn(
                            model=self.model,
                            dataset=self.test_datasets[
                                config.get("base_res", self.c2f_resolutions[-1])
                            ],
                            device=device,
                            epoch=epoch,
                            denormalizer=eval_denorm
                        )
                        
                        # Log to wandb if enabled
                        if self.use_wandb:
                            log_metrics['prediction_plot'] = wandb.Image(fig)
                        
                        # Save locally if checkpoint_dir is specified
                        if config.get('save_checkpoint', False) and config.get('checkpoint_dir'):
                            os.makedirs(config['checkpoint_dir'], exist_ok=True)
                            save_path = os.path.join(config['checkpoint_dir'], f'prediction_epoch_{epoch+1}.png')
                            fig.savefig(save_path, dpi=150, bbox_inches='tight')
                            print(f"  Saved prediction plot to {save_path}")
                        
                        plt.close(fig)
                    except Exception as e:
                        print(f"Warning: Failed to generate prediction plot: {e}")

                # Optional gradient analysis at evaluation epochs
                if (
                    self.grad_eval_fn is not None
                    and self.eval_grad_every is not None
                    and (epoch % self.eval_grad_every == 0 or epoch == total_epochs - 1)
                ):
                    # try:
                    # Choose batches for gradient analysis
                    if self.eval_grad_rand:
                        _, grad_batches = batcher.get_epoch_indices()
                    else:
                        grad_batches = copy.deepcopy(batches)

                    # Use single optimizer and standard resolution pairs for gradient analysis
                    grad_optimizer = self.optimizer
                    grad_resolution_pairs = [
                        (self.c2f_resolutions[i], self.c2f_resolutions[i + 1])
                        for i in range(len(self.c2f_resolutions) - 1)
                    ]

                    # Collect per-resolution stats for decoding (if available)
                    train_means, train_stds = {}, {}
                    for res, ds in self.train_datasets.items():
                        if hasattr(ds, "output_mean") and hasattr(ds, "output_std"):
                            train_means[res] = ds.output_mean
                            train_stds[res] = ds.output_std

                    grad_metrics = self.grad_eval_fn(
                        self.model,
                        self.config,
                        self.criterion,
                        self.train_datasets,
                        train_means,
                        train_stds,
                        grad_resolution_pairs,
                        device,
                        grad_batches,
                        self.c2f_resolutions,
                        optimizer=grad_optimizer,
                    )

                    log_metrics.update(grad_metrics)
                    # except Exception as e:
                    #     print(f"Warning: gradient analysis failed: {e}")

                if spectral_diag is not None:
                    spectral_start = time.time()
                    train_mean_eval, train_std_eval = self.eval_norm_stats_for_logging()
                    spectral_diag.log_epoch(
                        self.model,
                        epoch,
                        train_mean_eval,
                        train_std_eval,
                        config,
                        phase_res=active_resolutions[-1] if active_resolutions else None,
                    )
                    eval_time_total += time.time() - spectral_start

                epoch_eval_block_time = time.time() - eval_start
                eval_times.append(epoch_eval_block_time)
                total_eval_time += epoch_eval_block_time
                log_metrics['epoch_eval_block_time'] = epoch_eval_block_time

                # Print summary
                print(f"\nEpoch {epoch+1}/{total_epochs}")
                print(f"  Train - Coarse: {coarse_loss:.6f}, Total: {total_loss:.6f}")
                if pair_losses:
                    for k, v in pair_losses.items():
                        print(f"  Train - Pair {k}: {v:.6f}")
                print(f"  Test Loss: {test_loss:.6f}")
                print(f"  Time - Train: {epoch_train_time:.2f}s, Eval: {epoch_eval_time:.2f}s")
            else:
                # No evaluation this epoch - just log training metrics
                print(f"\nEpoch {epoch+1}/{total_epochs} - Train Loss: {total_loss:.6f}")

            if (
                kernel_drift_diag is not None
                and kernel_drift_diag.should_log(epoch + 1)
            ):
                kernel_start = time.time()
                kernel_metrics = kernel_drift_diag.log_snapshot(
                    self.model,
                    epoch_completed=epoch + 1,
                    phase=phase,
                    phase_res=active_resolutions[-1] if active_resolutions else None,
                    cum_train_time=cum_train_time,
                )
                log_metrics.update({
                    f"kernel_drift/{k}": v
                    for k, v in kernel_metrics.items()
                    if isinstance(v, (int, float, np.integer, np.floating))
                })
                log_metrics["kernel_drift/elapsed_time"] = time.time() - kernel_start

            epoch_total_time = time.time() - epoch_start
            log_metrics['epoch_total_time'] = epoch_total_time
            log_metrics.setdefault('epoch_eval_block_time', eval_time_total)

            if self.use_wandb:
                wandb.log(log_metrics, step=epoch + 1)

            local_row = json_ready(log_metrics)
            history_rows.append(local_row)
            if "test_loss" in local_row:
                eval_rows.append(local_row)

            if config['load_gpu_epoch']:
                self.unload_epoch_indices_from_gpu(self.train_datasets, idxs_per_res)


        # Print final summary
        print("\n" + "="*80)
        print("Training Complete")
        print("="*80)
        print(f"Total Training Time: {total_train_time:.2f}s")
        if eval_times:
            print(f"Total Evaluation Time: {total_eval_time:.2f}s")
        reported_best = (
            best_test_loss
            if np.isfinite(best_test_loss)
            else (last_test_loss if last_test_loss is not None else 0.0)
        )
        print(f"Best Test Loss: {reported_best:.6f}")
        print(f"Final Train Loss: {train_losses[-1] if train_losses else 0.0:.6f}")

        if spectral_diag is not None and spectral_diag.epochs:
            spectral_diag.save_and_plot()

        if kernel_drift_diag is not None:
            kernel_drift_diag.save_summary()

        out_dir = config.get("out_dir")
        if config.get("save_final_checkpoint", False) and out_dir:
            os.makedirs(out_dir, exist_ok=True)
            checkpoint = {
                "epoch_completed": total_epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "config": json_ready(config),
                "metrics": {
                    "best_test_loss": reported_best,
                    "final_test_loss": last_test_loss,
                    "final_train_loss": train_losses[-1] if train_losses else 0.0,
                    "cum_train_time": cum_train_time,
                    "total_eval_time": total_eval_time,
                },
            }
            final_path = os.path.join(out_dir, "final_checkpoint.pt")
            torch.save(checkpoint, final_path)
            print(f"Saved final checkpoint to {final_path}")

        write_local_outputs(config, history_rows, eval_rows)

        if self.use_wandb:
            wandb.finish()

        return reported_best, train_losses

    def save_model(self, model, optimizer, epoch, config, metrics, is_best=False):
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
            checkpoint_path = os.path.join(checkpoint_dir,
                                           f'{checkpoint_name}_best.pth' if is_best else f'{checkpoint_name}_last.pth')
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
