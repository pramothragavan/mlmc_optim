"""Mixed-resolution data loader for multi-resolution training."""

import random
import torch

from mlmc_optim.batcher import MLMCBatcher


class MixedResolutionBatchSampler:
    """Samples batches where each batch contains data from a single randomly chosen resolution."""
    
    def __init__(self, datasets_by_res, batch_size, total_samples):
        """
        Args:
            datasets_by_res: Dict mapping resolution -> dataset
            batch_size: Number of samples per batch
            total_samples: Total number of samples to generate (determines epoch length)
        """
        self.datasets_by_res = datasets_by_res
        self.resolutions = list(datasets_by_res.keys())
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.num_batches = total_samples // batch_size
        
    def __iter__(self):
        """Yield batches, each from a randomly sampled resolution."""
        for _ in range(self.num_batches):
            # Randomly choose a resolution for this batch
            res = random.choice(self.resolutions)
            dataset = self.datasets_by_res[res]
            
            # Sample batch_size indices from this resolution's dataset
            indices = torch.randint(0, len(dataset), (self.batch_size,))
            
            yield res, indices.tolist()
    
    def __len__(self):
        return self.num_batches


class MixedResolutionDataLoader:
    """Data loader that yields batches from randomly or MLMC-scheduled resolutions."""
    
    def __init__(self, datasets_by_res, batch_size, total_samples, mode='uniform', config=None):
        """Args:
            datasets_by_res: Dict mapping resolution -> dataset
            batch_size: Number of samples per batch (for uniform mode)
            total_samples: Total number of samples to generate per epoch (uniform mode)
            mode: 'uniform' (random per-batch resolution) or 'mlmc_schedule'
            config: Full config dict (required for 'mlmc_schedule')
        """
        self.datasets_by_res = datasets_by_res
        self.batch_size = batch_size
        self.total_samples = total_samples
        self.resolutions = list(datasets_by_res.keys())
        self.mode = mode
        self.config = config

        if mode == 'uniform':
            self.num_batches = total_samples // batch_size
            self.batches = None
        elif mode == 'mlmc_schedule':
            if config is None:
                raise ValueError("config must be provided for mlmc_schedule mode")
            c2f_resolutions = self.config['c2f_resolutions']

            batch_size_base = self.config.get('batch_size', 32)
            # Treat explicit None the same as missing: default to geom_prog / geom_prog_batch
            mlmc_sampling_style = self.config.get('mlmc_sampling_style') or 'geom_prog'
            mlmc_batch_style = self.config.get('mlmc_batch_style') or 'geom_prog_batch'
            max_level = self.config.get('mlmc_max_level') if self.config.get('mlmc_max_level') is not None else len(c2f_resolutions)
            mlmc_data_size_multiplier = self.config.get('mlmc_data_size_multiplier', 2)
            mlmc_batch_size_multiplier = self.config.get('mlmc_batch_size_multiplier', 2)
            mlmc_samples_per_level = self.config.get('mlmc_samples_per_level', None)
            mlmc_batch_sizes = self.config.get('mlmc_batch_sizes', None)

            # Use only the finest max_level resolutions for the MLMC schedule
            active_resolutions = c2f_resolutions[-max_level:]
            coarsest_res = active_resolutions[0]
            total_samples_eff = len(self.datasets_by_res[coarsest_res])

            def get_sample_sizes():
                if mlmc_sampling_style == 'geom_prog':
                    total_mult = sum(mlmc_data_size_multiplier ** l for l in range(max_level))
                    base_samples = total_samples_eff // total_mult
                    return [
                        int(base_samples * (mlmc_data_size_multiplier ** (max_level - l)))
                        for l in range(1, max_level + 1)
                    ]
                elif mlmc_sampling_style == 'prescribed':
                    return mlmc_samples_per_level
                else:
                    raise ValueError(f"Unknown MLMC sampling style: {mlmc_sampling_style}")

            if mlmc_sampling_style in ['geom_prog', 'prescribed']:
                sample_sizes = get_sample_sizes()
            else:
                sample_sizes = mlmc_samples_per_level

            if mlmc_batch_style == 'geom_prog_batch':
                batch_sizes = [
                    int(batch_size_base * (mlmc_batch_size_multiplier ** i))
                    for i in range(max_level)
                ][::-1]
            elif mlmc_batch_style == 'prescribed_batch':
                batch_sizes = mlmc_batch_sizes
            else:
                batch_sizes = [batch_size_base] * max_level

            batcher = MLMCBatcher(
                self.config,
                active_resolutions,
                sample_sizes,
                batch_sizes,
                total_samples_eff,
                self.config.get('seed', None),
            )
            _, self.batches = batcher.get_epoch_indices()
            self.num_batches = len(self.batches)
        else:
            raise ValueError(f"Unknown MixedResolutionDataLoader mode: {mode}")
        
    def __iter__(self):
        """Yield batches from sampled resolutions."""
        if self.mode == 'uniform':
            for _ in range(self.num_batches):
                # Randomly choose a resolution for this batch
                res = random.choice(self.resolutions)
                dataset = self.datasets_by_res[res]
                
                # Sample batch_size indices from this resolution's dataset
                indices = torch.randint(0, len(dataset), (self.batch_size,))
                
                # Gather batch data
                batch_data = []
                batch_targets = []
                
                for idx in indices:
                    data, target = dataset[idx.item()]
                    batch_data.append(data)
                    batch_targets.append(target)
                
                yield (torch.stack(batch_data), torch.stack(batch_targets))
        else:  # mlmc_schedule
            for batch in self.batches:
                # Each batch is a list of [res_label, indices]
                for res_label, res_indices in batch:
                    res = res_label[-1]
                    dataset = self.datasets_by_res[res]
                    data, targets = dataset.get_items(res_indices)
                    yield (data, targets)
    
    def __len__(self):
        return self.num_batches
