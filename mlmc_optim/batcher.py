"""MLMC batching logic for multi-resolution training."""

import numpy as np
from typing import Dict, List, Tuple, Optional


class MLMCBatcher:
    """Creates batches for MLMC training with hierarchical or random pairing."""

    def __init__(self, config, c2f_resolutions, subset_sizes, batch_sizes, total_samples, seed=None):
        self.config = config
        self.total_samples = total_samples
        self.c2f_resolutions = c2f_resolutions
        self.f2c_resolutions = c2f_resolutions[::-1]
        self.coarsest_res = c2f_resolutions[0]
        self.subset_sizes = subset_sizes
        self.batch_sizes = batch_sizes
        self.use_deterministic = (
            self.config.get('mlmc_deterministic_batching', False) or
            (self.config.get('eval_grad_every') is not None and not self.config.get('eval_grad_rand', False))
        )

    def get_indices_per_resolution(self, batches: List[Dict]) -> Dict[int, np.ndarray]:
        """Extract all indices used per resolution."""

        idxs_per_res = {}
        for b_idx, batch in enumerate(batches):
            for res_level, (res_lable, res_idxs) in enumerate(batch):
                if b_idx == 0:
                    idxs_per_res[res_lable[-1]] = res_idxs
                else:
                    idxs_per_res[res_lable[-1]] = np.concatenate([idxs_per_res[res_lable[-1]], res_idxs])

        return idxs_per_res
    
    def get_epoch_indices(self):
        if self.use_deterministic:
            # Deterministic: use sequential indices
            coarsest_subset_idxs = np.arange(self.subset_sizes[0])
        else:
            # Random: sample fresh indices each epoch
            coarsest_subset_idxs = np.random.choice(self.total_samples, self.subset_sizes[0], replace=False)

        coarsest_res = self.c2f_resolutions[0]

        batches = []
        # first create the batches
        num_batches = int(np.ceil(max([ss / bs for ss, bs in zip(self.subset_sizes, self.batch_sizes)])))
        if self.config['mlmc_pairing'] in ['hierarchy', 'random_disjoint']:
            remaining_coarset_idxs = coarsest_subset_idxs.copy()

        samples_used = [0] * len(self.batch_sizes)

        for b_idx in range(num_batches):
            remaining_needed = self.subset_sizes[0] - samples_used[0]
            batch_size = min(remaining_needed, self.batch_sizes[0])

            if self.config['mlmc_pairing'] == 'hierarchy':
                if self.use_deterministic:
                    # Take deterministic batches during gradient analysis
                    # start_idx = 0#samples_used[0] #this caused so many problems
                    # coarse_batch_idxs = remaining_coarset_idxs[start_idx:start_idx + batch_size]
                    coarse_batch_idxs = remaining_coarset_idxs[:batch_size]
                    remaining_coarset_idxs = np.setdiff1d(remaining_coarset_idxs, coarse_batch_idxs)
                else:
                    coarse_batch_idxs = np.random.choice(remaining_coarset_idxs, batch_size, replace=False)
                    remaining_coarset_idxs = np.setdiff1d(remaining_coarset_idxs, coarse_batch_idxs)
            elif self.config['mlmc_pairing'] == 'random':
                coarse_batch_idxs = np.random.choice(self.total_samples, batch_size, replace=False)
            # elif self.config['mlmc_pairing'] == 'random_disjoint':
            #     coarse_batch_idxs = np.random.choice(remaining_coarset_idxs, batch_size, replace=False)
            #     remaining_coarset_idxs = np.setdiff1d(remaining_coarset_idxs, coarse_batch_idxs)
            # elif self.config['mlmc_pairing'] == 'random_disjoint':
            # elif self.config['mlmc_pairing'] == 'random_set_comp':
                # choose random idxs from the set complement the coarsest subset idxs
                # coarse_subset_idxs = np.random.choice(np.setdiff1d(np.arange(self.total_samples), coarse_subset_idxs), self.subset_sizes[idx+1], replace=False)

            samples_used[0] += len(coarse_batch_idxs)
            batch = [[(coarsest_res,), coarse_batch_idxs]]

            for idx, res in enumerate(self.c2f_resolutions[1:]):
                res_label = (self.c2f_resolutions[idx], res)
                remaining_needed = self.subset_sizes[idx + 1] - samples_used[idx + 1]
                batch_size = min(remaining_needed, self.batch_sizes[idx + 1], len(coarse_batch_idxs))
                if self.config['mlmc_pairing'] == 'hierarchy':
                    coarse_batch_idxs = coarse_batch_idxs[:batch_size]
                elif self.config['mlmc_pairing'] == 'random':
                    if self.use_deterministic:
                        start_idx = samples_used[0]
                        coarse_batch_idxs = np.arange(start_idx, start_idx + batch_size) % self.total_samples
                    else:
                        coarse_batch_idxs = np.random.choice(self.total_samples, batch_size, replace=False)
                # elif self.config['mlmc_pairing'] == 'random_disjoint':
                #     coarse_batch_idxs = np.random.choice(remaining_coarset_idxs, batch_size, replace=False)
                samples_used[idx + 1] += len(coarse_batch_idxs)
                batch.append([res_label, coarse_batch_idxs])

            batches.append(batch)

        idxs_per_res = self.get_indices_per_resolution(batches)

        return idxs_per_res, batches
