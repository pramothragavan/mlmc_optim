
import numpy as np
import torch

from examples.examples_src.utils.data_utils import decode


def safe_norm(x, dim=None):
    """Complex-safe vector norm using torch.linalg.vector_norm.

    For complex tensors, view as real with an extra dimension of size 2 and
    compute the Euclidean norm over that real representation.
    """
    if torch.is_complex(x):
        x_real = torch.view_as_real(x)  # (..., 2)
        return torch.linalg.vector_norm(x_real, dim=dim)
    return torch.linalg.vector_norm(x, dim=dim)

def convert_optimizer_grads(model, optimizer=None, resolution=None, mode='raw'):
    """
    Convert raw gradients to effective gradients based on the optimizer type.

    Args:
        model: The model whose gradients we're converting
        optimizer: The optimizer instance or dictionary of optimizers (if None, returns raw gradients)
        resolution: The resolution level (only used when optimizer is a dictionary)

    Returns:
        Flattened gradient vector accounting for optimizer adjustments
    """
    # Get all parameters and their gradients
    all_params = list(model.parameters())
    raw_grads = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
                 for p in all_params]

    # For other optimizers, fallback to raw gradients
    if mode == 'raw':
        return torch.cat([g.flatten() for g in raw_grads])

    # If no optimizer provided or it's SGD, just return the raw gradients
    elif optimizer is None or isinstance(optimizer, torch.optim.SGD):
        # multiply by current learning rate after scheduler step
        # Get the current learning rate from the optimizer
        lr = optimizer.param_groups[0]['lr']
        scaled_grads = [g * lr for g in raw_grads]
        return torch.cat([g.flatten() for g in scaled_grads])

    # For Adam, apply momentum and velocity adjustments
    elif isinstance(optimizer, torch.optim.Adam):
        # Get actual Adam hyperparameters from the optimizer
        beta1, beta2 = optimizer.defaults['betas']
        eps = optimizer.defaults['eps']
        # Get the current learning rate from the optimizer
        lr = optimizer.param_groups[0]['lr']

        # We need to simulate Adam's state for each parameter
        adam_adjusted_grads = []

        for param_idx, (param, raw_grad) in enumerate(zip(all_params, raw_grads)):
            # Try to get existing state from optimizer, or initialize if not present
            if param in optimizer.state:
                # Extract existing state
                state = optimizer.state[param]
                exp_avg = state['exp_avg']  # momentum (m)
                exp_avg_sq = state['exp_avg_sq']  # velocity (v)
                step_t = state['step']
            else:
                # Initialize state as Adam would
                exp_avg = torch.zeros_like(param)
                exp_avg_sq = torch.zeros_like(param)
                step_t = 0

            # Update momentum and velocity as Adam would
            exp_avg_updated = beta1 * exp_avg + (1 - beta1) * raw_grad
            exp_avg_sq_updated = beta2 * exp_avg_sq + (1 - beta2) * (raw_grad * raw_grad)

            # Bias correction
            bias_correction1 = 1 - beta1 ** (step_t + 1)
            bias_correction2 = 1 - beta2 ** (step_t + 1)

            # Calculate Adam's effective gradient (what would actually be used for the update)
            adam_grad = exp_avg_updated / bias_correction1 / (torch.sqrt(exp_avg_sq_updated / bias_correction2) + eps)

            # Scale by learning rate just like we do for SGD
            adam_grad = adam_grad * lr

            adam_adjusted_grads.append(adam_grad.flatten())
        # Combine all adjusted gradients
        return torch.cat(adam_adjusted_grads)
    else:
        # For other optimizers, fallback to raw gradients
        print(f"Warning: Unsupported optimizer type: {type(optimizer).__name__}. Using raw gradients.")
        return torch.cat([g.flatten() for g in raw_grads])


def evaluate_gradient_differences(model, config, criterion, train_datasets, train_means, train_stds, resolution_pairs, device, batches, resolutions, optimizer=None):
    # Limit number of batches if specified in config
    max_batches = config.get('eval_grad_batches', None)
    if max_batches is not None:
        print(f"\nUsing first {max_batches} batches for gradient analysis (out of {len(batches)} total)")
        batches = batches[:max_batches]
    # Get optimizer type from config
    optimizer_type = config.get('optimizer', 'sgd')
    """
    1) Norm of the expectations - [batches, params]
    2) ###COMMENT OUT for now Compute gradients over full dataset for each resolution
    3) For each batch: Get finest resolution gradient using coarsest indices
    4) For each resolution: Compute gradient using same coarsest indices
    5) Compare against: a. Batch finest gradient ###COMMENT OUT b. Full data finest gradient
    6) Telescopic sum analysis: Method 1 (grad diff), Method 2 (loss diff grad)
    7) Compare both against: a. Batch finest gradient ###COMMENT OUT b. Full data finest gradient
    8) Log averages at the end
    """
    model.train()

    # Helper function for proper cosine similarity calculation including complex vectors
    def cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors, handling complex values properly.
        For complex vectors, uses the Hermitian inner product and returns magnitude.
        Returns a real value between 0 and 1.
        """
        if torch.is_complex(vec1) or torch.is_complex(vec2):
            # Use Hermitian inner product (complex conjugate of second vector)
            dot_product = torch.sum(vec1 * torch.conj(vec2))
            # Take magnitude of complex result and convert to real tensor
            dot_product = torch.abs(dot_product).real
        else:
            dot_product = torch.dot(vec1, vec2)

        norm1 = safe_norm(vec1)
        norm2 = safe_norm(vec2)

        # Ensure final result is a real tensor
        result = dot_product / (norm1 * norm2 + 1e-12)

        # Explicitly cast to float
        if torch.is_complex(result):
            result = result.real

        return result

    grad_metrics = {}

    # Create a mapping from resolution to optimizer key
    res_key_lookup = {}
    # For each resolution pair
    for (coarse_res, fine_res) in resolution_pairs:
        res_key_lookup[coarse_res] = f"{fine_res}_{coarse_res}"
        res_key_lookup[fine_res] = f"{fine_res}_{coarse_res}"
    # Add special key for coarsest resolution
    coarsest_res = resolutions[0]
    finest_res = resolutions[-1]
    res_key_lookup[coarsest_res] = "coarse"

    # Initialize batch_grad_metrics structure with nested dictionaries/lists
    batch_grad_metrics = {
        'norm_mean_samp': {},  # Norm of mean gradient per resolution pair
        'norm_std_samp': {},   # Norm of std gradient per resolution pair
        'mean_norm': {},       # Mean gradient norm per resolution pair
        'std_norm': {},        # Std of gradient norm per resolution pair
        'mean_of_means': {}    # Mean of mean gradient components
    }

    # Initialize lists for each resolution pair and metric type
    for metric_name in batch_grad_metrics:
        batch_grad_metrics[metric_name] = {}

    # Step 1: Norm of the expectations [batches, params]
    print("\nStep 1: Norm of the expectations of corrections [batches, params]")
    # Process each batch
    for batch_idx, batch in enumerate(batches):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Reached batch limit ({max_batches}), stopping batch analysis")
            break
        # batch_indices = batch[0]  # Coarse indices
        batch_corrections = batch[1:]  # List of (resolution pair, indices) for corrections

        # Process each resolution pair
        for (coarse_res, fine_res), correction_indices in batch_corrections:
            pair_key = f"{fine_res}_{coarse_res}"

            # Initialize lists for this pair if needed
            for metric_name in batch_grad_metrics:
                if pair_key not in batch_grad_metrics[metric_name]:
                    batch_grad_metrics[metric_name][pair_key] = []

            # Get data for both resolutions using exact batch indices
            fine_data, fine_target = train_datasets[fine_res].get_items(correction_indices)
            coarse_data, coarse_target = train_datasets[coarse_res].get_items(correction_indices)

            if len(fine_data) == 0 or len(coarse_data) == 0:
                continue

            fine_data = fine_data.to(device)
            fine_target = fine_target.to(device)
            coarse_data = coarse_data.to(device)
            coarse_target = coarse_target.to(device)
            if config['dataset'] == 'FlowPastCylinder':
                batch_size = fine_data.batch_size
            else:
                batch_size = fine_data.shape[0]

            # Forward pass at fine resolution
            model.zero_grad()
            # Ensure data is on correct device
            fine_data = fine_data.to(device)
            fine_target = fine_target.to(device)
            output_fine = model(fine_data)
            if config['normalize']:
                output_fine = decode(output_fine, train_means[fine_res], train_stds[fine_res])
                target_fine = decode(fine_target, train_means[fine_res], train_stds[fine_res])
            else:
                target_fine = fine_target

            # Compute per-sample gradients for fine resolution
            fine_grads = []
            for i in range(batch_size):
                model.zero_grad()
                sample_output = output_fine[i: i +1]
                sample_target = target_fine[i: i +1]
                sample_loss = criterion(sample_output, sample_target)
                if config.get('loss_reduction') == 'sum':
                    sample_loss = sample_loss.sum()
                else:
                    sample_loss = sample_loss.mean()
                sample_loss.backward(retain_graph=True)
                # Use convert_optimizer_grads to get optimizer-adjusted gradients
                flattened_grads = convert_optimizer_grads(model, optimizer, res_key_lookup[fine_res], config['eval_grad_mode'])
                fine_grads.append(flattened_grads)

            # Forward pass at coarse resolution
            model.zero_grad()
            # Ensure data is on correct device
            coarse_data = coarse_data.to(device)
            coarse_target = coarse_target.to(device)
            output_coarse = model(coarse_data)
            if config['normalize']:
                output_coarse = decode(output_coarse, train_means[fine_res], train_stds[fine_res])
                target_coarse = decode(coarse_target, train_means[fine_res], train_stds[fine_res])
            else:
                target_coarse = coarse_target

            # Compute per-sample gradients for coarse resolution
            coarse_grads = []
            for i in range(batch_size):
                model.zero_grad()
                sample_output = output_coarse[i: i +1]
                sample_target = target_coarse[i: i +1]
                sample_loss = criterion(sample_output, sample_target)
                if config['loss_reduction'] == 'sum':
                    sample_loss = sample_loss.sum()
                else:
                    sample_loss = sample_loss.mean()
                sample_loss.backward(retain_graph=True)
                # Use convert_optimizer_grads to get optimizer-adjusted gradients
                flattened_grads = convert_optimizer_grads(model, optimizer, res_key_lookup[fine_res], config['eval_grad_mode'])
                coarse_grads.append(flattened_grads)

            # Stack all samples for efficient computation
            fine_grads = torch.stack(fine_grads)    # [batch_size, total_params]
            coarse_grads = torch.stack(coarse_grads)  # [batch_size, total_params]

            # Compute correction for all parameters at once
            correction = fine_grads - coarse_grads  # [batch_size, total_params]

            # 1. Statistics over parameters of the mean/std over samples
            mean_over_samples = torch.mean(correction, dim=0)  # [total_params]
            std_over_samples = torch.std(correction, dim=0)   # [total_params]
            # useful ones
            norm_mean_samples = safe_norm(mean_over_samples)
            norm_std_samples = safe_norm(std_over_samples)

            mean_p_mean_s = torch.mean(mean_over_samples)
            std_p_mean_s = torch.std(mean_over_samples)
            mean_p_std_s = torch.mean(std_over_samples)
            std_p_std_s = torch.std(std_over_samples)

            # 2. Compute norms and stats of gradient differences
            correction_norms = safe_norm(correction, dim=1)  # [batch_size]
            correction_norm_mean = torch.mean(correction_norms)
            correction_norm_std = torch.std(correction_norms)

            # Print statistics for this resolution pair
            print(f"\nResolution pair {pair_key}:")
            print(f"  Parameter Statistics:")
            print(f"    Mean of means: {mean_p_mean_s.item():.3g}")
            print(f"    Std of means: {std_p_mean_s.item():.3g}")
            print(f"    Mean of stds: {mean_p_std_s.item():.3g}")
            print(f"    Std of stds: {std_p_std_s.item():.3g}")
            print(f"  Norm Expectation Statistics:")
            print(f"    Norm_mean_samples: {norm_mean_samples.item():.3g}")
            print(f"    Norm_std_samples: {norm_std_samples.item():.3g}")
            print(f"  Gradient Difference Norms:")
            print(f"    Mean norm: {correction_norm_mean.item():.3g}")
            print(f"    Norm std: {correction_norm_std.item():.3g}")

            # Append this batch's metrics to the structured batch_grad_metrics
            batch_grad_metrics['norm_mean_samp'][pair_key].append(norm_mean_samples.item())
            batch_grad_metrics['norm_std_samp'][pair_key].append(norm_std_samples.item())
            batch_grad_metrics['mean_norm'][pair_key].append(correction_norm_mean.item())
            batch_grad_metrics['std_norm'][pair_key].append(correction_norm_std.item())
            batch_grad_metrics['mean_of_means'][pair_key].append(mean_p_mean_s.item())

    # Display summary statistics table for gradient metrics
    print("\nStep 1 Summary: Gradient Metrics Across All Batches")

    # Use the original ordered resolution_pairs to maintain coarse-to-fine ordering
    # Convert resolution pairs to a wandb-friendly format without special characters
    ordered_pair_keys = [f"{fine_res}_{coarse_res}" for coarse_res, fine_res in resolution_pairs]

    # Print header
    print("    Pair   | Norm(E[Δg]) | Norm(std[Δg]) | E[||Δg||] | std[||Δg||] | Var[||Δg||] | Mean of means")
    print("    " + "-" * 86)

    # Print metrics for each pair and transfer to grad_metrics
    for pair_key in ordered_pair_keys:
        # Only process if we have data for this pair
        if pair_key in batch_grad_metrics['norm_mean_samp'] and len(batch_grad_metrics['norm_mean_samp'][pair_key]) > 0:
            # Calculate average metrics
            norm_mean = np.mean(batch_grad_metrics['norm_mean_samp'][pair_key])
            norm_std = np.mean(batch_grad_metrics['norm_std_samp'][pair_key])
            # E[||Δg||] and its std across batches
            mean_norm = np.mean(batch_grad_metrics['mean_norm'][pair_key])
            std_norm = np.mean(batch_grad_metrics['std_norm'][pair_key])
            var_norm = std_norm ** 2
            mean_means = np.mean(batch_grad_metrics['mean_of_means'][pair_key])

            # Store in grad_metrics for return
            grad_metrics[f"{pair_key}_norm_mean"] = norm_mean
            grad_metrics[f"{pair_key}_norm_std"] = norm_std
            # Aliases that make the MLMC interpretation explicit
            grad_metrics[f"{pair_key}_E_norm_diff"] = mean_norm
            grad_metrics[f"{pair_key}_std_norm_diff"] = std_norm
            grad_metrics[f"{pair_key}_var_norm_diff"] = var_norm
            grad_metrics[f"{pair_key}_mean_norm"] = mean_norm
            grad_metrics[f"{pair_key}_std_norm"] = std_norm

            # Handle potential complex values in mean_of_means
            if isinstance(mean_means, complex):
                # Take the magnitude of the complex value
                mean_means = abs(mean_means)
            grad_metrics[f"{pair_key}_mean_of_means"] = mean_means

            # Print row in table
            print \
                (f"    {pair_key:7} | {norm_mean:.6f} | {norm_std:.6f} | {mean_norm:.6f} | {std_norm:.6f} | {var_norm:.6f} | {mean_means:.10f}")

    # delete tensors from step 1 to save memory if they exist
    torch.cuda.empty_cache()
    for var in ['fine_grads', 'coarse_grads', 'correction', 'flattened_grads']:
        if var in locals():
            del locals()[var]

    # Step 2: Full(/some batches) data gradients for each resolution
    if False:
        print("\nStep 2: Full data gradients for each resolution")
        full_grads = {}
        full_grad_norms = {}
        for res in resolutions:
            print(f"  Resolution {res}...")
            # do an initial forward pass to setup the parameter gradients
            n_samples = len(train_datasets[res])
            model.zero_grad()
            data, targets = train_datasets[res].get_items([0])
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            if config['normalize']:
                output = decode(output, train_means[res], train_stds[res])
                targets = decode(targets, train_means[res], train_stds[res])
            if config['loss_type'] == 'L2':
                loss = criterion(output, targets, train_datasets[res].mass_matrix)
            else:
                loss = criterion(output, targets)
            if config.get('loss_reduction') == 'sum':
                loss = loss.sum()
            else:
                loss = loss.mean()
            loss.backward()

            res_grads = convert_optimizer_grads(model, optimizer, res_key_lookup[res], config['eval_grad_mode'])

            # for i in range(1, n_samples):
            i = 0
            for batch_idx, batch in enumerate(batches):
                if max_batches is not None and batch_idx >= max_batches:
                    print(f"Reached batch limit ({max_batches}), stopping batch analysis")
                    break
                for _ in batch[0][0]:
                    model.zero_grad()
                    data, targets = train_datasets[res].get_items([i])
                    data, targets = data.to(device), targets.to(device)
                    output = model(data)
                    if config['normalize']:
                        output = decode(output, train_means[res], train_stds[res])
                        targets = decode(targets, train_means[res], train_stds[res])
                    if config['loss_type'] == 'L2':
                        loss = criterion(output, targets, train_datasets[res].mass_matrix)
                    else:
                        loss = criterion(output, targets)
                    if config['loss_reduction'] == 'sum':
                        loss = loss.sum()
                    else:
                        loss = loss.mean()
                    loss.backward(retain_graph=True)

                    # Use the same all_params list to ensure index consistency
                    new_grads = convert_optimizer_grads(model, optimizer, res_key_lookup[res], config['eval_grad_mode'])
                    for j, p in enumerate(new_grads):
                        if p.grad is not None:
                            # When a parameter has a gradient, update the running average
                            res_grads[j] += (new_grads[j] - res_grads[j]) / (i + 1)
                    i += 1

            full_grads[res] = torch.cat([g.flatten() for g in res_grads])
            full_grad_norms[res] = safe_norm(full_grads[res])
            model.zero_grad()
        full_fine_grad = full_grads[finest_res].clone()
        full_grad_norm = full_grad_norms[finest_res].clone()
        print("  ...done.")

        # Print relative errors of full gradients vs finest
        print("\nFull Data Gradient Relative Errors vs Finest:")
        for res in resolutions[:-1]:
            rel_err = safe_norm(full_grads[res] - full_fine_grad) / safe_norm(full_fine_grad)
            print(f"  Resolution {res}: {rel_err:.6f}")
            grad_metrics[f"full_rel_err_{res}"] = rel_err.item()

        # delete full_grads and full_grad_norms to save memory
        # del full_grads, full_grad_norms, res_grads, new_grads, mean_over_samples, std_over_samples
        for var in ['full_grads', 'full_grad_norms', 'res_grads', 'new_grads', 'mean_over_samples', 'std_over_samples']:
            if var in locals():
                del locals()[var]
        torch.cuda.empty_cache()

    # Step 3-7: Per-batch analysis
    print("\nStep 3-7: Per-batch analysis")
    # Initialize metrics to track
    batch_metrics = {
        'direct_grad_norms': {res: [] for res in resolutions},
        'abs_direct_vs_batch': {res: [] for res in resolutions},
        'rel_direct_vs_batch': {res: [] for res in resolutions},
        'abs_direct_vs_full': {res: [] for res in resolutions},
        'rel_direct_vs_full': {res: [] for res in resolutions},
        'tel_loss': [],

        # Progressive errors for Method 2 - track each step
        'tel': {
            'grad_norms': {},  # GradNorm(batch) at each step
            'abs_err_batch': {},   # AbsErr(batch) at each step
            'abs_err_n_batch': {},  # AbsErr(N-batch) at each step
            'rel_err_batch': {},    # RelErr(batch) at each step
            'rel_err_full': {},     # RelErr(full) at each step
            'rel_err_n_batch': {},  # RelErr(N-batch) at each step
            'rel_err_n_full': {},   # RelErr(N-full) at each step
            'cos_batch': {},        # CosSim(batch) at each step
            'cos_full': {},         # CosSim(full) at each step
            'cos_n_batch': {},      # CosSim(N-batch) at each step
            'cos_n_full': {}        # CosSim(N-full) at each step
        }
    }

    # Initialize lists for each step and metric type
    for metric_name in batch_metrics['tel']:
        # Create empty dictionaries for each metric
        batch_metrics['tel'][metric_name] = {}
        # Create empty list for each step (0 to len(resolutions)-1)
        for step in range(len(resolutions)):
            batch_metrics['tel'][metric_name][step] = []

    for batch_idx, batch in enumerate(batches):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"Reached batch limit ({max_batches}), stopping telescopic analysis")
            break

        batch_indices = batch[0][1]
        print(f"\nBatch {batch_idx +1}/{len(batches)} | Indices: {batch_indices}")

        # Step 3: Get finest resolution gradient using coarsest indices
        data, targets = train_datasets[finest_res].get_items(batch_indices)
        data, targets = data.to(device), targets.to(device)
        model.zero_grad()
        output = model(data)
        if config['normalize']:
            output = decode(output, train_means[finest_res], train_stds[finest_res])
            targets = decode(targets, train_means[finest_res], train_stds[finest_res])
        if config['loss_type'] == 'L2':
            loss = criterion(output, targets, train_datasets[finest_res].mass_matrix)
        else:
            loss = criterion(output, targets)
        if config.get('loss_reduction') == 'sum':
            loss = loss.sum()
        else:
            loss = loss.mean()
        loss.backward()
        # Handle None gradients with zeros when computing batch_fine_grad
        batch_fine_grad = convert_optimizer_grads(model, optimizer, res_key_lookup[finest_res], config['eval_grad_mode'])
        fine_grad_norm = safe_norm(batch_fine_grad)
        print(f"  Norm of batch finest grad: {fine_grad_norm:.6f}")
        # print(f"  Norm of full data finest grad: {full_grad_norm:.6f}")
        model.zero_grad()

        # Step 4: Compute gradients for each resolution using same indices
        batch_grads = {}
        for res in resolutions:
            data, targets = train_datasets[res].get_items(batch_indices)
            data, targets = data.to(device), targets.to(device)
            model.zero_grad()
            output = model(data)
            if config.get('normalize', False):
                output = decode(output, train_means[res], train_stds[res])
                targets = decode(targets, train_means[res], train_stds[res])
            if config['loss_type'] == 'L2':
                loss = criterion(output, targets, train_datasets[res].mass_matrix)
            else:
                loss = criterion(output, targets)
            if config['loss_reduction'] == 'sum':
                loss = loss.sum()
            else:
                loss = loss.mean()
            loss.backward(retain_graph=True)
            # Use optimizer-adjusted gradients for batch_grads[res]
            batch_grads[res] = convert_optimizer_grads(model, optimizer, res_key_lookup[res], config['eval_grad_mode'])
            grad_norm = safe_norm(batch_grads[res])
            batch_metrics['direct_grad_norms'][res].append(grad_norm.item())
            print(f"    Resolution {res} batch grad norm: {grad_norm:.6f}")
            model.zero_grad()

        # Step 5: Compare each res against batch/full finest grad
        def normalize(x):
            return x / (safe_norm(x) + 1e-12)
        print("  Direct Resolution Comparisons:")

        # print \
        #     ("    Res | AbsErr(batch) | AbsErr(full) | RelErr(batch) | RelErr(full) | RelErr(N-batch) | RelErr(N-full) | CosSim(batch) | CosSim(full) | CosSim(N-batch) | CosSim(N-full)")
        print("    Res | AbsErr(batch) | RelErr(batch) | RelErr(N-batch) | CosSim(batch) | CosSim(N-batch)")


        for res in resolutions[:-1]:
            # Relative errors
            abs_batch_error = safe_norm(batch_grads[res] - batch_fine_grad)
            rel_batch_error = abs_batch_error / fine_grad_norm
            # Normalized relative errors
            batch_error_normed = safe_norm(normalize(batch_grads[res]) - normalize(batch_fine_grad)) / safe_norm(
                normalize(batch_fine_grad)
            )
            # Cosine similarities
            # Use the safe cosine_similarity function that handles complex values
            cos_batch = cosine_similarity(batch_grads[res], batch_fine_grad)
            cos_batch_normed = cosine_similarity(normalize(batch_grads[res]), normalize(batch_fine_grad))
            # print \
            #     (f"    {res:3} |   {abs_batch_error:.6f}   |   {abs_full_error:.6f}   |   {rel_batch_error:.6f}   |   {rel_full_error:.6f}   |   {batch_error_normed:.6f}   |   {full_error_normed:.6f}   |  {cos_batch:.6f}   |   {cos_full:.6f}   |   {cos_batch_normed:.6f}   |   {cos_full_normed:.6f}")
            print \
                (f"    {res:3} |   {abs_batch_error:.6f}   |   {rel_batch_error:.6f}   |   {batch_error_normed:.6f}   |  {cos_batch:.6f}   |   {cos_batch_normed:.6f}")
            batch_metrics['abs_direct_vs_batch'][res].append(abs_batch_error.item())
            batch_metrics['rel_direct_vs_batch'][res].append(rel_batch_error.item())
            if False:
                abs_full_error = safe_norm(batch_grads[res] - full_fine_grad)
                rel_full_error = abs_full_error / full_grad_norm
                full_error_normed = safe_norm(normalize(batch_grads[res]) - normalize(full_fine_grad)) / safe_norm(
                    normalize(full_fine_grad)
                )
                cos_full = cosine_similarity(batch_grads[res], full_fine_grad)
                cos_full_normed = cosine_similarity(normalize(batch_grads[res]), normalize(full_fine_grad))
                batch_metrics['abs_direct_vs_full'][res].append(abs_full_error.item())
                batch_metrics['rel_direct_vs_full'][res].append(rel_full_error.item())
                print \
                    (f"    {res:3} |   {abs_full_error:.6f}   |   {rel_full_error:.6f}   |   {full_error_normed:.6f}   |   {cos_full:.6f}   |   {cos_full_normed:.6f}")

        # Step 6: Telescopic sum analysis
        # todo this is not used in practice
        # Method 1: Grad diff (progressive table per batch)
        # tel_sum = batch_grads[coarsest_res].clone()
        # print("  Telescopic sum progressive errors (Method 1: grad diff):")
        # print("    Step | RelErr(batch) | RelErr(full) | RelErr(N-batch) | RelErr(N-full) | CosSim(batch) | CosSim(full) | CosSim(N-batch) | CosSim(N-full)")
        # for i in range(len(resolutions)-1):
        #     coarse_res = resolutions[i]
        #     fine_res = resolutions[i+1]
        #     tel_sum += batch_grads[fine_res] - batch_grads[coarse_res]
        #     tel_error_batch = safe_norm(tel_sum - batch_fine_grad) / fine_grad_norm
        #     tel_error_full = safe_norm(tel_sum - full_fine_grad) / full_grad_norm
        #     tel_error_batch_normed = safe_norm(normalize(tel_sum) - normalize(batch_fine_grad)) / safe_norm(normalize(batch_fine_grad))
        #     tel_error_full_normed = safe_norm(normalize(tel_sum) - normalize(full_fine_grad)) / safe_norm(normalize(full_fine_grad))
        #     # Use the safe cosine_similarity function that handles complex values
        #     cos_batch = cosine_similarity(tel_sum, batch_fine_grad)
        #     cos_full = cosine_similarity(tel_sum, full_fine_grad)
        #     cos_batch_normed = cosine_similarity(normalize(tel_sum), normalize(batch_fine_grad))
        #     cos_full_normed = cosine_similarity(normalize(tel_sum), normalize(full_fine_grad))
        #     print(f"    {i+1:4} |   {tel_error_batch:.6f}   |   {tel_error_full:.6f}   |   {tel_error_batch_normed:.6f}   |   {tel_error_full_normed:.6f}   |   {cos_batch:.6f}   |   {cos_full:.6f}   |   {cos_batch_normed:.6f}   |   {cos_full_normed:.6f}")

        # Method 2: Loss diff grad (reset and accumulate per batch, print progressive table)
        print("  Telescopic sum progressive errors (Method 2: diff loss grad):")
        # print("    Step | RelErr(batch) | RelErr(full) | RelErr(N-batch) | RelErr(N-full) | CosSim(batch) | CosSim(full) | CosSim(N-batch) | CosSim(N-full)")
        # print \
        #     ("    Step |   AbsErr(batch)   |   AbsErr(full)   |   RelErr(batch)   |   RelErr(full)   |   RelErr(N-batch)   |   RelErr(N-full)   |   CosSim(batch)   |   CosSim(full)   |   CosSim(N-batch)   |   CosSim(N-full)")
        print \
            ("    Step |   AbsErr(batch)   |   RelErr(batch)   |   RelErr(N-batch)   |   CosSim(batch)   |   CosSim(N-batch)")

        # 1. Start with the coarsest resolution gradient (already computed in batch_grads)
        coarsest_res = resolutions[0]
        tel_grad = batch_grads[coarsest_res].clone()

        # Compute and print initial metrics for coarsest resolution
        tel_error_loss_batch = safe_norm(tel_grad - batch_fine_grad)
        tel_rel_error_loss_batch = tel_error_loss_batch / fine_grad_norm
        tel_error_loss_batch_normed = safe_norm(normalize(tel_grad) - normalize(batch_fine_grad))
        tel_rel_error_loss_batch_normed = tel_error_loss_batch_normed / safe_norm(normalize(batch_fine_grad))
        # Use proper cosine similarity that handles complex vectors
        cos_loss_batch = cosine_similarity(tel_grad, batch_fine_grad)
        cos_loss_batch_normed = cosine_similarity(normalize(tel_grad), normalize(batch_fine_grad))
        print \
            (f"    {0:4} |   {tel_error_loss_batch:.6f}   |   {tel_rel_error_loss_batch:.6f}   |   {tel_error_loss_batch_normed:.6f}   |   {cos_loss_batch:.6f}   |   {cos_loss_batch_normed:.6f}")

        # Log metrics for step 0 (coarsest resolution)
        batch_metrics['tel']['grad_norms'][0].append(safe_norm(tel_grad).item())
        batch_metrics['tel']['abs_err_batch'][0].append(tel_error_loss_batch.item())
        batch_metrics['tel']['abs_err_n_batch'][0].append(tel_error_loss_batch_normed.item())
        batch_metrics['tel']['rel_err_batch'][0].append(tel_rel_error_loss_batch.item())
        batch_metrics['tel']['rel_err_n_batch'][0].append(tel_rel_error_loss_batch_normed.item())
        batch_metrics['tel']['cos_n_batch'][0].append(cos_loss_batch_normed.item())
        batch_metrics['tel']['cos_batch'][0].append(cos_loss_batch.item())

        if False:
            tel_error_loss_full = torch.norm(tel_grad - full_fine_grad)
            tel_rel_error_loss_full = tel_error_loss_full / full_grad_norm
            tel_error_loss_full_normed = torch.norm(normalize(tel_grad) - normalize(full_fine_grad))
            tel_rel_error_loss_full_normed = tel_error_loss_full_normed / torch.norm(normalize(full_fine_grad))

            cos_loss_full = cosine_similarity(tel_grad, full_fine_grad)
            cos_loss_full_normed = cosine_similarity(normalize(tel_grad), normalize(full_fine_grad))
            print \
                (f"    {0:4} |   {tel_error_loss_full:.6f}   |   {tel_rel_error_loss_full:.6f}   |   {tel_error_loss_full_normed:.6f}   |   {cos_loss_full:.6f}   |   {cos_loss_full_normed:.6f}")

            batch_metrics['tel']['rel_err_full'][0].append(tel_rel_error_loss_full.item())
            batch_metrics['tel']['rel_err_n_full'][0].append(tel_rel_error_loss_full_normed.item())
            batch_metrics['tel']['cos_full'][0].append(cos_loss_full.item())
            batch_metrics['tel']['cos_n_full'][0].append(cos_loss_full_normed.item())

        # 2. Then add the correction terms as before
        batch_corrections = batch[1:]
        for i, ((coarse_res, fine_res), correction_indices) in enumerate(batch_corrections):
            # For each step, accumulate loss diff grad using actual correction_indices
            # Fine
            fine_data, fine_targets = train_datasets[fine_res].get_items(correction_indices)
            fine_data, fine_targets = fine_data.to(device), fine_targets.to(device)
            output_fine = model(fine_data)
            if config['normalize']:
                output_fine = decode(output_fine, train_means[fine_res], train_stds[fine_res])
                targets_fine = decode(fine_targets, train_means[fine_res], train_stds[fine_res])
            else:
                targets_fine = fine_targets
            if config['loss_type'] == 'L2':
                loss_fine = criterion(output_fine, targets_fine, train_datasets[fine_res].mass_matrix)
            else:
                loss_fine = criterion(output_fine, targets_fine)
            # Coarse
            coarse_data, coarse_targets = train_datasets[coarse_res].get_items(correction_indices)
            coarse_data, coarse_targets = coarse_data.to(device), coarse_targets.to(device)
            output_coarse = model(coarse_data)
            if config['normalize']:
                output_coarse = decode(output_coarse, train_means[coarse_res], train_stds[coarse_res])
                targets_coarse = decode(coarse_targets, train_means[coarse_res], train_stds[coarse_res])
            else:
                targets_coarse = coarse_targets
            if config['loss_type'] == 'L2':
                loss_coarse = criterion(output_coarse, targets_coarse, train_datasets[coarse_res].mass_matrix)
            else:
                loss_coarse = criterion(output_coarse, targets_coarse)
            model.zero_grad()
            # Diff
            loss_diff = loss_fine - loss_coarse
            if config['loss_reduction'] == 'sum':
                loss_diff = loss_diff.sum()
            else:
                loss_diff = loss_diff.mean()

            loss_diff.backward(retain_graph=True)
            # Use optimizer-adjusted gradients for telescopic sum
            tel_grad += convert_optimizer_grads(model, optimizer, res_key_lookup[fine_res], config['eval_grad_mode'])
            model.zero_grad()

            # Batch-based telescopic errors (no full-data metrics here)
            tel_error_loss_batch = safe_norm(tel_grad - batch_fine_grad)
            tel_rel_error_loss_batch = tel_error_loss_batch / fine_grad_norm
            tel_error_loss_batch_normed = safe_norm(normalize(tel_grad) - normalize(batch_fine_grad))
            tel_rel_error_loss_batch_normed = (
                tel_error_loss_batch_normed / safe_norm(normalize(batch_fine_grad))
            )

            # Use proper cosine similarity that handles complex vectors consistently
            cos_loss_batch = cosine_similarity(tel_grad, batch_fine_grad)
            cos_loss_batch_normed = cosine_similarity(normalize(tel_grad), normalize(batch_fine_grad))

            print(
                f"    {i + 1:4} |   {tel_error_loss_batch:.6f}   |   {tel_rel_error_loss_batch:.6f}   |   "
                f"{tel_error_loss_batch_normed:.6f}   |   {cos_loss_batch:.6f}   |   {cos_loss_batch_normed:.6f}"
            )

            # Log metrics for correction step i+1 (batch-based only)
            step = i + 1
            batch_metrics['tel']['grad_norms'][step].append(safe_norm(tel_grad).item())
            batch_metrics['tel']['abs_err_batch'][step].append(tel_error_loss_batch.item())
            batch_metrics['tel']['abs_err_n_batch'][step].append(tel_error_loss_batch_normed.item())
            batch_metrics['tel']['rel_err_batch'][step].append(tel_rel_error_loss_batch.item())
            batch_metrics['tel']['rel_err_n_batch'][step].append(tel_rel_error_loss_batch_normed.item())
            batch_metrics['tel']['cos_batch'][step].append(cos_loss_batch.item())
            batch_metrics['tel']['cos_n_batch'][step].append(cos_loss_batch_normed.item())

        # Step 7: Compare telescopic sums (final, per batch)
        print("  Telescopic sum errors (final, vs batch finest):")
        tel_error_loss_batch = safe_norm(tel_grad - batch_fine_grad) / fine_grad_norm
        print(f"    Method 2 (loss diff grad): {tel_error_loss_batch:.6f}")
        batch_metrics['tel_loss'].append(tel_error_loss_batch.item())

    # Step 8: Log averages
    print("\nStep 8: Log averages")
    print("Direct Comparison Means:")
    for res in resolutions[:-1]:
        print(
            f"  Resolution {res}: {np.mean(batch_metrics['direct_grad_norms'][res]):.6f} ± {np.std(batch_metrics['direct_grad_norms'][res]):.6f}")
        print(
            f"  Resolution {res}: {np.mean(batch_metrics['abs_direct_vs_batch'][res]):.6f} ± {np.std(batch_metrics['abs_direct_vs_batch'][res]):.6f}")
        # print(
        #     f"  Resolution {res}: {np.mean(batch_metrics['abs_direct_vs_full'][res]):.6f} ± {np.std(batch_metrics['abs_direct_vs_full'][res]):.6f}")
        print(
            f"  Resolution {res}: {np.mean(batch_metrics['rel_direct_vs_batch'][res]):.6f} ± {np.std(batch_metrics['rel_direct_vs_batch'][res]):.6f}")
        # print(
        #     f"  Resolution {res}: {np.mean(batch_metrics['rel_direct_vs_full'][res]):.6f} ± {np.std(batch_metrics['rel_direct_vs_full'][res]):.6f}")

    print("Telescopic Sum Means:")
    # print(f"  Method 1: {np.mean(batch_metrics['tel_grads']):.6f} ± {np.std(batch_metrics['tel_grads']):.6f}")
    print(f"  Method 2: {np.mean(batch_metrics['tel_loss']):.6f} ± {np.std(batch_metrics['tel_loss']):.6f}")

    # Print summary of progressive metrics for Method 2
    print("\nMethod 2 Progressive Metrics (averaged across all batches):")
    print("    Step | RelErr(batch) | RelErr(N-batch) | CosSim(batch)")
    print("    " + "-" * 70)

    # Calculate and print averages for each step (batch-based only)
    for step in range(len(resolutions)):
        tel_grad_norm = np.mean(batch_metrics['tel']['grad_norms'][step])
        abs_err_batch = np.mean(batch_metrics['tel']['abs_err_batch'][step])
        abs_err_n_batch = np.mean(batch_metrics['tel']['abs_err_n_batch'][step])
        rel_err_batch = np.mean(batch_metrics['tel']['rel_err_batch'][step])
        rel_err_n_batch = np.mean(batch_metrics['tel']['rel_err_n_batch'][step])
        cos_batch = np.mean(batch_metrics['tel']['cos_batch'][step])

        print(
            f"    {step:4} |   {rel_err_batch:.6f}   |   {rel_err_n_batch:.6f}   |   {cos_batch:.6f}"
        )

        # Store in grad_metrics for return (batch-based only)
        grad_metrics[f'tel_step{step}_grad_norm'] = tel_grad_norm
        grad_metrics[f'tel_step{step}_abs_err_batch'] = abs_err_batch
        grad_metrics[f'tel_step{step}_abs_err_n_batch'] = abs_err_n_batch
        grad_metrics[f'tel_step{step}_rel_err_batch'] = rel_err_batch
        grad_metrics[f'tel_step{step}_rel_err_n_batch'] = rel_err_n_batch
        grad_metrics[f'tel_step{step}_cos_batch'] = cos_batch

    if False:
        # Optional: full-gradient progressive metrics (kept for debugging, disabled by default)
        print("\nMethod 2 Progressive Metrics vs full (averaged across all batches):")
        print("    Step | RelErr(full) | RelErr(N-full) | CosSim(full)")
        print("    " + "-" * 70)

        for step in range(len(resolutions)):
            rel_err_full = np.mean(batch_metrics['tel']['rel_err_full'][step])
            rel_err_n_full = np.mean(batch_metrics['tel']['rel_err_n_full'][step])
            cos_full = np.mean(batch_metrics['tel']['cos_full'][step])

            print(
                f"    {step:4} |   {rel_err_full:.6f}   |   "
                f"{rel_err_n_full:.6f}   |   {cos_full:.6f}"
            )

            grad_metrics[f'tel_step{step}_rel_err_full'] = rel_err_full
            grad_metrics[f'tel_step{step}_rel_err_n_full'] = rel_err_n_full
            grad_metrics[f'tel_step{step}_cos_full'] = cos_full

    # Store original metrics
    # grad_metrics['direct_means'] = {res: np.mean(batch_metrics['direct'][res]) for res in resolutions[:-1]}
    grad_metrics['direct_grad_norms_mean'] = {res: np.mean(batch_metrics['direct_grad_norms'][res]) for res in
                                              resolutions}
    grad_metrics['abs_direct_vs_batch_mean'] = {res: np.mean(batch_metrics['abs_direct_vs_batch'][res]) for res in
                                                resolutions[:-1]}
    grad_metrics['rel_direct_vs_batch_mean'] = {res: np.mean(batch_metrics['rel_direct_vs_batch'][res]) for res in
                                                resolutions[:-1]}
    if False:
        grad_metrics['abs_direct_vs_full_mean'] = {res: np.mean(batch_metrics['abs_direct_vs_full'][res]) for res in
                                                   resolutions[:-1]}
        grad_metrics['rel_direct_vs_full_mean'] = {res: np.mean(batch_metrics['rel_direct_vs_full'][res]) for res in
                                                   resolutions[:-1]}

    grad_metrics['tel_loss_mean'] = np.mean(batch_metrics['tel_loss'])

    return grad_metrics

