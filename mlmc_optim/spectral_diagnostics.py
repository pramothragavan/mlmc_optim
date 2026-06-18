"""
Spectral diagnostics for FNO training.
"""

import os
import re
import numpy as np
import torch
from collections import defaultdict


def decode(x, mean, std):
    return x * (std + 1e-5) + mean


class SpectralDiagnostics:
    """Track mode-wise training dynamics for FNO.
    """

    def __init__(self, modes, eval_loader, device='cuda',
                 out_dir='./spectral_logs'):
        self.modes = modes
        self.eval_loader = eval_loader
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Storage
        self.epochs = []
        self.mode_errors = []         # list of P(|k|) arrays (radial)
        self.power_spectra_2d = []    # list of full rfft2 residual spectra
        self.rect_in_band = []        # rectangular in-band total energy
        self.rect_out_band = []       # rectangular out-of-band total energy
        self.block_grad_norms = []    # list of dicts
        self.mse_losses = []          # MSE loss
        self.lp_losses = []           # relative L2 loss
        self.phase_transitions = []   # (epoch, resolution) pairs
        self.current_resolution = None
        self.training_resolutions = []  # resolution at each eval step

    def log_epoch(self, model, epoch, train_mean, train_std, config,
                  phase_res=None):
        """Call this at each eval step."""

        # Track phase transitions and current resolution
        if phase_res is not None:
            if self.current_resolution is None:
                self.current_resolution = phase_res
            elif phase_res != self.current_resolution:
                self.phase_transitions.append((epoch, phase_res))
                self.current_resolution = phase_res
        self.training_resolutions.append(phase_res if phase_res is not None else 0)

        # 1. Mode-wise residual spectrum + both loss types
        P, power_2d, mse, lp, rect_in, rect_out = self.compute_residual_spectrum(
            model, train_mean, train_std, config)
        self.epochs.append(epoch)
        self.mode_errors.append(P)
        self.power_spectra_2d.append(power_2d)
        self.mse_losses.append(mse)
        self.lp_losses.append(lp)
        self.rect_in_band.append(rect_in)
        self.rect_out_band.append(rect_out)

        # 2. Per-block gradient norms
        grad_norms = self.compute_block_grad_norms(model, train_mean, train_std, config)
        self.block_grad_norms.append(grad_norms)

        ratio = rect_in / rect_out if rect_out > 0 else float('inf')
        print(f"  [spectral] epoch {epoch}: MSE={mse:.2e}, Lp={lp:.6f}, "
              f"rect in-band={rect_in:.2e}, out-band={rect_out:.2e}, ratio={ratio:.1f}")

    @torch.no_grad()
    def compute_residual_spectrum(self, model, train_mean, train_std, config):
        """Compute radially averaged |hat{e}(k)|^2 and rectangular in/out energy."""
        model.eval()
        all_power = None
        total_mse = 0.0
        total_lp_num = 0.0
        total_lp_den = 0.0
        n_samples = 0

        for batch in self.eval_loader:
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)

            if config.get('normalize_output', True) and train_mean is not None:
                output = decode(output, train_mean, train_std)
                target = decode(target, train_mean, train_std)

            residual = output - target
            if residual.dim() == 4:
                residual = residual.squeeze(1)
            elif residual.dim() == 2:
                residual = residual.unsqueeze(0)

            B, n1, n2 = residual.shape
            total_mse += torch.sum(residual**2).item()
            n_samples += B

            # Relative L2 components
            res_flat = residual.reshape(B, -1)
            tgt = target.squeeze(1) if target.dim() == 4 else target
            if tgt.dim() == 2:
                tgt = tgt.unsqueeze(0)
            tgt_flat = tgt.reshape(B, -1)
            total_lp_num += torch.norm(res_flat, dim=1).sum().item()
            total_lp_den += torch.norm(tgt_flat, dim=1).sum().item()

            # 2D rFFT
            res_ft = torch.fft.rfft2(residual)
            power = (res_ft.abs()**2).sum(dim=0).cpu().numpy()

            if all_power is None:
                all_power = power
                self._n1, self._n2 = n1, n2
            else:
                all_power += power

        if all_power is None or n_samples == 0:
            return np.zeros(1), np.zeros((1, 1)), 0.0, 0.0, 0.0, 0.0

        all_power /= n_samples
        avg_mse = total_mse / (n_samples * self._n1 * self._n2)
        avg_lp = total_lp_num / total_lp_den if total_lp_den > 0 else 0.0

        P = self.radial_average_rfft(all_power, self._n1, self._n2)
        rect_in, rect_out = self.rectangular_band_energy(
            all_power, self._n1, self._n2, self.modes)

        return P, all_power, avg_mse, avg_lp, rect_in, rect_out

    def radial_average_rfft(self, power_2d, n1, n2):
        """Radially average a 2D rFFT power spectrum."""
        max_k = n1 // 2
        k1 = np.arange(n1)
        k1 = np.where(k1 > n1 // 2, k1 - n1, k1)
        k2 = np.arange(n2 // 2 + 1)

        K1, K2 = np.meshgrid(k1, k2, indexing='ij')
        Km = np.sqrt(K1**2 + K2**2)

        P = np.zeros(max_k + 1)
        counts = np.zeros(max_k + 1)
        for i in range(n1):
            for j in range(n2 // 2 + 1):
                k = int(round(Km[i, j]))
                if k <= max_k:
                    P[k] += power_2d[i, j]
                    counts[k] += 1

        mask = counts > 0
        P[mask] /= counts[mask]
        return P

    @staticmethod
    def rectangular_band_energy(power_2d, n1, n2, modes):
        """Compute in-band and out-of-band energy using FNO's rectangular mask.

        The FNO retains indices [:modes] and [-modes:] along k1, and [:modes]
        along k2 (rfft2), i.e. k1 in [-modes, modes-1] and k2 in [0, modes-1].
        power_2d has shape (n1, n2//2+1) from rfft2.
        """
        k1 = np.arange(n1)
        k1 = np.where(k1 > n1 // 2, k1 - n1, k1)
        k2 = np.arange(n2 // 2 + 1)

        K1, K2 = np.meshgrid(k1, k2, indexing='ij')
        in_mask = (K1 >= -modes) & (K1 < modes) & (K2 < modes)

        in_band = np.sum(power_2d[in_mask])
        out_band = np.sum(power_2d[~in_mask])
        return float(in_band), float(out_band)

    def compute_block_grad_norms(self, model, train_mean, train_std, config):
        """Compute per-block gradient norms from one test batch."""
        model.train()
        model.zero_grad()

        batch = next(iter(self.eval_loader))
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        output = model(data)

        if config.get('normalize_output', True) and train_mean is not None:
            output = decode(output, train_mean, train_std)
            target = decode(target, train_mean, train_std)

        loss = torch.mean((output - target)**2)
        loss.backward()

        # Auto-discover blocks from parameter names
        spectral_re = re.compile(r'conv(\d+)')
        pointwise_re = re.compile(r'(?<!\w)w(\d+)')
        lift_re = re.compile(r'fc0\b')
        project_re = re.compile(r'fc[12]\b')

        norms = {}
        spectral_layers = defaultdict(float)
        pointwise_layers = defaultdict(float)
        lift_sq = 0.0
        project_sq = 0.0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_sq = param.grad.abs().pow(2).sum().item()

            m = spectral_re.search(name)
            if m:
                spectral_layers[int(m.group(1))] += grad_sq
                continue
            m = pointwise_re.search(name)
            if m:
                pointwise_layers[int(m.group(1))] += grad_sq
                continue
            if lift_re.search(name):
                lift_sq += grad_sq
                continue
            if project_re.search(name):
                project_sq += grad_sq
                continue

        for idx, sq in spectral_layers.items():
            norms[f'spectral_{idx}'] = np.sqrt(sq)
        for idx, sq in pointwise_layers.items():
            norms[f'pointwise_{idx}'] = np.sqrt(sq)
        if lift_sq > 0:
            norms['lift'] = np.sqrt(lift_sq)
        if project_sq > 0:
            norms['project'] = np.sqrt(project_sq)

        norms['spectral_all'] = np.sqrt(sum(spectral_layers.values()))
        norms['pointwise_all'] = np.sqrt(sum(pointwise_layers.values()))
        norms['n_spectral_layers'] = len(spectral_layers)
        norms['n_pointwise_layers'] = len(pointwise_layers)

        model.eval()
        model.zero_grad()
        return norms

    @staticmethod
    def compute_decay_rates(epochs, mode_errors):
        """Compute per-mode effective decay rates via log-linear regression.
        """
        n_modes = mode_errors.shape[1]
        rates = np.full(n_modes, np.nan)
        r_squared = np.full(n_modes, np.nan)
        t = epochs.astype(float)

        for k in range(n_modes):
            vals = mode_errors[:, k]
            pos = vals > 0
            if pos.sum() < 3:
                continue
            log_vals = np.log(vals[pos])
            t_pos = t[pos]
            # Linear regression: log P = -rate * t + c
            A = np.vstack([t_pos, np.ones_like(t_pos)]).T
            result = np.linalg.lstsq(A, log_vals, rcond=None)
            slope, intercept = result[0]
            rates[k] = -slope  # positive rate = decay
            # R^2
            ss_res = np.sum((log_vals - (slope * t_pos + intercept))**2)
            ss_tot = np.sum((log_vals - log_vals.mean())**2)
            if ss_tot > 0:
                r_squared[k] = 1.0 - ss_res / ss_tot

        return rates, r_squared

    def save_and_plot(self):
        epochs = np.array(self.epochs)
        mode_errors = np.array(self.mode_errors)
        power_spectra_2d = np.array(self.power_spectra_2d, dtype=np.float32)
        mse_losses = np.array(self.mse_losses)
        lp_losses = np.array(self.lp_losses)
        rect_in = np.array(self.rect_in_band)
        rect_out = np.array(self.rect_out_band)

        # Compute per-mode decay rates
        decay_rates, decay_r2 = self.compute_decay_rates(epochs, mode_errors)

        np.savez(os.path.join(self.out_dir, 'spectral_data.npz'),
                 epochs=epochs, mode_errors=mode_errors,
                 power_spectra_2d=power_spectra_2d,
                 mse_losses=mse_losses, lp_losses=lp_losses,
                 rect_in_band=rect_in, rect_out_band=rect_out,
                 modes=self.modes,
                 phase_transitions=np.array(self.phase_transitions) if self.phase_transitions else np.array([]),
                 block_grad_norms=[dict(d) for d in self.block_grad_norms],
                 decay_rates=decay_rates, decay_r2=decay_r2,
                 training_resolutions=np.array(self.training_resolutions))

        self.plot_diagnostics(epochs, mode_errors, mse_losses, lp_losses,
                               rect_in, rect_out, decay_rates, decay_r2)

    def plot_diagnostics(self, epochs, mode_errors, mse_losses, lp_losses,
                          rect_in, rect_out, decay_rates, decay_r2):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        max_k = mode_errors.shape[1] - 1

        # (0,0) Mode-wise error heatmap
        ax = axes[0, 0]
        n_show = min(max_k, 30)
        extent = [epochs[0], epochs[-1], 0.5, n_show + 0.5]
        im = ax.imshow(np.log10(mode_errors[:, 1:n_show+1].T + 1e-30),
                        aspect='auto', origin='lower', extent=extent,
                        cmap='viridis')
        for ep, res in self.phase_transitions:
            ax.axvline(x=ep, color='red', ls='--', lw=1, alpha=0.7)
        ax.axhline(y=self.modes, color='white', ls=':', lw=1, alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('$|k|$')
        ax.set_title('$\\log_{10} P(|k|, t)$')
        plt.colorbar(im, ax=ax)

        # (0,1) Mode decay curves with phase labels
        ax = axes[0, 1]
        modes_to_show = [1, 2, 4, self.modes, self.modes + 2,
                          min(2 * self.modes, max_k)]
        modes_to_show = sorted(set(m for m in modes_to_show if 1 <= m <= max_k))
        for k in modes_to_show:
            vals = mode_errors[:, k]
            in_band = k < self.modes
            ax.semilogy(epochs, vals,
                         '-' if in_band else '--', lw=1.5,
                         label=f'|k|={k}' + (' (in)' if in_band else ' (out)'))
        for ep, res in self.phase_transitions:
            ax.axvline(x=ep, color='red', ls='--', lw=0.8, alpha=0.5)
            ax.text(ep, ax.get_ylim()[0] * 2, f'r{res}', fontsize=7,
                    color='red', ha='center', rotation=90)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('$P(|k|,t)$')
        ax.set_title('Mode-wise residual energy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (0,2) Loss curves
        ax = axes[0, 2]
        ax.semilogy(epochs, lp_losses, 'b-', lw=1.5, label='Relative L2')
        for ep, res in self.phase_transitions:
            ax.axvline(x=ep, color='red', ls='--', lw=0.8, alpha=0.5)
            ax.text(ep, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
                    f'r{res}', fontsize=7, color='red', ha='center', va='bottom')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Test loss (relative L2)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1,0) Per-block gradient norms
        ax = axes[1, 0]
        block_keys = ['spectral_all', 'pointwise_all', 'lift', 'project']
        colors_block = {'spectral_all': '#3498db', 'pointwise_all': '#e74c3c',
                  'lift': '#27ae60', 'project': '#8e44ad'}
        for key in block_keys:
            vals = [d.get(key, 0) for d in self.block_grad_norms]
            if any(v > 0 for v in vals):
                ax.semilogy(epochs, vals, '-', lw=1.5,
                             color=colors_block.get(key, 'gray'), label=key)
        for ep, res in self.phase_transitions:
            ax.axvline(x=ep, color='red', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('$\\|\\nabla_{\\mathrm{block}} \\mathcal{L}\\|_F$')
        ax.set_title('Per-block gradient norms')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # (1,1) In-band / out-band energy ratio over time
        ax = axes[1, 1]
        ratio = rect_in / np.maximum(rect_out, 1e-30)
        ax.semilogy(epochs, rect_in, '-', color='#3498db', lw=1.5, label='In-band')
        ax.semilogy(epochs, rect_out, '-', color='#e74c3c', lw=1.5, label='Out-of-band')
        ax2 = ax.twinx()
        ax2.plot(epochs, ratio, '-', color='gray', lw=1, alpha=0.5, label='Ratio')
        ax2.set_ylabel('In/Out ratio', color='gray', fontsize=8)
        for ep, res in self.phase_transitions:
            ax.axvline(x=ep, color='red', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Residual energy')
        ax.set_title('In-band vs out-of-band residual')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # (1,2) Per-mode decay rates (the NTK comparison panel)
        ax = axes[1, 2]
        n_show_k = min(max_k + 1, 3 * self.modes)
        k_plot = np.arange(1, n_show_k)
        valid = k_plot < len(decay_rates)
        k_valid = k_plot[valid]
        rates_valid = decay_rates[k_valid]

        has_data = ~np.isnan(rates_valid) & (rates_valid > 0)
        if has_data.any():
            colors_bar = ['#3498db' if k < self.modes else '#e74c3c' for k in k_valid]
            bars = ax.bar(k_valid[has_data], rates_valid[has_data],
                   color=[c for c, h in zip(colors_bar, has_data) if h],
                   alpha=0.8, width=0.8)
            ax.axvline(x=self.modes + 0.5, color='gray', ls=':', lw=1.5)
            ax.set_yscale('log')
        ax.set_xlabel('$|k|$')
        ax.set_ylabel('Decay rate (epoch$^{-1}$)')
        ax.set_title('Per-mode decay rates (blue=in-band, red=out)')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Spectral Training Diagnostics', fontsize=14)
        plt.tight_layout()
        fig_path = os.path.join(self.out_dir, 'spectral_diagnostics.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved spectral diagnostics to {fig_path}")
