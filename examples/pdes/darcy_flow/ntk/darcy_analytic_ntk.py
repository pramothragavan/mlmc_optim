"""Analytical infinite-width FNO NTK for Darcy-style 2D grids.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _resolve_selfconj_fix(fix):
    """Resolve the self-conjugate spectral-mask fix flag.
    """
    if fix is None:
        return os.environ.get("NTK_SELFCONJ_FIX", "0").strip().lower() in (
            "1", "true", "yes", "on")
    return bool(fix)


def _resolve_spectral_ntk_dof2(dof2):
    """Resolve the spectral-NTK-term factor-2 correction.
    """
    if dof2 is None:
        return os.environ.get("NTK_SPECTRAL_NTK_DOF2", "0").strip().lower() in (
            "1", "true", "yes", "on")
    return bool(dof2)


@dataclass(frozen=True)
class LayerSpec:

    kind: str
    pointwise_sigma: float
    spectral_sigma: float = 0.0
    bias_sigma: float = 0.0
    modes1: int = 0
    modes2: Optional[int] = None
    activation: str = "relu"


def relu_covariance(q, qp, c):
    denom = np.sqrt(q * qp)
    if denom < 1e-30:
        return np.zeros_like(c)
    rho = np.clip(c / denom, -1.0, 1.0)
    theta = np.arccos(rho)
    return (denom / (2.0 * np.pi)) * (
        np.sqrt(1.0 - rho ** 2) + rho * (np.pi - theta))


def relu_gate_covariance(q, qp, c):
    denom = np.sqrt(q * qp)
    if denom < 1e-30:
        return np.zeros_like(c)
    rho = np.clip(c / denom, -1.0, 1.0)
    return (np.pi - np.arccos(rho)) / (2.0 * np.pi)


def lambda_mask_2d(n, modes1, modes2=None):
    """Full-spectrum rectangle matching FNO2d_NTK's retained rfft slices."""
    modes2 = modes1 if modes2 is None else modes2
    k = np.arange(n)
    retained1 = (k < modes1) | (k >= n - modes1)
    retained2 = (k < modes2) | (k >= n - modes2)
    r1, r2 = np.meshgrid(retained1, retained2, indexing="ij")
    return r1 & r2


def build_fno2d_ntk_layer_specs(num_fourier_layers=4, head_layers=1,
                                modes=8, modes1=None, modes2=None,
                                sigma_u=1.0, sigma_w=1.0,
                                sigma_b=1.0, sigma_q=1.0,
                                sigma_q_bias=1.0,
                                final_fourier_relu=False):
    modes1 = int(modes if modes1 is None else modes1)
    modes2 = int(modes1 if modes2 is None else modes2)

    specs = []
    num_fourier_layers = int(num_fourier_layers)
    for ell in range(num_fourier_layers):
        activation = "relu"
        if ell == num_fourier_layers - 1 and not final_fourier_relu:
            activation = "identity"
        specs.append(LayerSpec(
            kind="fourier",
            pointwise_sigma=float(sigma_u),
            spectral_sigma=float(sigma_w),
            bias_sigma=float(sigma_b),
            modes1=modes1,
            modes2=modes2,
            activation=activation,
        ))
    specs.extend(
        LayerSpec(
            kind="head",
            pointwise_sigma=float(sigma_q),
            bias_sigma=float(sigma_q_bias),
            activation="relu",
        )
        for _ in range(int(head_layers))
    )
    return specs


def gamma_2d(n, spec):
    gamma = spec.pointwise_sigma ** 2 * np.ones((n, n), dtype=np.float64)
    if spec.spectral_sigma:
        modes2 = spec.modes1 if spec.modes2 is None else spec.modes2
        gamma += spec.spectral_sigma ** 2 * lambda_mask_2d(
            n, spec.modes1, modes2).astype(np.float64)
    return gamma


def realized_spectral_gain(X_hat, n, modes1, modes2=None):
    modes2 = modes1 if modes2 is None else modes2
    k1 = np.arange(n)
    ret1 = (k1 < modes1) | (k1 >= n - modes1)
    even_n = (n % 2 == 0)
    half = n // 2
    idx_mirror = (n - np.arange(n)) % n          # k1 -> (n - k1) mod n
    sc_row = (k1 == 0) | (even_n & (k1 == half))  # self-conjugate rows

    P = np.zeros_like(X_hat)
    ncols = min(int(modes2), n)
    for k2 in range(ncols):
        col = np.where(ret1, X_hat[..., :, k2], 0)        # (lead..., n) over k1
        col_mir = col[..., idx_mirror]                    # mirror in k1
        is_sc_col = (k2 == 0) or (even_n and k2 == half)
        if is_sc_col:
            # Reality projection couples (k1, k2_sc) with (n-k1, k2_sc).
            term = np.where(sc_row, col / 2.0, (col + col_mir) / 4.0)
            P[..., :, k2] += term
        else:
            mk2 = (n - k2) % n
            P[..., :, k2] += col                          # direct mode
            P[..., :, mk2] += col_mir                     # Hermitian mirror
    return P


def apply_layer_gain(X_hat, n, spec, fix):
    if not fix:
        return gamma_2d(n, spec) * X_hat
    out = (spec.pointwise_sigma ** 2) * X_hat
    if spec.spectral_sigma:
        modes2 = spec.modes1 if spec.modes2 is None else spec.modes2
        out = out + (spec.spectral_sigma ** 2) * realized_spectral_gain(
            X_hat, n, spec.modes1, modes2)
    return out


def initial_feature_covariance(input_covariance, in_channels, sigma_p=1.0,
                               sigma_p_bias=1.0):
    return (sigma_p ** 2 / float(in_channels)) * input_covariance + sigma_p_bias ** 2


def forward_covariance_2d(C0, layer_specs, fix=None):
    fix = _resolve_selfconj_fix(fix)
    C = [C0.copy()]
    C_hat = [np.fft.fft2(C0)]
    Q = []
    Q_hat = []
    dot_C = []

    for ell, spec in enumerate(layer_specs):
        Qh = apply_layer_gain(C_hat[ell], C0.shape[0], spec, fix)
        Q_ell = np.fft.ifft2(Qh).real + spec.bias_sigma ** 2
        Qh = np.fft.fft2(Q_ell)
        q0 = Q_ell[0, 0]

        activation = getattr(spec, "activation", "relu")
        if activation == "relu":
            C_next = relu_covariance(q0, q0, Q_ell)
            dot_next = relu_gate_covariance(q0, q0, Q_ell)
        elif activation == "identity":
            C_next = Q_ell
            dot_next = np.ones_like(Q_ell)
        else:
            raise ValueError(f"Unsupported activation {activation!r}; use 'relu' or 'identity'.")

        Q.append(Q_ell)
        Q_hat.append(Qh)
        C.append(C_next)
        C_hat.append(np.fft.fft2(C_next))
        dot_C.append(dot_next)

    return {
        "C": C,
        "C_hat": C_hat,
        "Q": Q,
        "Q_hat": Q_hat,
        "dot_C": dot_C,
    }


def compute_ntk_spectrum_2d(fwd, layer_specs, input_covariance, in_channels,
                            sigma_p=1.0, sigma_p_bias=1.0,
                            sigma_out=1.0, sigma_out_bias=1.0,
                            chunk=None, verbose=True, fix=None,
                            spectral_ntk_dof2=None):
    fix = _resolve_selfconj_fix(fix)
    dof2 = _resolve_spectral_ntk_dof2(spectral_ntk_dof2)
    n = fwd["C"][0].shape[0]
    num_layers = len(layer_specs)
    K_r = sigma_out ** 2 * fwd["C"][num_layers].real.copy()
    K_r += sigma_out_bias ** 2

    if chunk is None or chunk >= n:
        K_r += _backward_chunk(
            fwd, layer_specs, input_covariance, in_channels,
            sigma_p, sigma_p_bias, sigma_out, 0, n, fix, dof2)
    else:
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            K_r[start:stop] += _backward_chunk(
                fwd, layer_specs, input_covariance, in_channels,
                sigma_p, sigma_p_bias, sigma_out, start, stop, fix, dof2)
            if verbose:
                print(f"    ntk rows {stop}/{n}", end="\r", flush=True)
        if verbose:
            print("    ntk rows done     ")

    return K_r, np.fft.fft2(K_r).real


def _backward_chunk(fwd, layer_specs, input_covariance, in_channels, sigma_p,
                    sigma_p_bias, sigma_out, r1_start, r1_stop, fix=False,
                    dof2=False):
    n = fwd["C"][0].shape[0]
    grid_size = n * n
    chunk_n = r1_stop - r1_start
    top_layer = len(layer_specs) - 1

    # B axes: (chunked r1, r2, s1, s2).
    B = np.zeros((chunk_n, n, n, n), dtype=np.float64)
    r2 = np.arange(n)
    top_gate = fwd["dot_C"][top_layer]
    for local_i in range(chunk_n):
        r1 = r1_start + local_i
        B[local_i, r2, r1, r2] = (
            sigma_out ** 2 * top_gate[r1, r2] / grid_size)

    K_extra = np.zeros((chunk_n, n), dtype=np.float64)
    B_hat = np.fft.fft2(B, axes=(-2, -1))
    K_extra += _layer_terms(B, B_hat, fwd, top_layer, layer_specs[top_layer],
                            grid_size, fix, dof2)

    for ell in range(top_layer, 0, -1):
        B = np.fft.ifft2(
            apply_layer_gain(B_hat, n, layer_specs[ell], fix),
            axes=(-2, -1)).real
        B *= fwd["dot_C"][ell - 1][None, None]
        B_hat = np.fft.fft2(B, axes=(-2, -1))
        K_extra += _layer_terms(B, B_hat, fwd, ell - 1,
                                layer_specs[ell - 1], grid_size, fix, dof2)

    B_eps0 = np.fft.ifft2(
        apply_layer_gain(B_hat, n, layer_specs[0], fix), axes=(-2, -1)).real
    K_extra += _lift_terms(B_eps0, input_covariance, in_channels,
                           sigma_p, sigma_p_bias, grid_size)

    return K_extra


def _layer_terms(B, B_hat, fwd, ell, spec, grid_size, fix=False, dof2=False):
    n = fwd["C"][0].shape[0]
    K = spec.pointwise_sigma ** 2 * grid_size * np.einsum(
        "ijkl,kl->ij", B, fwd["C"][ell])
    if spec.spectral_sigma:
        modes2 = spec.modes1 if spec.modes2 is None else spec.modes2
        if fix:
            spectral_cov = realized_spectral_gain(
                fwd["C_hat"][ell], n, spec.modes1, modes2)
        else:
            mask = lambda_mask_2d(n, spec.modes1, modes2).astype(np.float64)
            spectral_cov = mask * fwd["C_hat"][ell]
        dof_factor = 2.0 if (fix and dof2) else 1.0
        K += dof_factor * spec.spectral_sigma ** 2 * np.einsum(
            "ijkl,kl->ij", B_hat, spectral_cov).real
    if spec.bias_sigma:
        K += spec.bias_sigma ** 2 * grid_size * B.sum(axis=(-2, -1))
    return K


def _lift_terms(B_eps0, input_covariance, in_channels, sigma_p,
                sigma_p_bias, grid_size):
    K = (sigma_p ** 2 / float(in_channels)) * grid_size * np.einsum(
        "ijkl,kl->ij", B_eps0, input_covariance)
    if sigma_p_bias:
        K += sigma_p_bias ** 2 * grid_size * B_eps0.sum(axis=(-2, -1))
    return K


def compute_input_covariance(inputs):
    """Shift-averaged channel inner-product covariance of input fields."""
    arr = np.asarray(inputs, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, None]
    if arr.ndim != 4:
        raise ValueError(f"Expected inputs with shape (N, C, n, n), got {arr.shape}")
    n_samples, _, n, n2 = arr.shape
    if n != n2:
        raise ValueError(f"Expected square input fields, got {arr.shape}")

    power = np.zeros((n, n), dtype=np.float64)
    for sample in arr:
        sample_hat = np.fft.fft2(sample, axes=(-2, -1))
        power += np.sum(np.abs(sample_hat) ** 2, axis=0)
    power /= float(n_samples)
    return np.fft.ifft2(power).real / float(n * n)


def load_darcy_inputs(data_dir, resolution, max_samples=200,
                      normalize_coeff=True):
    """Load coefficient-only FNO2d_NTK inputs from Darcy ``train/test_rN.pt`` files."""
    import torch

    data_dir = Path(data_dir)
    roots = [data_dir / "darcy2d", data_dir]
    for root in roots:
        for prefix in ("train", "test"):
            path = root / f"{prefix}_r{resolution}.pt"
            if not path.exists():
                continue
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(path, map_location="cpu")
            coeff_all = data["coeff"].float().cpu().numpy()
            if coeff_all.ndim == 4:
                coeff_all = (coeff_all[:, 0] if coeff_all.shape[1] < coeff_all.shape[2]
                             else coeff_all[..., 0])
            if normalize_coeff:
                coeff_all = ((coeff_all - coeff_all.mean())
                             / (coeff_all.std() + 1e-5))
            coeff = coeff_all[:max_samples]
            if coeff.ndim == 4:
                coeff = coeff[:, 0] if coeff.shape[1] < coeff.shape[2] else coeff[..., 0]
            coeff = np.asarray(coeff[:max_samples], dtype=np.float64)
            if coeff.ndim != 3:
                raise ValueError(f"Unsupported coeff shape in {path}: {coeff.shape}")
            inputs = coeff[:, None]
            return inputs, path
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(
        f"No train_r{resolution}.pt or test_r{resolution}.pt under {searched}")

def default_chunk_size(resolution):
    if resolution <= 60:
        return None
    if resolution <= 120:
        return 8
    return 1
