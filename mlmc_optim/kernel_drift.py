"""Empirical projected-NTK drift diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


def _json_ready(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if torch.is_tensor(obj):
        obj = obj.detach().cpu()
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_ready(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    return str(obj)


def _parse_int_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value[0] in "[(":
            import ast

            value = ast.literal_eval(value)
        else:
            value = [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(x) for x in value]


def _as_field(output):
    if output.dim() == 4 and output.shape[1] == 1:
        return output[:, 0]
    if output.dim() == 4 and output.shape[-1] == 1:
        return output[..., 0]
    if output.dim() == 3:
        return output
    raise ValueError(f"Expected 2D scalar field output, got shape {tuple(output.shape)}")


def _normalise_projection(proj):
    norm = torch.linalg.vector_norm(proj)
    if float(norm) == 0.0:
        return proj
    return proj / norm


def make_fourier_projections(height, width, radius, max_projections,
                             include_mean=False, device="cpu"):
    """Build real cosine/sine Fourier output probes."""
    yy = torch.arange(height, dtype=torch.float32, device=device).reshape(height, 1)
    xx = torch.arange(width, dtype=torch.float32, device=device).reshape(1, width)
    specs = []
    if include_mean:
        specs.append(("mean", 0, 0))

    freqs = []
    radius_int = int(radius)
    for k1 in range(-radius_int, radius_int + 1):
        for k2 in range(-radius_int, radius_int + 1):
            if k1 == 0 and k2 == 0:
                continue
            if k1 < 0 or (k1 == 0 and k2 < 0):
                continue
            norm = math.sqrt(k1 * k1 + k2 * k2)
            if norm <= float(radius):
                freqs.append((norm, abs(k1) + abs(k2), k1, k2))
    freqs.sort()
    for _, _, k1, k2 in freqs:
        specs.append(("cos", k1, k2))
        specs.append(("sin", k1, k2))
    specs = specs[: int(max_projections)]

    projections = []
    for kind, k1, k2 in specs:
        if kind == "mean":
            proj = torch.ones((height, width), dtype=torch.float32, device=device)
        else:
            phase = 2.0 * math.pi * (k1 * yy / float(height) + k2 * xx / float(width))
            proj = torch.cos(phase) if kind == "cos" else torch.sin(phase)
        projections.append(_normalise_projection(proj))

    if not projections:
        raise ValueError("No Fourier projections were generated.")
    return torch.stack(projections), specs


def make_random_projections(height, width, max_projections, seed, device="cpu"):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    projections = []
    specs = []
    for idx in range(int(max_projections)):
        proj = torch.empty((height, width), dtype=torch.float32).bernoulli_(
            0.5, generator=gen)
        proj = 2.0 * proj - 1.0
        projections.append(_normalise_projection(proj.to(device)))
        specs.append(("rademacher", idx, int(seed)))
    return torch.stack(projections), specs


def flatten_grads(parameters):
    chunks = []
    for param in parameters:
        if param.grad is None:
            grad = torch.zeros_like(param)
        else:
            grad = param.grad.detach()
        if torch.is_complex(grad):
            chunks.append(torch.view_as_real(grad).reshape(-1))
        else:
            chunks.append(grad.reshape(-1))
    return torch.cat(chunks)


def top_eigenvalue_psd(matrix, n_iter=60):
    if matrix.numel() == 0:
        return float("nan")
    vec = torch.ones(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    vec = vec / torch.linalg.vector_norm(vec)
    for _ in range(int(n_iter)):
        vec = matrix @ vec
        norm = torch.linalg.vector_norm(vec)
        if float(norm) == 0.0:
            return 0.0
        vec = vec / norm
    return float(vec @ (matrix @ vec))


class KernelDriftDiagnostics:
    """Track exact projected empirical-NTK drift on a fixed probe set."""

    def __init__(
        self,
        config,
        dataset,
        device,
        denormalizer=None,
        out_dir: Optional[str] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.device = torch.device(device)
        self.denormalizer = denormalizer
        self.out_dir = Path(out_dir or config.get("kernel_drift_out_dir")
                            or Path(config.get("out_dir", ".")) / "kernel_drift")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        default_epochs = [0, 1, 2, 5, 10, 20, 50, 100]
        self.epochs = set(_parse_int_list(
            config.get("kernel_drift_epochs"), default_epochs))
        self.probe_count = int(config.get("kernel_drift_probe_samples", 64))
        self.probe_batch_size = int(config.get("kernel_drift_probe_batch_size", 4))
        self.num_projections = int(config.get("kernel_drift_num_projections", 16))
        self.fourier_radius = int(config.get("kernel_drift_fourier_radius", 8))
        self.projection_type = str(config.get("kernel_drift_projection_type", "fourier"))
        self.include_mean = bool(config.get("kernel_drift_include_mean", False))
        self.max_rows = int(config.get("kernel_drift_max_rows", 2048))
        self.save_kernels = bool(config.get("kernel_drift_save_kernels", True))
        self.save_jacobian = bool(config.get("kernel_drift_save_jacobian", False))
        self.save_projections = bool(config.get("kernel_drift_save_projections", True))
        self.denormalize_output = bool(config.get("kernel_drift_denormalize_output", False))
        seed_value = config.get("kernel_drift_seed")
        if seed_value is None:
            seed_value = int(config.get("seed", 0)) + 1729
        self.seed = int(seed_value)

        n_data = len(dataset)
        if self.probe_count > n_data:
            raise ValueError(
                f"kernel_drift_probe_samples={self.probe_count} exceeds dataset size {n_data}.")
        explicit_indices = config.get("kernel_drift_probe_indices")
        if explicit_indices is not None:
            indices = _parse_int_list(explicit_indices, [])
            if len(indices) < self.probe_count:
                raise ValueError(
                    "kernel_drift_probe_indices has fewer entries than "
                    "kernel_drift_probe_samples.")
            self.indices = np.asarray(indices[: self.probe_count], dtype=np.int64)
        else:
            rng = np.random.default_rng(self.seed)
            self.indices = np.sort(
                rng.choice(n_data, size=self.probe_count, replace=False)
            ).astype(np.int64)

        with torch.no_grad():
            sample_x, sample_y = dataset.get_items(torch.as_tensor(self.indices[:1]))
            sample_y = _as_field(sample_y.to(self.device))
        height, width = int(sample_y.shape[-2]), int(sample_y.shape[-1])
        if self.projection_type == "fourier":
            self.projections, self.projection_specs = make_fourier_projections(
                height, width, self.fourier_radius, self.num_projections,
                include_mean=self.include_mean, device=self.device)
        elif self.projection_type == "random":
            self.projections, self.projection_specs = make_random_projections(
                height, width, self.num_projections, self.seed, device=self.device)
        else:
            raise ValueError(
                "kernel_drift_projection_type must be 'fourier' or 'random'.")

        self.row_count = self.probe_count * int(self.projections.shape[0])
        if self.row_count > self.max_rows:
            raise ValueError(
                f"Kernel drift probe has {self.row_count} rows "
                f"({self.probe_count} samples x {self.projections.shape[0]} projections), "
                f"exceeding kernel_drift_max_rows={self.max_rows}. Raise the limit "
                "intentionally for a heavier Dawn run.")

        self.k0 = None
        self.k0_norm = None
        self.rows = []
        self.metadata_written = False
        self.write_metadata()

    def should_log(self, epoch_completed):
        return int(epoch_completed) in self.epochs

    def write_metadata(self):
        if self.metadata_written:
            return
        meta = {
            "probe_indices": self.indices.tolist(),
            "probe_samples": self.probe_count,
            "probe_batch_size": self.probe_batch_size,
            "num_projections": int(self.projections.shape[0]),
            "projection_type": self.projection_type,
            "projection_specs": self.projection_specs,
            "fourier_radius": self.fourier_radius,
            "include_mean": self.include_mean,
            "row_count": self.row_count,
            "epochs": sorted(self.epochs),
            "denormalize_output": self.denormalize_output,
            "seed": self.seed,
        }
        with (self.out_dir / "metadata.json").open("w") as f:
            json.dump(_json_ready(meta), f, indent=2)
        self.metadata_written = True

    def _probe_batch(self, batch_indices):
        idx = torch.as_tensor(batch_indices, dtype=torch.long)
        data, target = self.dataset.get_items(idx)
        data = data.to(self.device)
        target = _as_field(target.to(self.device))
        return data, target

    def _projected_residuals(self, output, target):
        output = _as_field(output)
        target = _as_field(target)
        if self.denormalize_output and self.denormalizer is not None:
            output = _as_field(self.denormalizer(output))
            target = _as_field(self.denormalizer(target))
        out_proj = torch.einsum("bhw,phw->bp", output, self.projections)
        tgt_proj = torch.einsum("bhw,phw->bp", target, self.projections)
        return out_proj, tgt_proj, out_proj - tgt_proj

    def compute_projected_jacobian(self, model):
        was_training = model.training
        model.eval()
        parameters = [p for p in model.parameters() if p.requires_grad]
        rows = []
        output_proj_chunks = []
        target_proj_chunks = []
        residual_proj_chunks = []
        n_proj = int(self.projections.shape[0])

        for start in range(0, self.probe_count, self.probe_batch_size):
            stop = min(start + self.probe_batch_size, self.probe_count)
            data, target = self._probe_batch(self.indices[start:stop])
            output = model(data)
            out_proj, tgt_proj, res_proj = self._projected_residuals(output, target)
            output_proj_chunks.append(out_proj.detach().cpu())
            target_proj_chunks.append(tgt_proj.detach().cpu())
            residual_proj_chunks.append(res_proj.detach().cpu())
            scalars = out_proj.reshape(-1)

            for row_idx, scalar in enumerate(scalars):
                model.zero_grad(set_to_none=True)
                scalar.backward(retain_graph=row_idx < scalars.numel() - 1)
                rows.append(flatten_grads(parameters).detach().float().cpu())

            model.zero_grad(set_to_none=True)
            del data, target, output, out_proj, tgt_proj, res_proj, scalars

        if was_training:
            model.train()

        jac = torch.stack(rows, dim=0)
        output_proj = torch.cat(output_proj_chunks, dim=0)
        target_proj = torch.cat(target_proj_chunks, dim=0)
        residual_proj = torch.cat(residual_proj_chunks, dim=0)
        return jac, output_proj, target_proj, residual_proj

    def summarize_kernel(self, kernel):
        fro = float(torch.linalg.matrix_norm(kernel))
        trace = float(torch.trace(kernel))
        diag = torch.diag(kernel)
        top = top_eigenvalue_psd(kernel)
        if self.k0 is None:
            return {
                "kernel_fro_norm": fro,
                "kernel_trace": trace,
                "kernel_top_eig": top,
                "kernel_rel_fro_drift": 0.0,
                "kernel_alignment": 1.0,
                "kernel_trace_ratio": 1.0,
                "kernel_top_eig_ratio": 1.0,
                "kernel_diag_rel_drift": 0.0,
            }

        diff = kernel - self.k0
        k0_norm = max(float(self.k0_norm), 1e-30)
        align_denom = max(
            float(torch.linalg.matrix_norm(kernel) * self.k0_norm), 1e-30)
        alignment = float(torch.sum(kernel * self.k0)) / align_denom
        diag0 = torch.diag(self.k0)
        diag0_norm = max(float(torch.linalg.vector_norm(diag0)), 1e-30)
        diag_rel = float(torch.linalg.vector_norm(diag - diag0) / diag0_norm)
        trace0 = max(float(torch.trace(self.k0)), 1e-30)
        top0 = max(float(top_eigenvalue_psd(self.k0)), 1e-30)
        return {
            "kernel_fro_norm": fro,
            "kernel_trace": trace,
            "kernel_top_eig": top,
            "kernel_rel_fro_drift": float(torch.linalg.matrix_norm(diff) / k0_norm),
            "kernel_alignment": alignment,
            "kernel_trace_ratio": trace / trace0,
            "kernel_top_eig_ratio": top / top0,
            "kernel_diag_rel_drift": diag_rel,
        }

    def log_snapshot(self, model, epoch_completed, phase=None, phase_res=None,
                     cum_train_time=None):
        print(
            f"[kernel_drift] snapshot epoch_completed={epoch_completed} "
            f"rows={self.row_count}")
        jac, output_proj, target_proj, residual_proj = self.compute_projected_jacobian(model)
        kernel = jac @ jac.T
        if self.k0 is None:
            self.k0 = kernel.clone()
            self.k0_norm = torch.linalg.matrix_norm(self.k0)

        metrics = self.summarize_kernel(kernel)
        row = {
            "epoch_completed": int(epoch_completed),
            "phase": -1 if phase is None else int(phase),
            "phase_res": -1 if phase_res is None else int(phase_res),
            "cum_train_time": cum_train_time,
            "probe_samples": self.probe_count,
            "num_projections": int(self.projections.shape[0]),
            "row_count": self.row_count,
            **metrics,
        }
        self.rows.append(row)

        snapshot_path = self.out_dir / f"snapshot_epoch{int(epoch_completed):06d}.npz"
        payload = {
            "epoch_completed": np.int64(epoch_completed),
            "phase": np.int64(-1 if phase is None else phase),
            "phase_res": np.int64(-1 if phase_res is None else phase_res),
            "metrics_json": np.array(json.dumps(_json_ready(row))),
        }
        if self.save_kernels:
            payload["kernel"] = kernel.numpy().astype(np.float32)
        if self.save_jacobian:
            payload["jacobian"] = jac.numpy().astype(np.float32)
        if self.save_projections:
            payload["output_proj"] = output_proj.numpy().astype(np.float32)
            payload["target_proj"] = target_proj.numpy().astype(np.float32)
            payload["residual_proj"] = residual_proj.numpy().astype(np.float32)
        np.savez_compressed(snapshot_path, **payload)
        self.save_summary()
        print(
            "[kernel_drift] rel_fro={:.4g} align={:.4g} trace_ratio={:.4g} -> {}".format(
                row["kernel_rel_fro_drift"],
                row["kernel_alignment"],
                row["kernel_trace_ratio"],
                snapshot_path,
            )
        )
        return row

    def save_summary(self):
        if not self.rows:
            return
        csv_path = self.out_dir / "summary.csv"
        keys = list(self.rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(_json_ready(row) for row in self.rows)
        with (self.out_dir / "summary.json").open("w") as f:
            json.dump(_json_ready(self.rows), f, indent=2)
