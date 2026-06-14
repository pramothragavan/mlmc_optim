#!/usr/bin/env python3
"""Generate full 2D analytical NTK spectra for Darcy allocator experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from .darcy_analytic_ntk import (
        LayerSpec,
        build_fno2d_ntk_layer_specs,
        compute_input_covariance,
        compute_ntk_spectrum_2d,
        default_chunk_size,
        forward_covariance_2d,
        initial_feature_covariance,
        load_darcy_inputs,
    )
except ImportError:  # Allow direct script execution.
    from darcy_analytic_ntk import (
        LayerSpec,
        build_fno2d_ntk_layer_specs,
        compute_input_covariance,
        compute_ntk_spectrum_2d,
        default_chunk_size,
        forward_covariance_2d,
        initial_feature_covariance,
        load_darcy_inputs,
    )


def parse_resolutions(raw):
    return [int(item) for item in raw.split(",") if item.strip()]


def parse_chunks(raw):
    if raw is None:
        return {}
    out = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        out[int(key)] = None if value.strip().lower() == "none" else int(value)
    return out


def parse_layer_specs(raw):
    if raw is None:
        return None
    data = json.loads(raw)
    specs = []
    for item in data:
        modes = int(item.get("modes", 0))
        modes1 = int(item.get("modes1", modes))
        modes2_raw = item.get("modes2", modes1)
        specs.append(LayerSpec(
            kind=str(item.get("kind", "custom")),
            pointwise_sigma=float(item.get("pointwise_sigma", 0.0)),
            spectral_sigma=float(item.get("spectral_sigma", 0.0)),
            bias_sigma=float(item.get("bias_sigma", 0.0)),
            modes1=modes1,
            modes2=None if modes2_raw is None else int(modes2_raw),
            activation=str(item.get("activation", "relu")),
        ))
    return specs


def layer_specs_metadata(layer_specs):
    return np.array([
        {
            "kind": spec.kind,
            "pointwise_sigma": spec.pointwise_sigma,
            "spectral_sigma": spec.spectral_sigma,
            "bias_sigma": spec.bias_sigma,
            "modes1": spec.modes1,
            "modes2": spec.modes1 if spec.modes2 is None else spec.modes2,
            "activation": getattr(spec, "activation", "relu"),
        }
        for spec in layer_specs
    ], dtype=object)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate full fft2 NTK spectra K_hat_r for Darcy.")
    parser.add_argument("--data_dir", type=Path, default=Path("examples/pdes/darcy_flow/data"),
                        help="Directory containing darcy2d/train_r*.pt files.")
    parser.add_argument("--out", type=Path,
                        default=Path("examples/pdes/darcy_flow/data/darcy_ntk_spectrum_2d.npz"))
    parser.add_argument("--resolutions", default="15,30,60,120,241")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--chunks",
                        help="Optional comma map like '120:8,241:1'.")

    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--modes1", type=int, default=None)
    parser.add_argument("--modes2", type=int, default=None)
    parser.add_argument("--num_fourier_layers", "--num_layers",
                        dest="num_fourier_layers", type=int, default=4)
    parser.add_argument("--head_layers", type=int, default=1,
                        help="Number of biased ReLU pointwise head layers before readout.")
    parser.add_argument("--final_fourier_relu", action="store_true",
                        help=("Match FNO2d/FNO2d_NTK final_fourier_relu=True. "
                              "By default the final Fourier block uses identity activation, "
                              "matching the current no-final-Fourier-ReLU experiments."))
    parser.add_argument("--layer_specs_json", default=None,
                        help=("Full layer override. JSON list of objects with "
                              "kind, pointwise_sigma, spectral_sigma, "
                              "bias_sigma, modes/modes1/modes2."))

    parser.add_argument("--sigma_p", type=float, default=1.0)
    parser.add_argument("--sigma_p_bias", type=float, default=1.0)
    parser.add_argument("--sigma_u", type=float, default=1.0,
                        help="Fourier-block pointwise branch sigma.")
    parser.add_argument("--sigma_w", type=float, default=1.0,
                        help="Fourier-block spectral branch sigma.")
    parser.add_argument("--sigma_b", type=float, default=1.0,
                        help="Fourier-block pointwise bias sigma.")
    parser.add_argument("--sigma_q", type=float, default=1.0,
                        help="Hidden pointwise head sigma.")
    parser.add_argument("--sigma_q_bias", type=float, default=1.0)
    parser.add_argument("--sigma_out", type=float, default=1.0)
    parser.add_argument("--sigma_out_bias", type=float, default=1.0)

    parser.add_argument("--no_normalize_coeff", action="store_true",
                        help="Use raw coefficient values instead of train-style normalization.")
    parser.add_argument("--selfconj_fix", action="store_true",
                        help=("Use the exact irfft2-realized spectral operator "
                              "(self-conjugate column halving + k2 off-by-one "
                              "fix) instead of the symmetric lambda_mask_2d. "
                              "Default off reproduces the original spectrum."))
    parser.add_argument("--spectral_ntk_dof2", action="store_true",
                        help=("With --selfconj_fix, apply the realized factor-2 "
                              "on the spectral NTK parameter-block term (each "
                              "free rfft-half weight couples to mode + Hermitian "
                              "mirror). Validated to bring the conv NTK groups "
                              "to the model. No effect without --selfconj_fix."))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    resolutions = parse_resolutions(args.resolutions)
    chunk_overrides = parse_chunks(args.chunks)
    layer_specs = parse_layer_specs(args.layer_specs_json)
    if layer_specs is None:
        layer_specs = build_fno2d_ntk_layer_specs(
            num_fourier_layers=args.num_fourier_layers,
            head_layers=args.head_layers,
            modes=args.modes,
            modes1=args.modes1,
            modes2=args.modes2,
            sigma_u=args.sigma_u,
            sigma_w=args.sigma_w,
            sigma_b=args.sigma_b,
            sigma_q=args.sigma_q,
            sigma_q_bias=args.sigma_q_bias,
            final_fourier_relu=args.final_fourier_relu,
        )
    if not layer_specs:
        raise ValueError("At least one nonlinear layer is required.")

    modes1 = int(args.modes if args.modes1 is None else args.modes1)
    modes2 = int(modes1 if args.modes2 is None else args.modes2)

    payload = {
        "format": np.array("full_2d_fft_spectrum"),
        "architecture": np.array("FNO2d_NTK_general"),
        "resolutions": np.asarray(resolutions, dtype=np.int64),
        "modes": np.int64(args.modes),
        "modes1": np.int64(modes1),
        "modes2": np.int64(modes2),
        "num_fourier_layers": np.int64(args.num_fourier_layers),
        "head_layers": np.int64(args.head_layers),
        "final_fourier_relu": np.int64(1 if args.final_fourier_relu else 0),
        "layer_specs": layer_specs_metadata(layer_specs),
        "sigma_p": np.float64(args.sigma_p),
        "sigma_p_bias": np.float64(args.sigma_p_bias),
        "sigma_out": np.float64(args.sigma_out),
        "sigma_out_bias": np.float64(args.sigma_out_bias),
        "add_coords": np.int64(0),
        "normalize_coeff": np.int64(0 if args.no_normalize_coeff else 1),
        "max_samples": np.int64(args.max_samples),
        "selfconj_fix": np.int64(1 if args.selfconj_fix else 0),
        "spectral_ntk_dof2": np.int64(1 if args.spectral_ntk_dof2 else 0),
    }

    for resolution in resolutions:
        print(f"\n=== r={resolution} ===", flush=True)
        inputs, source_path = load_darcy_inputs(
            args.data_dir,
            resolution,
            max_samples=args.max_samples,
            normalize_coeff=not args.no_normalize_coeff,
        )
        in_channels = inputs.shape[1]
        print(f"  loaded inputs {inputs.shape} from {source_path}", flush=True)

        input_cov = compute_input_covariance(inputs)
        C0 = initial_feature_covariance(
            input_cov,
            in_channels=in_channels,
            sigma_p=args.sigma_p,
            sigma_p_bias=args.sigma_p_bias,
        )
        fwd = forward_covariance_2d(C0, layer_specs, fix=args.selfconj_fix)
        chunk = chunk_overrides.get(resolution, default_chunk_size(resolution))

        start = time.time()
        _, K_hat = compute_ntk_spectrum_2d(
            fwd,
            layer_specs,
            input_covariance=input_cov,
            in_channels=in_channels,
            sigma_p=args.sigma_p,
            sigma_p_bias=args.sigma_p_bias,
            sigma_out=args.sigma_out,
            sigma_out_bias=args.sigma_out_bias,
            chunk=chunk,
            verbose=not args.quiet,
            fix=args.selfconj_fix,
            spectral_ntk_dof2=args.spectral_ntk_dof2,
        )
        elapsed = time.time() - start

        payload[f"K_hat_{resolution}"] = K_hat.astype(np.float64)
        payload[f"source_path_{resolution}"] = np.array(str(source_path))
        payload[f"input_channels_{resolution}"] = np.int64(in_channels)
        print(f"  K_hat shape={K_hat.shape} min={K_hat.min():.6g} "
              f"max={K_hat.max():.6g} chunk={chunk} time={elapsed:.1f}s",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
