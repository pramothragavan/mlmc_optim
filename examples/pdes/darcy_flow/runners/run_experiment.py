#!/usr/bin/env python3
"""Darcy multiresolution experiment runner.

Builds a
normal mlmc_optim config, then calls the existing example utilities and
MLMCTrainer directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from examples.examples_src.utils.config import str2bool


def parse_json_arg(raw, name):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {name} as JSON: {exc}") from exc


def json_ready(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [json_ready(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    return str(obj)


def detect_device(requested):
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and hasattr(xpu, "is_available") and xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_phase_lengths(epochs, c2f_res, subsets, levels, batch_sizes, lrs):
    n_phase = len(epochs)
    for name, value in [
        ("c2f_res_per_phase", c2f_res),
        ("subset_size_per_phase", subsets),
        ("levels_per_phase", levels),
    ]:
        if len(value) != n_phase:
            raise ValueError(
                f"{name} has length {len(value)} but epochs_per_phase has length {n_phase}."
            )
    if batch_sizes is not None and len(batch_sizes) != n_phase:
        raise ValueError("batch_size_per_phase length must match epochs_per_phase.")
    if lrs is not None and len(lrs) != n_phase:
        raise ValueError("lr_per_phase length must match epochs_per_phase.")

    for phase, (res, sample) in enumerate(zip(c2f_res, subsets)):
        if len(res) != len(sample):
            raise ValueError(
                f"phase {phase}: c2f_res_per_phase and subset_size_per_phase lengths differ."
            )
    if batch_sizes is not None:
        for phase, (res, batch) in enumerate(zip(c2f_res, batch_sizes)):
            if len(res) != len(batch):
                raise ValueError(
                    f"phase {phase}: c2f_res_per_phase and batch_size_per_phase lengths differ."
                )


def per_resolution_defaults(unique_res, c2f_res_per_phase, values_per_phase, fallback):
    values = {int(res): int(fallback) for res in unique_res}
    for phase_res, phase_values in zip(c2f_res_per_phase, values_per_phase):
        for res, value in zip(phase_res, phase_values):
            values[int(res)] = max(values[int(res)], int(value))
    return [values[int(res)] for res in unique_res]


def build_config(args):
    epochs_per_phase = [int(x) for x in parse_json_arg(
        args.epochs_per_phase_json, "epochs_per_phase_json")]
    c2f_res_per_phase = [
        [int(r) for r in phase]
        for phase in parse_json_arg(args.c2f_res_per_phase_json, "c2f_res_per_phase_json")
    ]
    subset_size_per_phase = [
        [int(n) for n in phase]
        for phase in parse_json_arg(args.subset_size_per_phase_json, "subset_size_per_phase_json")
    ]
    levels_per_phase = [int(x) for x in parse_json_arg(
        args.levels_per_phase_json, "levels_per_phase_json")]

    batch_size_per_phase = None
    if args.batch_size_per_phase_json is not None:
        batch_size_per_phase = [
            [int(n) for n in phase]
            for phase in parse_json_arg(
                args.batch_size_per_phase_json, "batch_size_per_phase_json")
        ]

    lr_per_phase = None
    if args.lr_per_phase_json is not None:
        lr_per_phase = [
            float(x) for x in parse_json_arg(args.lr_per_phase_json, "lr_per_phase_json")
        ]

    validate_phase_lengths(
        epochs_per_phase,
        c2f_res_per_phase,
        subset_size_per_phase,
        levels_per_phase,
        batch_size_per_phase,
        lr_per_phase,
    )

    unique_res = sorted({int(r) for phase in c2f_res_per_phase for r in phase})
    batch_defaults = (
        per_resolution_defaults(unique_res, c2f_res_per_phase, batch_size_per_phase, 0)
        if batch_size_per_phase is not None
        else [int(args.batch_size)] * len(unique_res)
    )

    out_dir = Path(args.out_dir)
    config = {
        "dataset": "darcy",
        "model": "fno",
        "device": detect_device(args.device),
        "seed": int(args.seed),
        "data_dir": str(args.data_dir),
        "base_res": int(args.base_res),
        "override_res": None,
        "c2f_resolutions": unique_res,
        "total_samples": int(args.total_samples),
        "train_subset": float(args.train_subset),
        "test_subset": float(args.test_subset),
        "normalize": bool(args.normalize),
        "normalize_input": bool(args.normalize_input) if args.normalize_input is not None else bool(args.normalize),
        "normalize_output": bool(args.normalize_output) if args.normalize_output is not None else bool(args.normalize),
        "add_coords": bool(args.add_coords),
        "use_grads": bool(args.darcy_use_grads),
        "load_in_memory": not args.no_load_in_memory,
        "load_gpu": bool(args.load_gpu),
        "load_gpu_epoch": bool(args.load_gpu_epoch),
        "pin_memory": bool(args.pin_memory),
        "out_channels": 1,
        "fno_modes": int(args.fno_modes),
        "fno_width": int(args.fno_width),
        "fno_final_fourier_relu": bool(args.fno_final_fourier_relu),
        "fno_head_width": args.fno_head_width,
        "optimizer": args.optimizer,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "momentum": float(args.momentum),
        "use_scheduler": bool(args.use_scheduler),
        "scheduler": "step",
        "step_size": int(args.step_size),
        "gamma": float(args.gamma),
        "loss_type": args.loss_type,
        "loss_reduction": args.loss_reduction,
        "batch_size": int(args.batch_size),
        "epochs": int(sum(epochs_per_phase)),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "eval_every": int(args.eval_every),
        "eval_plot_every": None,
        "eval_grad_every": None,
        "eval_grad_rand": False,
        "eval_grad_mode": "raw",
        "eval_grad_batches": None,
        "mlmc_pairing": "hierarchy",
        "mlmc_sampling_style": "prescribed",
        "mlmc_samples_per_level": per_resolution_defaults(
            unique_res, c2f_res_per_phase, subset_size_per_phase, 0),
        "mlmc_batch_style": "prescribed_batch",
        "mlmc_batch_sizes": batch_defaults,
        "mlmc_data_size_multiplier": 2,
        "mlmc_batch_size_multiplier": 2,
        "mlmc_max_level": len(unique_res),
        "mlmc_deterministic_batching": bool(args.deterministic_batching),
        "mlmc_hierarchy_cache": bool(args.mlmc_hierarchy_cache),
        "epochs_per_phase": epochs_per_phase,
        "levels_per_phase": levels_per_phase,
        "c2f_res_per_phase": c2f_res_per_phase,
        "subset_size_per_phase": subset_size_per_phase,
        "batch_size_per_phase": batch_size_per_phase,
        "lr_per_phase": lr_per_phase,
        "save_model": False,
        "save_checkpoint": False,
        "save_wandb_artifact": False,
        "checkpoint_dir": str(out_dir / "checkpoints"),
        "save_final_checkpoint": bool(args.save_final_checkpoint),
        "out_dir": str(out_dir),
        "output_dir": str(out_dir),
        "spectral_diagnostics": not args.no_spectral,
        "spectral_out_dir": str(out_dir / "spectral"),
        "use_wandb": bool(args.use_wandb),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_group": args.wandb_group,
        "wandb_run_name": args.wandb_run_name,
    }

    if lr_per_phase is not None:
        config["lr"] = float(lr_per_phase[0])

    if args.extra_config_json is not None:
        extra = parse_json_arg(args.extra_config_json, "extra_config_json")
        if not isinstance(extra, dict):
            raise ValueError("--extra_config_json must parse to a JSON object.")
        config.update(extra)

    return config


def make_denorm(mean, std):
    def denorm(x):
        return x * (std + 1e-5) + mean

    return denorm


def eval_norm_stats(dataset_specific, config):
    if not config.get("normalize_output", True):
        return None
    mean = dataset_specific.get("eval_output_mean")
    std = dataset_specific.get("eval_output_std")
    if mean is None or std is None:
        return None
    return mean, std


def make_denormalizers(train_datasets, config, dataset_specific=None):
    if not config.get("normalize_output", True) or config.get("model") not in ["fno", "fno3d"]:
        return None

    denormalizers = {}
    device = config["device"]
    for res, dataset in train_datasets.items():
        if hasattr(dataset, "output_mean") and hasattr(dataset, "output_std"):
            mean = dataset.output_mean.to(device)
            std = dataset.output_std.to(device)

            denormalizers[res] = make_denorm(mean, std)

    if dataset_specific is not None:
        stats = eval_norm_stats(dataset_specific, config)
        if stats is not None:
            mean, std = stats
            denormalizers[config["base_res"]] = make_denorm(
                mean.to(device), std.to(device))
    return denormalizers


def run(config):
    sys.path.insert(0, str(REPO))

    from mlmc_optim import MLMCTrainer
    from examples.examples_src.models.model_utils import get_model
    from examples.examples_src.utils.data_utils import get_datasets
    from examples.examples_src.utils.eval_utils import evaluate_model, get_plot_fn
    from examples.examples_src.utils.train_utils import (
        get_criterion,
        get_data_loader,
        setup_optimizers,
    )
    from examples.examples_src.utils.utils import set_seed

    set_seed(config["seed"])

    train_datasets, test_datasets, input_channels, input_size, dataset_specific = get_datasets(
        config, config["c2f_resolutions"], config["device"])
    model = get_model(config, input_channels=input_channels, input_size=input_size)
    optimizer, scheduler = setup_optimizers(model, config)
    criterion = get_criterion(config)
    eval_criterion = get_criterion(config, for_eval=True)
    denormalizers = make_denormalizers(train_datasets, config, dataset_specific)
    eval_stats = eval_norm_stats(dataset_specific, config)
    eval_loader = get_data_loader(
        test_datasets[config["base_res"]], config, shuffle=False)
    plot_fn = get_plot_fn(config)

    trainer = MLMCTrainer(
        model=model,
        c2f_resolutions=config["c2f_resolutions"],
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        eval_criterion=eval_criterion,
        eval_fn=evaluate_model,
        plot_fn=plot_fn,
        eval_loader=eval_loader,
        grad_eval_fn=None,
        denormalizers=denormalizers,
        eval_norm_stats=eval_stats,
        device=config["device"],
        sample_sizes=config["mlmc_samples_per_level"],
        batch_sizes=config["mlmc_batch_sizes"],
        seed=config["seed"],
        epochs=config["epochs"],
        eval_every=config["eval_every"],
        use_wandb=config["use_wandb"],
        config=config,
    )
    return trainer.fit()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Darcy baseline or multiresolution schedules.")
    parser.add_argument("--condition", choices=["multires"], default="multires")
    parser.add_argument("--data_dir", type=Path, default=Path("examples/pdes/darcy_flow/data"))
    parser.add_argument("--base_res", type=int, default=241)
    parser.add_argument("--total_samples", type=int, default=1024)
    parser.add_argument("--train_subset", type=float, default=1.0)
    parser.add_argument("--test_subset", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps", "xpu"], default="auto")

    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--use_scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--loss_type", choices=["lp", "mse"], default="lp")
    parser.add_argument("--loss_reduction", choices=["mean", "sum"], default="sum")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--fno_width", type=int, default=32)
    parser.add_argument("--fno_modes", type=int, default=8)
    parser.add_argument("--fno_final_fourier_relu", action="store_true")
    parser.add_argument("--fno_head_width", type=int, default=None)
    parser.add_argument("--add_coords", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--darcy_use_grads", action="store_true")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--normalize_input", type=str2bool, default=None)
    parser.add_argument("--normalize_output", type=str2bool, default=None)

    parser.add_argument("--epochs_per_phase_json", default="[50, 50, 50, 50, 50]")
    parser.add_argument("--c2f_res_per_phase_json", default="[[15], [30], [60], [120], [241]]")
    parser.add_argument("--subset_size_per_phase_json", default="[[1024], [1024], [1024], [1024], [1024]]")
    parser.add_argument("--levels_per_phase_json", default="[1, 1, 1, 1, 1]")
    parser.add_argument("--batch_size_per_phase_json", default=None)
    parser.add_argument("--lr_per_phase_json", default=None)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--load_gpu", action="store_true")
    parser.add_argument("--load_gpu_epoch", action="store_true")
    parser.add_argument("--no_load_in_memory", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--deterministic_batching", action="store_true")
    parser.add_argument("--mlmc_hierarchy_cache", action="store_true")
    parser.add_argument("--no_spectral", action="store_true")
    parser.add_argument("--save_final_checkpoint", action="store_true")
    parser.add_argument("--extra_config_json", default=None)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="mlmc-optim-darcy")
    parser.add_argument("--wandb_entity", default="local")
    parser.add_argument("--wandb_group", default="darcy")
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args(argv)

    config = build_config(args)
    print(json.dumps(json_ready(config), indent=2))

    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)
    test_loss, _ = run(config)
    print(f"Saved outputs to {config['out_dir']}")
    print(f"Final reported test loss: {test_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
