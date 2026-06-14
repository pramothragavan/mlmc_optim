#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from . import darcy_ntk_optimal_allocator as base
except ImportError:  # Allow direct script execution.
    import darcy_ntk_optimal_allocator as base


DEFAULT_LEVELS = [15, 30, 60, 120, 241]
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_BUDGETS = [200, 500, 1000, 2000]
DEFAULT_BATCH_SIZES = {15: 160, 30: 80, 60: 80, 120: 20, 241: 20}


def parse_int_list(raw):
    return [int(x) for x in json.loads(raw)]


def parse_int_map(raw, default):
    if raw is None:
        return dict(default)
    data = json.loads(raw)
    return {int(k): int(v) for k, v in data.items()}


def parse_float_map(raw):
    if raw is None:
        return None
    data = json.loads(raw)
    return {int(k): float(v) for k, v in data.items()}


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def spectral_path(root, template, level, seed):
    return Path(root) / template.format(level=level, seed=seed) / "spectral" / "spectral_data.npz"


def summary_path(root, template, level, seed):
    return Path(root) / template.format(level=level, seed=seed) / "summary.json"


def load_seed_averaged_problem(args, levels, seeds, batch_sizes):
    seed_curves = {}
    seed_summary_rows = []
    spectral_shape = None
    spectral_paths = {}
    summary_paths = {}

    for level in levels:
        seed_curves[level] = []
        spectral_paths[level] = []
        summary_paths[level] = []
        level_epochs = None

        for seed in seeds:
            spec_path = spectral_path(args.spectral_root, args.run_template, level, seed)
            summ_path = summary_path(args.spectral_root, args.run_template, level, seed)
            if not spec_path.exists():
                raise FileNotFoundError(spec_path)
            if not summ_path.exists():
                raise FileNotFoundError(summ_path)

            spec = np.load(spec_path, allow_pickle=True)
            if "power_spectra_2d" not in spec.files:
                raise KeyError(f"{spec_path} is missing power_spectra_2d")
            power = np.asarray(spec["power_spectra_2d"], dtype=np.float64)
            if power.ndim != 3:
                raise ValueError(
                    f"{spec_path}: power_spectra_2d has shape {power.shape}; "
                    "expected (n_eval,H,H//2+1)")
            shape = power.shape[1:]
            if shape[1] != shape[0] // 2 + 1:
                raise ValueError(f"{spec_path}: non-rfft spectral shape {shape}")
            if spectral_shape is None:
                spectral_shape = shape
            elif shape != spectral_shape:
                raise ValueError(
                    f"{spec_path}: spectral shape {shape} differs from {spectral_shape}")

            epochs = np.asarray(spec["epochs"], dtype=np.int64)
            if level_epochs is None:
                level_epochs = epochs
            elif not np.array_equal(level_epochs, epochs):
                raise ValueError(f"{spec_path}: epoch grid differs within r={level}")

            seed_curves[level].append(power.reshape(power.shape[0], -1))
            spectral_paths[level].append(str(spec_path))
            summary_paths[level].append(str(summ_path))

            with open(summ_path) as f:
                summary = json.load(f)
            completed = int(summary["final_epoch_completed"])
            train_time = float(summary["final_eval_cum_train_time"])
            seed_summary_rows.append({
                "level": level,
                "seed": seed,
                "epochs": completed,
                "n_eval_rows": int(summary.get("n_eval_rows", -1)),
                "cost_s_per_epoch": train_time / completed,
                "final_test_loss": float(summary["final_test_loss"]),
                "best_test_loss": float(summary["best_test_loss"]),
                "spectral_path": str(spec_path),
                "summary_path": str(summ_path),
            })

    P0 = np.mean(
        np.stack([curve[0] for curve in seed_curves[levels[0]]], axis=0),
        axis=0,
    )
    F_raw = {
        level: np.mean(
            np.stack([
                base.estimate_floor(curve, args.tail_frac)
                for curve in seed_curves[level]
            ], axis=0),
            axis=0,
        )
        for level in levels
    }
    F = base.enforce_monotone(F_raw, levels)

    inferred_costs = {}
    scalar_summary = {}
    for level in levels:
        rows = [row for row in seed_summary_rows if row["level"] == level]
        cost_values = [row["cost_s_per_epoch"] for row in rows]
        inferred_costs[level] = float(np.mean(cost_values))
        scalar_summary[level] = {
            "n_seeds": len(rows),
            "epochs": sorted(set(row["epochs"] for row in rows)),
            "cost_s_per_epoch_mean": inferred_costs[level],
            "cost_s_per_epoch_sd": (
                float(np.std(cost_values, ddof=1)) if len(cost_values) > 1 else 0.0),
            "best_test_loss_mean": float(np.mean([row["best_test_loss"] for row in rows])),
            "best_test_loss_sd": (
                float(np.std([row["best_test_loss"] for row in rows], ddof=1))
                if len(rows) > 1 else 0.0),
            "final_test_loss_mean": float(np.mean([row["final_test_loss"] for row in rows])),
            "final_test_loss_sd": (
                float(np.std([row["final_test_loss"] for row in rows], ddof=1))
                if len(rows) > 1 else 0.0),
        }

    costs = parse_float_map(args.costs_json) or inferred_costs
    for level in levels:
        if level not in costs:
            raise KeyError(f"Missing cost for r={level}")
        if level not in batch_sizes:
            raise KeyError(f"Missing batch size for r={level}")

    return {
        "P0": P0,
        "F": F,
        "costs": costs,
        "inferred_costs": inferred_costs,
        "scalar_summary": scalar_summary,
        "seed_summary_rows": seed_summary_rows,
        "spectral_shape": spectral_shape,
        "spectral_paths": spectral_paths,
        "summary_paths": summary_paths,
    }


def nearest_integer_rows(fractional_rows, costs, levels):
    rows = []
    for row in fractional_rows:
        budget = float(row["B"])
        out = {"B": budget}
        total_wall = 0.0
        for level in levels:
            epochs = int(np.floor(float(row[f"ep_{level}"]) + 0.5))
            wall = epochs * float(costs[level])
            out[f"ep_{level}"] = epochs
            out[f"wall_{level}"] = wall
            total_wall += wall
        out["total_wall"] = total_wall
        out["budget_delta"] = total_wall - budget
        rows.append(out)
    return rows


def equal_epoch_rows(budgets, costs, levels):
    rows = []
    cost_per_equal_epoch = sum(float(costs[level]) for level in levels)
    for budget in budgets:
        n_epochs = int(np.floor(float(budget) / cost_per_equal_epoch + 0.5))
        out = {
            "B": float(budget),
            "equal_epochs": n_epochs,
        }
        total_wall = 0.0
        for level in levels:
            wall = n_epochs * float(costs[level])
            out[f"ep_{level}"] = n_epochs
            out[f"wall_{level}"] = wall
            total_wall += wall
        out["total_wall"] = total_wall
        out["budget_delta"] = total_wall - float(budget)
        rows.append(out)
    return rows


def comparison_rows(new_rows, old_csv, levels):
    if old_csv is None or not old_csv.exists():
        return []

    with old_csv.open(newline="") as f:
        old_by_budget = {
            float(row["B"]): {k: float(v) for k, v in row.items()}
            for row in csv.DictReader(f)
        }

    rows = []
    for new in new_rows:
        budget = float(new["B"])
        if budget not in old_by_budget:
            continue
        old = old_by_budget[budget]
        comp = {
            "B": budget,
            "old_pred_loss": old["pred_loss"],
            "new_pred_loss": float(new["pred_loss"]),
        }
        for level in levels:
            old_ep_key = f"ep_{level}"
            old_wall_key = f"wall_{level}"
            if old_ep_key not in old and level == 241:
                old_ep_key = "ep_240"
                old_wall_key = "wall_240"
            comp[f"old_ep_{level}"] = old.get(old_ep_key, float("nan"))
            comp[f"new_ep_{level}"] = float(new[f"ep_{level}"])
            comp[f"delta_ep_{level}"] = comp[f"new_ep_{level}"] - comp[f"old_ep_{level}"]
            comp[f"old_wall_{level}"] = old.get(old_wall_key, float("nan"))
            comp[f"new_wall_{level}"] = float(new[f"wall_{level}"])
            comp[f"delta_wall_{level}"] = comp[f"new_wall_{level}"] - comp[f"old_wall_{level}"]
        rows.append(comp)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate seed-averaged Darcy full-2D NTK-optimal schedules.")
    parser.add_argument(
        "--spectral_root", type=Path,
        default=Path("results/NEW_darcy_constlr_spectral_baselines_maxlen"))
    parser.add_argument(
        "--run_template", default="baseline_r{level}_seed{seed}",
        help="Directory template under spectral_root.")
    parser.add_argument(
        "--ntk_path", type=Path,
        default=Path("examples/pdes/darcy_flow/data/darcy_ntk_spectrum_2d_r241_rawcoeff_no_coords_norelu_selfconj_dof2.npz"))
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path("results/NEW_darcy_ntk_optimal_allocator_tripleseed"))
    parser.add_argument("--levels_json", default=json.dumps(DEFAULT_LEVELS))
    parser.add_argument("--seeds_json", default=json.dumps(DEFAULT_SEEDS))
    parser.add_argument("--budgets_json", default=json.dumps(DEFAULT_BUDGETS))
    parser.add_argument("--batch_sizes_json", default=json.dumps(DEFAULT_BATCH_SIZES))
    parser.add_argument(
        "--costs_json", default=None,
        help="Optional JSON dict keyed by resolution; otherwise infer from summaries.")
    parser.add_argument(
        "--compare_csv", type=Path,
        default=Path("results/darcy_ntk_optimal_allocator_r241_seed42/ntk_optimal.csv"))
    parser.add_argument("--eval_grid", type=int, default=241)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=1024)
    parser.add_argument("--tail_frac", type=float, default=0.10)
    args = parser.parse_args(argv)

    levels = parse_int_list(args.levels_json)
    seeds = parse_int_list(args.seeds_json)
    budgets = [float(x) for x in json.loads(args.budgets_json)]
    batch_sizes = parse_int_map(args.batch_sizes_json, DEFAULT_BATCH_SIZES)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    problem = load_seed_averaged_problem(args, levels, seeds, batch_sizes)
    spectral_shape = problem["spectral_shape"]
    if int(args.eval_grid) != int(spectral_shape[0]):
        raise ValueError(
            f"--eval_grid={args.eval_grid} but spectral_data uses H={spectral_shape[0]}")

    K_hat = base.load_ntk_spectra(args.ntk_path, levels)
    _, _, K1, K2 = base.rfft_mode_bins(args.eval_grid, args.eval_grid // 2 + 1)
    weights = base.rfft_feature_weights(args.eval_grid)
    rates = base.ntk_rates_alias_lifted_full(
        K_hat, K1, K2, levels, args.lr, args.subset, batch_sizes)

    fractional_rows = base.solve_for_budgets(
        "Darcy seed-averaged NTK-optimal alias-lifted",
        rates,
        problem["F"],
        problem["P0"],
        weights,
        problem["costs"],
        budgets,
        levels,
    )
    integer_rows = nearest_integer_rows(fractional_rows, problem["costs"], levels)
    equal_rows = equal_epoch_rows(budgets, problem["costs"], levels)

    write_rows(args.out_dir / "ntk_optimal.csv", fractional_rows)
    write_rows(args.out_dir / "ntk_optimal_integer_schedules.csv", integer_rows)
    write_rows(args.out_dir / "equal_epochs_integer_schedules.csv", equal_rows)
    write_rows(args.out_dir / "seed_summary.csv", problem["seed_summary_rows"])

    comp = comparison_rows(fractional_rows, args.compare_csv, levels)
    if comp:
        write_rows(args.out_dir / "comparison_to_previous_allocator.csv", comp)

    with open(args.out_dir / "metadata.json", "w") as f:
        json.dump({
            "allocator": "darcy_ntk_optimal_alias_lifted_full_2d",
            "schedule_generator": Path(__file__).name,
            "rounding": "nearest_integer_per_level_no_budget_repair",
            "averaging": (
                "P0 is averaged across seed initial spectra; floors are the mean "
                "of per-seed tail-median full-2D spectra, then monotone-enforced "
                "across levels; costs are mean train seconds per epoch across seeds."),
            "spectral_root": str(args.spectral_root),
            "run_template": args.run_template,
            "ntk_path": str(args.ntk_path),
            "levels": levels,
            "seeds": seeds,
            "budgets": budgets,
            "batch_sizes": {str(k): int(v) for k, v in batch_sizes.items()},
            "costs_s_per_epoch": {
                str(k): float(v) for k, v in problem["costs"].items()},
            "inferred_costs_s_per_epoch": {
                str(k): float(v) for k, v in problem["inferred_costs"].items()},
            "lr": float(args.lr),
            "subset": int(args.subset),
            "tail_frac": float(args.tail_frac),
            "eval_grid": int(args.eval_grid),
            "spectral_shape": list(spectral_shape),
            "scalar_baseline_summary": {
                str(k): v for k, v in problem["scalar_summary"].items()},
            "spectral_paths": {
                str(k): v for k, v in problem["spectral_paths"].items()},
            "summary_paths": {
                str(k): v for k, v in problem["summary_paths"].items()},
        }, f, indent=2)

    print(f"\nWrote {args.out_dir}")
    print("\nNearest-integer schedules:")
    for row in integer_rows:
        schedule = [row[f"ep_{level}"] for level in levels]
        print(
            f"  B={row['B']:.0f}: {schedule} "
            f"wall={row['total_wall']:.2f}s delta={row['budget_delta']:+.2f}s")
    print("\nEqual-epochs schedules:")
    for row in equal_rows:
        schedule = [row[f"ep_{level}"] for level in levels]
        print(
            f"  B={row['B']:.0f}: {schedule} "
            f"wall={row['total_wall']:.2f}s delta={row['budget_delta']:+.2f}s")


if __name__ == "__main__":
    main()
