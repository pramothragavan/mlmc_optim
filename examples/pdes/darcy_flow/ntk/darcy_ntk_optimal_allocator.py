#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_LEVELS = [15, 30, 60, 120, 241]
DEFAULT_COSTS = {
    15: 0.22414207458496094,
    30: 0.4220454692840576,
    60: 0.46949100494384766,
    120: 2.3204739093780518,
    241: 6.1506054401397705,
}
DEFAULT_BATCH_SIZES = {15: 160, 30: 80, 60: 80, 120: 20, 241: 20}
DEFAULT_BUDGETS = [200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000]
EPS = 1e-30
COEF_FLOOR = 1e-30


def parse_int_map(raw, default):
    if raw is None:
        return dict(default)
    data = json.loads(raw)
    return {int(k): v for k, v in data.items()}


def parse_int_list(raw):
    return [int(x) for x in json.loads(raw)]


def rfft_mode_bins(H, n_bins):
    """Return rfft2 mode coordinates, radial bins, and bin weights."""
    Wh = H // 2 + 1
    k1 = np.arange(H)
    k1 = np.where(k1 > H // 2, k1 - H, k1)
    k2 = np.arange(Wh)
    K1, K2 = np.meshgrid(k1, k2, indexing="ij")
    K = np.sqrt(K1 ** 2 + K2 ** 2)

    weights = np.zeros(n_bins)
    bin_assignment = np.zeros((H, Wh), dtype=int)
    for i in range(H):
        for j in range(Wh):
            b = int(round(K[i, j]))
            if b < n_bins:
                weights[b] += 1
            bin_assignment[i, j] = b

    return weights, bin_assignment, K1, K2


def rfft_feature_weights(H):
    Wh = H // 2 + 1
    weights = np.ones((H, Wh), dtype=np.float64)
    if H % 2 == 0:
        weights[:, 1:H // 2] = 2.0
    else:
        weights[:, 1:] = 2.0
    return weights.reshape(-1)


def aggregate_by_bin(values, bin_assignment, weights, n_bins):
    totals = np.zeros(n_bins, dtype=np.float64)
    valid = bin_assignment < n_bins
    np.add.at(totals, bin_assignment[valid], values[valid])
    out = np.zeros(n_bins, dtype=np.float64)
    nonempty = weights > 0
    out[nonempty] = totals[nonempty] / weights[nonempty]
    return out


def signed_alias(k, level):
    return ((k + level // 2) % level) - level // 2


def load_ntk_spectra(path, levels):
    ntk = np.load(path, allow_pickle=True)
    spectra = {}
    for level in levels:
        full_key = f"K_hat_{level}"
        if full_key not in ntk:
            raise KeyError(
                f"{path} is missing {full_key}; regenerate full spectra with "
                "examples/pdes/darcy_flow/ntk/generate_darcy_ntk_spectrum.py.")
        K = np.asarray(ntk[full_key], dtype=np.float64)
        if K.shape != (level, level):
            raise ValueError(
                f"{path}:{full_key} has shape {K.shape}, expected "
                f"({level}, {level})")
        spectra[level] = K
    return spectra


def load_full_spectral_curves(spec_data, levels):
    curves = {}
    spectral_shape = None
    for level in levels:
        if "power_spectra_2d" not in spec_data[level].files:
            raise KeyError(
                f"spectral_data.npz for r={level} is missing "
                "power_spectra_2d; rerun spectral diagnostics")
        power = np.asarray(spec_data[level]["power_spectra_2d"],
                           dtype=np.float64)
        if power.ndim != 3:
            raise ValueError(
                f"power_spectra_2d for r={level} has shape {power.shape}; "
                "expected (n_eval, H, H//2+1).")
        shape = power.shape[1:]
        if shape[1] != shape[0] // 2 + 1:
            raise ValueError(
                f"power_spectra_2d for r={level} has non-rfft shape {shape}.")
        if spectral_shape is None:
            spectral_shape = shape
        elif shape != spectral_shape:
            raise ValueError(
                f"power_spectra_2d shape mismatch for r={level}: {shape} "
                f"vs {spectral_shape}")
        curves[level] = power.reshape(power.shape[0], -1)
    return curves, spectral_shape


def estimate_floor(curves, tail_frac):
    n_tail = max(2, int(round(float(tail_frac) * curves.shape[0])))
    return np.median(curves[-n_tail:], axis=0)


def enforce_monotone(F, levels):
    out = {l: F[l].astype(np.float64).copy() for l in levels}
    for i in range(len(levels) - 1, 0, -1):
        out[levels[i - 1]] = np.maximum(out[levels[i - 1]], out[levels[i]])
    return out


def estimate_rate_local_median(t, P, F):
    diff_t = np.diff(t)
    diff_P = np.diff(P)
    prev = P[:-1] - F
    nxt = P[1:] - F
    mask = (prev > EPS) & (nxt > EPS) & (diff_P < 0)
    if mask.sum() < 3:
        return np.nan
    dlog = np.log(nxt[mask]) - np.log(prev[mask])
    return float(np.median(-dlog / (2.0 * diff_t[mask])))


def empirical_rates_for_features(spec_data, curves, F, levels):
    rates = {}
    for level in levels:
        ep = spec_data[level]["epochs"]
        values = curves[level]
        rates[level] = np.asarray([
            estimate_rate_local_median(ep, values[:, b], F[level][b])
            for b in range(values.shape[1])
        ])
    return rates


def ntk_values_same_mode(K_hat, K1, K2, level):
    half = level // 2
    representable = (np.abs(K1) <= half) & (np.abs(K2) <= half)
    vals = np.zeros_like(K1, dtype=np.float64)
    vals[representable] = K_hat[level][
        np.mod(K1[representable], level),
        np.mod(K2[representable], level)]
    return vals


def ntk_values_alias_lifted(K_hat, K1, K2, level):
    alias_k1 = signed_alias(K1, level)
    alias_k2 = signed_alias(K2, level)
    return K_hat[level][np.mod(alias_k1, level), np.mod(alias_k2, level)]


def ntk_rates_same_mode_full(K_hat, K1, K2, levels, lr, subset, batch_sizes):
    rates = {}
    for level in levels:
        steps = int(np.ceil(subset / int(batch_sizes[level])))
        vals = ntk_values_same_mode(K_hat, K1, K2, level).reshape(-1)
        rates[level] = 2.0 * lr * steps * vals / (level * level)
    return rates


def ntk_rates_alias_lifted_full(K_hat, K1, K2, levels, lr, subset,
                                batch_sizes):
    rates = {}
    for level in levels:
        steps = int(np.ceil(subset / int(batch_sizes[level])))
        vals = ntk_values_alias_lifted(K_hat, K1, K2, level).reshape(-1)
        rates[level] = 2.0 * lr * steps * vals / (level * level)
    return rates


def build_floored_weighted(F, P0, rates, weights, levels,
                           coef_floor=COEF_FLOOR):
    n_level = len(levels)
    n_features = len(P0)
    log_coefs = []
    A = []

    const = float(np.sum(weights * F[levels[-1]]))
    if const > coef_floor:
        log_coefs.append(np.log(const))
        A.append([0.0] * n_level)

    for i in range(n_level - 1, -1, -1):
        base = F[levels[i - 1]] if i >= 1 else P0
        diff = base - F[levels[i]]
        for b in range(n_features):
            c = weights[b] * diff[b]
            if c <= coef_floor:
                continue
            row = [0.0] * n_level
            for j in range(i, n_level):
                rate = rates[levels[j]][b]
                row[j] = -2.0 * float(rate if np.isfinite(rate) else 0.0)
            log_coefs.append(float(np.log(c)))
            A.append(row)

    return np.asarray(log_coefs), np.asarray(A)


def radialise_rates_for_plot(rates, spectral_shape, n_bins, levels):
    weights, bin_assignment, _, _ = rfft_mode_bins(spectral_shape[0], n_bins)
    out = {}
    for level in levels:
        values = rates[level].reshape(spectral_shape)
        out[level] = aggregate_by_bin(values, bin_assignment, weights, n_bins)
    return out


def cvx_solve(log_coefs, A, costs_arr, budget, levels):
    if A.size == 0:
        return {level: 0.0 for level in levels}, float("nan")

    try:
        from scipy.optimize import minimize
        from scipy.special import logsumexp

        def objective_and_grad(T):
            z = log_coefs + A @ T
            value = logsumexp(z)
            probs = np.exp(z - value)
            grad = A.T @ probs
            return float(value), grad

        starts = []
        for i in range(len(levels)):
            x0 = np.zeros(len(levels), dtype=np.float64)
            x0[i] = float(budget) / max(float(costs_arr[i]), EPS)
            starts.append(x0)
        starts.append(float(budget) / (len(levels) * costs_arr))
        starts.append(np.ones(len(levels), dtype=np.float64)
                      * float(budget) / float(np.sum(costs_arr)))

        constraints = [{
            "type": "ineq",
            "fun": lambda T: float(budget) - costs_arr @ T,
            "jac": lambda T: -costs_arr,
        }]

        best = None
        for x0 in starts:
            result = minimize(
                lambda T: objective_and_grad(T)[0],
                x0,
                jac=lambda T: objective_and_grad(T)[1],
                method="SLSQP",
                bounds=[(0.0, None)] * len(levels),
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
            )
            if not result.success:
                continue
            T = np.maximum(result.x, 0.0)
            value = float(objective_and_grad(T)[0])
            if best is None or value < best[0]:
                best = (value, T)

        if best is not None:
            value, T = best
            return ({level: float(T[i])
                     for i, level in enumerate(levels)}, value)
    except Exception:
        pass

    import cvxpy as cp

    T = cp.Variable(len(levels), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.log_sum_exp(log_coefs + A @ T)),
                      [costs_arr @ T <= budget])
    try:
        prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200_000)
    except Exception:
        prob.solve(solver=cp.ECOS)
    return ({level: float(max(0.0, T.value[i]))
             for i, level in enumerate(levels)},
            float(prob.value))


def solve_for_budgets(label, rates, F, P0, weights, costs, budgets, levels):
    log_coefs, A = build_floored_weighted(F, P0, rates, weights, levels)
    costs_arr = np.asarray([float(costs[level]) for level in levels])
    rows = []
    print(f"\n--- {label} ---")
    print(f"{'B (s)':>8}  " + "  ".join(
        [f"ep_{level}".rjust(8) for level in levels]) + f"  {'pred':>10}")
    print("-" * (14 + 10 * len(levels) + 12))
    for budget in budgets:
        T, log_loss = cvx_solve(log_coefs, A, costs_arr, budget, levels)
        ep_str = "  ".join([f"{T[level]:8.0f}" for level in levels])
        pred = float(np.exp(log_loss))
        print(f"{budget:8.0f}  {ep_str}  {pred:10.3e}")
        rows.append({
            "B": float(budget),
            **{f"ep_{level}": T[level] for level in levels},
            **{f"wall_{level}": T[level] * float(costs[level])
               for level in levels},
            "pred_loss": pred,
        })
    return rows


def diagnostic_alias_plot(out_dir, p_emp, p_same, p_alias, n_bins, levels,
                          max_bin):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(levels), figsize=(4.4 * len(levels), 4.2),
                             sharey=True)
    if len(levels) == 1:
        axes = [axes]
    b = np.arange(n_bins)
    for ax, level in zip(axes, levels):
        ax.semilogy(b, np.where(p_emp[level] > 0, p_emp[level], np.nan),
                    "o-", lw=1.4, ms=3, label="empirical")
        ax.semilogy(b, np.where(p_same[level] > 0, p_same[level], np.nan),
                    "s-", lw=1.2, ms=3, alpha=0.75, label="same-mode NTK")
        ax.semilogy(b, np.where(p_alias[level] > 0, p_alias[level], np.nan),
                    "d-", lw=1.2, ms=3, alpha=0.75, label="alias-lifted NTK")
        ax.axvline(level // 2, color="0.5", ls=":", alpha=0.6)
        ax.set_title(f"r={level}")
        ax.set_xlabel("fine-grid radial bin")
        ax.set_xlim(0, max_bin)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("rate p_l(b) [1/epoch]")
    fig.tight_layout()
    out_path = Path(out_dir) / "alias_vs_empirical.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for row in rows:
            wr.writerow(row)


def spectral_path(level, spectral_base, spectral_long):
    candidates = [
        spectral_long / f"baseline_r{level}" / "spectral" / "spectral_data.npz",
        spectral_long / f"baseline_r{level}_seed42" / "spectral" / "spectral_data.npz",
        spectral_base / f"baseline_r{level}" / "spectral" / "spectral_data.npz",
        spectral_base / f"baseline_r{level}_seed42" / "spectral" / "spectral_data.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Solve Darcy empirical/same-mode/NTK-optimal allocations.")
    parser.add_argument("--spectral_base", type=Path,
                        default=Path("results/darcy_constlr_baselines_spectral"))
    parser.add_argument("--spectral_long", type=Path,
                        default=Path("results/darcy_constlr_baselines_spectral_long"))
    parser.add_argument("--ntk_path", type=Path,
                        default=Path("examples/pdes/darcy_flow/data/darcy_ntk_spectrum_2d.npz"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("results/darcy_ntk_optimal_allocator"))
    parser.add_argument("--levels_json", default=json.dumps(DEFAULT_LEVELS))
    parser.add_argument("--budgets_json", default=json.dumps(DEFAULT_BUDGETS))
    parser.add_argument("--costs_json", default=None,
                        help="JSON dict keyed by resolution, in seconds/epoch.")
    parser.add_argument("--batch_sizes_json", default=None,
                        help="JSON dict keyed by resolution.")
    parser.add_argument("--eval_grid", type=int, default=241)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=int, default=1024)
    parser.add_argument("--tail_frac", type=float, default=0.10)
    parser.add_argument("--max_plot_bin", type=int, default=60)
    parser.add_argument(
        "--only",
        choices=["all", "empirical", "same_mode_ntk", "ntk_optimal"],
        default="all",
        help="Limit which convex allocation CSVs are solved and written.")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args(argv)

    levels = parse_int_list(args.levels_json)
    budgets = [float(x) for x in json.loads(args.budgets_json)]
    costs = parse_int_map(args.costs_json, DEFAULT_COSTS)
    batch_sizes = parse_int_map(args.batch_sizes_json, DEFAULT_BATCH_SIZES)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = {}
    for level in levels:
        path = spectral_path(level, args.spectral_base, args.spectral_long)
        if not path.exists():
            raise FileNotFoundError(f"Missing spectral_data.npz for r={level}: {path}")
        spec[level] = np.load(path, allow_pickle=True)

    K_hat = load_ntk_spectra(args.ntk_path, levels)
    curves, spectral_shape = load_full_spectral_curves(spec, levels)
    if int(args.eval_grid) != int(spectral_shape[0]):
        raise ValueError(
            f"--eval_grid={args.eval_grid} but spectral_data uses "
            f"H={spectral_shape[0]}")

    F_raw = {
        level: estimate_floor(curves[level], args.tail_frac)
        for level in levels
    }
    F = enforce_monotone(F_raw, levels)
    P0 = curves[levels[0]][0]

    print("Building full fine-grid mode map...")
    _, _, K1, K2 = rfft_mode_bins(args.eval_grid, args.eval_grid // 2 + 1)
    weights = rfft_feature_weights(args.eval_grid)

    p_emp = (
        empirical_rates_for_features(spec, curves, F, levels)
        if args.only in {"all", "empirical"} else None
    )
    p_same = (
        ntk_rates_same_mode_full(
            K_hat, K1, K2, levels, args.lr, args.subset, batch_sizes)
        if args.only in {"all", "same_mode_ntk"} else None
    )
    p_alias = (
        ntk_rates_alias_lifted_full(
            K_hat, K1, K2, levels, args.lr, args.subset, batch_sizes)
        if args.only in {"all", "ntk_optimal"} else None
    )

    outputs = []
    if args.only in {"all", "empirical"}:
        outputs.append((
            "empirical",
            solve_for_budgets(
                "empirical rates", p_emp, F, P0, weights, costs, budgets, levels),
        ))
    if args.only in {"all", "same_mode_ntk"}:
        outputs.append((
            "same_mode_ntk",
            solve_for_budgets(
                "same-mode NTK", p_same, F, P0, weights, costs, budgets, levels),
        ))
    if args.only in {"all", "ntk_optimal"}:
        outputs.append((
            "ntk_optimal",
            solve_for_budgets(
                "NTK-optimal alias-lifted", p_alias, F, P0, weights, costs, budgets, levels),
        ))

    for name, rows in outputs:
        path = args.out_dir / f"{name}.csv"
        write_rows(path, rows)
        print(f"saved {path}")

    if not args.no_plot:
        n_plot_bins = min(args.eval_grid // 2 + 1, args.max_plot_bin + 1)
        p_emp_plot = radialise_rates_for_plot(
            p_emp, spectral_shape, n_plot_bins, levels)
        p_same_plot = radialise_rates_for_plot(
            p_same, spectral_shape, n_plot_bins, levels)
        p_alias_plot = radialise_rates_for_plot(
            p_alias, spectral_shape, n_plot_bins, levels)
        diagnostic_alias_plot(
            args.out_dir, p_emp_plot, p_same_plot, p_alias_plot, n_plot_bins, levels,
            args.max_plot_bin)


if __name__ == "__main__":
    main()
