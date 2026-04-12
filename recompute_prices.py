#!/usr/bin/env python
"""
recompute_prices.py — Re-run option pricing on saved weights using the fixed formula.

Use this after applying the B_k sign fix in finance.py (ak_bk_from_complex_coefficients)
to update result.csv files without re-running training.

Usage:
    python recompute_prices.py --results_dir results/experiments
    python recompute_prices.py --results_dir results/verification
    python recompute_prices.py --results_dir results/experiments --method 2  # M2 only
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from finance import bs_put_price, estimate_price_from_trained_pqc, estimate_price_ibp  # noqa: E402
from qml4var.architectures import hardware_efficient_ansatz  # noqa: E402

# ── Shared constants (must match run_experiments.py) ─────────────────────────
BS_S0, BS_R, BS_T, BS_SIGMA = 100.0, 0.1, 1.0, 0.25
TRAIN_INTERVAL = (-np.pi, np.pi)
K_TERMS = 12

METHOD_PRICING = {1: "pdf_fourier", 2: "ibp"}
METHOD_EVAL_INTERVAL = {1: (-np.pi, np.pi), 2: None}

# base_frecuency and shift_feature per method (must match run_experiments.py METHOD_CONFIGS)
METHOD_BASE_FREQ = {1: 1.0, 2: 0.5}
METHOD_SHIFT = {1: 0.0, 2: -np.pi / 2.0}


def recompute_one(run_folder: pathlib.Path, device: str) -> dict | None:
    """Load weights from run_folder and recompute option price."""
    config_path = run_folder / "config.json"
    weights_path = run_folder / "final_weights.npy"

    if not config_path.exists() or not weights_path.exists():
        return None

    with open(config_path) as f:
        cfg = json.load(f)

    method = int(cfg["method"])
    K = float(cfg["K"])
    n_qubits = int(cfg["n_qubits"])
    n_layers = int(cfg["n_layers"])
    n_data = int(cfg["n_data"])
    rep = int(cfg["rep"])

    weights = np.load(weights_path).tolist()

    base_freq = METHOD_BASE_FREQ.get(method, 1.0)
    shift_feat = METHOD_SHIFT.get(method, 0.0)

    circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
        features_number=1,
        n_qubits_by_feature=n_qubits,
        n_layers=n_layers,
        base_frecuency=[base_freq],
        shift_feature=[shift_feat],
        torch_device=device,
    )

    # Rebuild x_min_raw / x_max_raw from analytical BS bounds
    from qml4var.data_utils import simulate_black_scholes_data_rescaled
    seed = rep * 7919 + int(K) * 97 + n_data
    _, _, x_min_raw, x_max_raw = simulate_black_scholes_data_rescaled(
        S0_=BS_S0, r_=BS_R, T_=BS_T, sigma_=BS_SIGMA, K_=K,
        n_points=n_data, seed=seed,
    )

    workflow_cfg = dict(
        circuit_fn=circuit_fn,
        torch_device=device,
        loss_weights=[0.2, 0.8],
        minval=[-np.pi],
        maxval=[np.pi],
        points=100,
        features_names=features_names,
    )
    artifacts = {"workflow_cfg": workflow_cfg}
    base_kwargs = dict(
        weights=weights,
        artifacts=artifacts,
        K_=K,
        x_min_raw=x_min_raw,
        x_max_raw=x_max_raw,
        train_interval=TRAIN_INTERVAL,
        risk_free_rate=BS_R,
        delta_t=BS_T,
        k_terms=K_TERMS,
    )

    pricing = METHOD_PRICING.get(method, "ibp")
    try:
        if pricing == "pdf_fourier":
            eval_interval = METHOD_EVAL_INTERVAL.get(method, TRAIN_INTERVAL)
            est_price = float(estimate_price_from_trained_pqc(**base_kwargs, eval_interval=eval_interval))
        else:
            est_price = float(estimate_price_ibp(**base_kwargs))
    except Exception as exc:
        print(f"  [pricing error] {run_folder.name}: {exc}")
        est_price = float("nan")

    bs_price = bs_put_price(S0_=BS_S0, K_=K, r_=BS_R, sigma_=BS_SIGMA, T_=BS_T)
    abs_err = abs(est_price - bs_price) if np.isfinite(est_price) else float("nan")
    rel_err = abs_err / abs(bs_price) if np.isfinite(abs_err) and bs_price != 0 else float("nan")

    result = dict(
        method=method, K=K, n_qubits=n_qubits, n_layers=n_layers,
        n_data=n_data, rep=rep, seed=seed,
        bs_price=bs_price, estimated_price=est_price,
        abs_error=abs_err, rel_error=rel_err,
    )

    # Overwrite result.csv in the run folder
    pd.DataFrame([result]).to_csv(run_folder / "result.csv", sep=";", index=False)
    print(f"  {run_folder.name}: BS={bs_price:.4f} Est={est_price:.4f} AbsErr={abs_err:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Recompute option prices from saved weights")
    parser.add_argument("--results_dir", default="results/experiments")
    parser.add_argument("--method", type=int, choices=[1, 2], default=None,
                        help="Restrict to method 1 or 2 (default: both)")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device (default: cpu)")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Find all run folders (contain config.json + final_weights.npy)
    run_folders = sorted(results_dir.rglob("config.json"))
    print(f"Found {len(run_folders)} completed runs in {results_dir}")

    all_results = []
    for config_path in run_folders:
        run_folder = config_path.parent

        # Filter by method if requested
        if args.method is not None:
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                if int(cfg.get("method", 0)) != args.method:
                    continue
            except Exception:
                continue

        result = recompute_one(run_folder, args.device)
        if result is not None:
            all_results.append(result)

    if all_results:
        master_csv = results_dir / "master_results_recomputed.csv"
        pd.DataFrame(all_results).to_csv(master_csv, sep=";", index=False)
        print(f"\nRecomputed {len(all_results)} prices → {master_csv}")

        # Quick summary
        df = pd.DataFrame(all_results)
        for m in df["method"].unique():
            for k in sorted(df["K"].unique()):
                sub = df[(df["method"] == m) & (df["K"] == k)]
                print(f"  M{int(m)} K={k:.0f}: mean AbsErr = {sub['abs_error'].mean():.4f}")


if __name__ == "__main__":
    main()
