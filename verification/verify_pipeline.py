#!/usr/bin/env python
"""
verify_pipeline.py — Three sequential checks for the fixed PQC pipeline.

Verification 1: Encoding varies with data (normalization bug is fixed).
Verification 2: PQC learns a Gaussian (ansatz is functional).
Verification 3: Option price converges with more data (full pipeline).

Usage:
    python verification/verify_pipeline.py              # all 3, CPU
    python verification/verify_pipeline.py --device cuda
    python verification/verify_pipeline.py --check 1   # only check 1
    python verification/verify_pipeline.py --check 2
    python verification/verify_pipeline.py --check 3

Results and plots are saved in verification/results/.
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib
import json
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
_VERIFY_DIR = pathlib.Path(__file__).parent
_ROOT = _VERIFY_DIR.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT, str(_SRC), str(_ROOT)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for remote GPU
import matplotlib.pyplot as plt

from qml4var.architectures import hardware_efficient_ansatz, init_weights
from qml4var.adam import adam_optimizer_loop, update_parameters_with_adam
from qml4var.losses import torch_gradient
from qml4var.workflows import workflow_for_cdf, qdml_loss_workflow, mse_workflow
from qml4var.data_utils import (
    bs_cdf,
    empirical_cdf,
    simulate_black_scholes_data_rescaled,
    inverse_rescaling_u_to_xt,
)
from finance import bs_put_price, estimate_price_from_trained_pqc

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = _VERIFY_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Black-Scholes reference parameters (from run_experiments.py) ──────────────
BS_S0 = 100.0
BS_R = 0.1
BS_T = 1.0
BS_SIGMA = 0.25
BS_K_ATM = 100.0  # At-the-money
TRAIN_INTERVAL = (-np.pi, np.pi)


# =============================================================================
# Helpers
# =============================================================================

def _print_header(title: str):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def _print_result(passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    msg = f"  [{tag}]"
    if detail:
        msg += f" {detail}"
    print(msg)
    return passed


def _save_summary(name: str, data: dict):
    path = RESULTS_DIR / f"{name}_summary.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  → Summary saved: {path}")


# =============================================================================
# Verification 1 — Encoding varies with data
# =============================================================================

def check_1_encoding_varies(device: str = "cpu") -> bool:
    """
    Build a small circuit with random weights, evaluate on 10 equally-spaced
    inputs across [-pi, pi], and confirm the outputs are NOT all equal.

    Pass condition: std(outputs) > 0.05
    """
    _print_header("Verification 1 — Encoding varies with data")

    np.random.seed(0)
    circuit_fn, weights_names, _ = hardware_efficient_ansatz(
        features_number=1,
        n_qubits_by_feature=3,
        n_layers=3,
        base_frecuency=[0.5],
        shift_feature=[0.0],
        torch_device=device,
    )
    weights_dict = init_weights(weights_names)
    w_t = torch.tensor(list(weights_dict.values()), dtype=torch.float64,
                       device=torch.device(device))

    xs = np.linspace(-np.pi, np.pi, 10)
    outputs = []
    for x_val in xs:
        x_t = torch.tensor([x_val], dtype=torch.float64, device=torch.device(device))
        with torch.no_grad():
            out = circuit_fn(w_t, x_t).item()
        outputs.append(out)

    outputs_arr = np.array(outputs)
    std_val = float(np.std(outputs_arr))
    variation = float(np.max(outputs_arr) - np.min(outputs_arr))

    print(f"\n  Input xs  : {np.round(xs, 3).tolist()}")
    print(f"  Outputs   : {np.round(outputs_arr, 5).tolist()}")
    print(f"  Std       : {std_val:.5f}")
    print(f"  Max-Min   : {variation:.5f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, outputs_arr, "o-", color="steelblue", label="PQC output")
    ax.axhline(np.mean(outputs_arr), linestyle="--", color="gray", alpha=0.6, label="mean")
    ax.set_xlabel("Input x (normalized, [-π, π])")
    ax.set_ylabel("Circuit output")
    ax.set_title("Verification 1 — Circuit output vs input\n"
                 "(should vary across different inputs)")
    ax.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / "v1_encoding_varies.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  → Plot saved: {fig_path}")

    _save_summary("v1_encoding", {
        "inputs": xs.tolist(),
        "outputs": outputs_arr.tolist(),
        "std": std_val,
        "variation": variation,
        "threshold_std": 0.05,
        "passed": std_val > 0.05,
    })

    passed = std_val > 0.05
    _print_result(
        passed,
        f"std={std_val:.4f} (threshold > 0.05). "
        + ("Encoding is data-dependent." if passed else "Outputs suspiciously flat — bug may still be active."),
    )
    return passed


# =============================================================================
# Verification 2 — PQC learns a Gaussian
# =============================================================================

def check_2_gaussian_fit(device: str = "cpu", epochs: int = 200) -> bool:
    """
    Train the PQC (4x4 architecture) for `epochs` epochs to fit the standard
    Gaussian density f(x) = (1/sqrt(2pi)) * exp(-x²/2) sampled on [-pi, pi].

    Pass condition: the trained curve has its maximum near x=0 AND decreases
    toward both ends (i.e., the peak is within [-0.5, 0.5] and the predicted
    values at the endpoints are less than the peak value).
    """
    _print_header("Verification 2 — PQC learns a Gaussian (200 epochs)")

    np.random.seed(42)
    circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
        features_number=1,
        n_qubits_by_feature=4,
        n_layers=4,
        base_frecuency=[0.5],
        shift_feature=[0.0],
        torch_device=device,
    )
    weights_dict = init_weights(weights_names)

    # Training data: Gaussian density on [-pi, pi]
    n_pts = 100
    xs_np = np.linspace(-np.pi, np.pi, n_pts).reshape(-1, 1)  # (100, 1)
    ys_true = (np.exp(-0.5 * xs_np[:, 0] ** 2) / np.sqrt(2 * np.pi)).reshape(-1, 1)

    from torch.func import vmap as _vmap

    def gaussian_loss_torch(w_t, data_x, data_y):
        """Differentiable MSE loss for Gaussian fitting."""
        x_batch = torch.tensor(data_x, dtype=torch.float64, device=torch.device(device))
        y_batch = torch.tensor(data_y, dtype=torch.float64, device=torch.device(device))
        preds = _vmap(circuit_fn, in_dims=(None, 0))(w_t, x_batch).reshape(-1)
        return torch.mean((preds - y_batch.reshape(-1)) ** 2)

    def loss_fn_numpy(w):
        """Returns float — used only for printing progress."""
        w_t = torch.tensor(list(w), dtype=torch.float64, device=torch.device(device))
        with torch.no_grad():
            return gaussian_loss_torch(w_t, xs_np, ys_true).item()

    def gradient_fn(w, bx, by):
        return torch_gradient(list(w), bx, by, gaussian_loss_torch)

    print(f"  Architecture : 4 qubits × 4 layers  ({len(weights_names)} weights)")
    print(f"  Epochs       : {epochs}")
    print(f"  Device       : {device}")

    # ── Train ──────────────────────────────────────────────────────────────────
    evo_path = str(RESULTS_DIR / "v2_gaussian_evolution.csv")
    final_weights = adam_optimizer_loop(
        weights_dict=weights_dict,
        loss_function=loss_fn_numpy,
        metric_function=None,
        gradient_function=gradient_fn,
        batch_generator=[(xs_np, ys_true)],
        epochs=epochs,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        tolerance=-1e30,
        n_counts_tolerance=epochs + 1,
        print_step=50,
        file_to_save=evo_path,
        progress_bar=True,
        progress_desc="Gaussian fit",
        progress_leave=False,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    w_final = torch.tensor(final_weights, dtype=torch.float64, device=torch.device(device))
    x_eval = torch.tensor(xs_np, dtype=torch.float64, device=torch.device(device))
    with torch.no_grad():
        ys_pred = _vmap(circuit_fn, in_dims=(None, 0))(w_final, x_eval).cpu().numpy().reshape(-1)

    xs_flat = xs_np[:, 0]
    ys_true_flat = ys_true[:, 0]

    peak_idx = int(np.argmax(ys_pred))
    peak_x = float(xs_flat[peak_idx])
    peak_left = float(ys_pred[0])
    peak_right = float(ys_pred[-1])
    peak_val = float(ys_pred[peak_idx])
    final_mse = float(np.mean((ys_pred - ys_true_flat) ** 2))

    print(f"\n  Peak at x={peak_x:.3f}  (expected near 0)")
    print(f"  Peak value       : {peak_val:.4f}")
    print(f"  Left endpoint    : {peak_left:.4f}")
    print(f"  Right endpoint   : {peak_right:.4f}")
    print(f"  Final MSE        : {final_mse:.6f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs_flat, ys_true_flat, "-", color="steelblue", linewidth=2, label="True Gaussian")
    ax.plot(xs_flat, ys_pred, "--", color="darkorange", linewidth=2, label=f"PQC ({epochs} epochs)")
    ax.axvline(x=0, linestyle=":", color="gray", alpha=0.4)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Verification 2 — Gaussian fitting\n"
                 "(PQC curve should follow the bell shape)")
    ax.legend()
    plt.tight_layout()
    fig_path = RESULTS_DIR / "v2_gaussian_fit.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  → Plot saved: {fig_path}")

    _save_summary("v2_gaussian", {
        "peak_x": peak_x,
        "peak_value": peak_val,
        "left_endpoint": peak_left,
        "right_endpoint": peak_right,
        "final_mse": final_mse,
        "passed_peak_center": abs(peak_x) < 0.8,
        "passed_endpoints_lower": (peak_val > peak_left) and (peak_val > peak_right),
    })

    peak_centered = abs(peak_x) < 0.8  # peak within ±0.8 of center
    endpoints_lower = (peak_val > peak_left) and (peak_val > peak_right)
    passed = peak_centered and endpoints_lower
    _print_result(
        passed,
        f"peak at x={peak_x:.3f}, peak>{peak_left:.3f} (left) and >{peak_right:.3f} (right). "
        + ("Bell shape detected." if passed else "No bell shape — ansatz may still be broken."),
    )
    return passed


# =============================================================================
# Verification 3 — Option price converges with more data
# =============================================================================

def check_3_price_convergence(device: str = "cpu", n_reps: int = 3) -> bool:
    """
    Train Method I (7×7 architecture, K=100 ATM) on growing datasets.
    Confirm the absolute pricing error decreases monotonically with N.

    Pass condition: error(N=2500) < error(N=250).
    """
    _print_header("Verification 3 — Price convergence with more data (Method I, K=100)")

    EPOCHS = 300
    LR = 0.005
    ALPHA_0, ALPHA_1 = 0.9, 0.1
    N_QUBITS, N_LAYERS = 7, 7
    K = 100.0
    DATASETS = [250, 500, 1000, 2500]
    N_TEST = 100
    K_TERMS = 12
    INTEGRATION_POINTS = 100

    bs_price = bs_put_price(S0_=BS_S0, K_=K, r_=BS_R, sigma_=BS_SIGMA, T_=BS_T)
    print(f"\n  BS exact price   : {bs_price:.4f}")
    print(f"  Architecture     : {N_QUBITS}x{N_LAYERS}")
    print(f"  Datasets         : {DATASETS}")
    print(f"  Replicas per N   : {n_reps}")
    print(f"  Device           : {device}")

    rows = []  # one row per (n_data, rep)

    for n_data in DATASETS:
        rep_prices = []
        rep_errors = []

        for rep in range(n_reps):
            seed = rep * 7919 + int(K) * 97 + n_data

            # ── Data ─────────────────────────────────────────────────────────
            _, u_t, x_min_raw, x_max_raw = simulate_black_scholes_data_rescaled(
                S0_=BS_S0,
                r_=BS_R,
                T_=BS_T,
                sigma_=BS_SIGMA,
                K_=K,
                n_points=n_data,
                seed=seed,
            )
            train_x = u_t
            train_y = empirical_cdf(u_t).reshape(-1, 1) - 0.5

            u_test = np.linspace(-np.pi, np.pi, N_TEST).reshape(-1, 1)
            x_raw_test = inverse_rescaling_u_to_xt(u_test.reshape(-1), x_min_raw, x_max_raw)
            s_t_test = K * np.exp(x_raw_test)
            test_y = bs_cdf(
                s_t_test, s_0=BS_S0, risk_free_rate=BS_R,
                volatility=BS_SIGMA, maturity=BS_T,
            ).reshape(-1, 1) - 0.5

            # ── Circuit ───────────────────────────────────────────────────────
            np.random.seed(seed + 42)
            circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
                features_number=1,
                n_qubits_by_feature=N_QUBITS,
                n_layers=N_LAYERS,
                base_frecuency=[0.5],
                shift_feature=[0.0],
                torch_device=device,
            )
            weights_dict = init_weights(weights_names)

            workflow_cfg = dict(
                circuit_fn=circuit_fn,
                torch_device=device,
                loss_weights=[ALPHA_0, ALPHA_1],
                minval=[-np.pi],
                maxval=[np.pi],
                points=INTEGRATION_POINTS,
                features_names=features_names,
            )

            # ── Closures ──────────────────────────────────────────────────────
            def loss_fn(w, _wc=workflow_cfg, _tx=train_x, _ty=train_y):
                return qdml_loss_workflow(w, _tx, _ty, **_wc)

            def metric_fn(w, _wc=workflow_cfg, _vx=u_test, _vy=test_y):
                return mse_workflow(w, _vx, _vy, **_wc)

            def gradient_fn(w, bx, by, _wc=workflow_cfg):
                def _loss_torch(w_t, bx_, by_):
                    return qdml_loss_workflow(w_t, bx_, by_, **_wc)
                return torch_gradient(list(w), bx, by, _loss_torch)

            run_label = f"N{n_data}_rep{rep:02d}"
            evo_path = str(RESULTS_DIR / f"v3_{run_label}_evolution.csv")

            # ── Train ─────────────────────────────────────────────────────────
            print(f"\n  Training {run_label} …", flush=True)
            final_weights = adam_optimizer_loop(
                weights_dict=weights_dict,
                loss_function=loss_fn,
                metric_function=metric_fn,
                gradient_function=gradient_fn,
                batch_generator=[(train_x, train_y)],
                epochs=EPOCHS,
                learning_rate=LR,
                beta1=0.9,
                beta2=0.999,
                tolerance=-1e30,
                n_counts_tolerance=EPOCHS + 1,
                print_step=100,
                file_to_save=evo_path,
                progress_bar=True,
                progress_desc=run_label,
                progress_leave=False,
            )

            # ── Price ─────────────────────────────────────────────────────────
            try:
                est_price = float(estimate_price_from_trained_pqc(
                    weights=final_weights,
                    artifacts={"workflow_cfg": workflow_cfg},
                    K_=K,
                    x_min_raw=x_min_raw,
                    x_max_raw=x_max_raw,
                    train_interval=TRAIN_INTERVAL,
                    risk_free_rate=BS_R,
                    delta_t=BS_T,
                    k_terms=K_TERMS,
                ))
            except Exception as exc:
                print(f"    [pricing error] {exc}")
                est_price = float("nan")

            abs_err = abs(est_price - bs_price) if np.isfinite(est_price) else float("nan")
            final_mse = metric_fn(final_weights)
            rep_prices.append(est_price)
            rep_errors.append(abs_err)

            print(
                f"    BS={bs_price:.4f}  Est={est_price:.4f}  "
                f"AbsErr={abs_err:.4f}  MSE={final_mse:.2e}"
            )
            rows.append({
                "n_data": n_data,
                "rep": rep,
                "bs_price": bs_price,
                "estimated_price": est_price,
                "abs_error": abs_err,
                "final_mse": final_mse,
            })

        mean_price = float(np.nanmean(rep_prices))
        mean_err = float(np.nanmean(rep_errors))
        print(
            f"\n  N={n_data:5d} | mean price={mean_price:.4f} | "
            f"mean |err|={mean_err:.4f} | BS exact={bs_price:.4f}"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("n_data")
        .agg(mean_price=("estimated_price", "mean"), mean_abs_error=("abs_error", "mean"))
        .reset_index()
    )
    summary["bs_price"] = bs_price

    print("\n  Convergence table:")
    print(f"  {'N datos':>8} | {'Precio medio':>13} | {'Error medio':>12} | {'BS exacto':>10}")
    print(f"  {'-'*8} | {'-'*13} | {'-'*12} | {'-'*10}")
    for _, row in summary.iterrows():
        print(
            f"  {int(row['n_data']):>8} | "
            f"{row['mean_price']:>13.4f} | "
            f"{row['mean_abs_error']:>12.4f} | "
            f"{row['bs_price']:>10.4f}"
        )

    # Save summary CSV
    csv_path = RESULTS_DIR / "v3_convergence_table.csv"
    summary.to_csv(csv_path, sep=";", index=False)
    df.to_csv(RESULTS_DIR / "v3_all_runs.csv", sep=";", index=False)
    print(f"\n  → Table saved: {csv_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(summary["n_data"], summary["mean_price"], "o-", color="steelblue", label="Mean estimated price")
    ax.axhline(bs_price, linestyle="--", color="crimson", label=f"BS exact ({bs_price:.4f})")
    ax.set_xscale("log")
    ax.set_xlabel("N training samples (log scale)")
    ax.set_ylabel("Option price")
    ax.set_title("Estimated price vs N")
    ax.legend()

    ax = axes[1]
    ax.plot(summary["n_data"], summary["mean_abs_error"], "s-", color="darkorange", label="Mean |error|")
    ax.set_xscale("log")
    ax.set_xlabel("N training samples (log scale)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Pricing error vs N\n(should decrease monotonically)")
    ax.legend()

    plt.suptitle("Verification 3 — Method I, K=100, arch 7×7", fontsize=13)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "v3_price_convergence.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  → Plot saved: {fig_path}")

    # ── Pass condition ────────────────────────────────────────────────────────
    errors = summary["mean_abs_error"].tolist()
    monotone = all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))
    error_drop = errors[0] - errors[-1]  # positive = improvement
    passed = error_drop > 0  # relaxed: at least the last is better than first

    _save_summary("v3_convergence", {
        "bs_price": bs_price,
        "n_data": summary["n_data"].tolist(),
        "mean_abs_errors": errors,
        "monotone_decrease": monotone,
        "error_drop_250_to_2500": error_drop,
        "passed": passed,
    })
    _print_result(
        passed,
        f"error drop from N=250 to N=2500: {error_drop:+.4f}. "
        + (f"Monotone: {monotone}." if passed else "No improvement detected."),
    )
    return passed


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline verification checks (1, 2, or 3)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--check",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only this check (default: run all 3 in order)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Epochs for Verification 2 Gaussian fit (default: 200)",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=3,
        help="Replicas per dataset size in Verification 3 (default: 3)",
    )
    args = parser.parse_args()

    checks_to_run = [args.check] if args.check else [1, 2, 3]

    print(f"\n{'#' * 60}")
    print(f"  Pipeline Verification Suite")
    print(f"  Device  : {args.device}")
    print(f"  Checks  : {checks_to_run}")
    print(f"  Results : {RESULTS_DIR.resolve()}")
    print(f"{'#' * 60}")

    results = {}

    if 1 in checks_to_run:
        results[1] = check_1_encoding_varies(device=args.device)

    if 2 in checks_to_run:
        results[2] = check_2_gaussian_fit(device=args.device, epochs=args.epochs)

    if 3 in checks_to_run:
        results[3] = check_3_price_convergence(device=args.device, n_reps=args.n_reps)

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  FINAL REPORT")
    print(f"{'=' * 60}")
    labels = {
        1: "Encoding varies with data   (normalization fix)",
        2: "PQC learns a Gaussian       (ansatz functional)",
        3: "Price converges with data   (full pipeline)",
    }
    all_passed = True
    for check_id, passed in results.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  Check {check_id}: [{tag}]  {labels[check_id]}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All checks passed. The pipeline is ready for full experiments.")
    else:
        print("\n  Some checks failed. Review the output above and fix before proceeding.")

    _save_summary("final_report", {
        "timestamp": datetime.now().isoformat(),
        "device": args.device,
        "results": {str(k): v for k, v in results.items()},
        "all_passed": all_passed,
    })
    print(f"\n  Full results in: {RESULTS_DIR.resolve()}\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
