#!/usr/bin/env python
"""
run_experiments.py — Orchestration for Method I & II QML option pricing experiments.

Method I  — supervised CDF training, PDF-Fourier pricing
Method II — semi-supervised CDF training, IBP pricing

Usage:
  python src/run_experiments.py                        # all, CPU
  python src/run_experiments.py --device cuda          # external GPU
  python src/run_experiments.py --methods 1            # only Method I
  python src/run_experiments.py --dry_run              # list experiments, no training
  python src/run_experiments.py --n_reps 1             # quick smoke test
"""

from __future__ import annotations

import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import pathlib
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

import torch

from qml4var.architectures import hardware_efficient_ansatz, init_weights
from qml4var.data_utils import (
    bs_cdf,
    empirical_cdf,
    generate_method_I_data,
    inverse_rescaling_u_to_xt,
    simulate_black_scholes_data_rescaled,
)
from qml4var.losses import method_I_h1_loss, torch_gradient, qdml_loss_workflow
from qml4var.workflows import mse_workflow, workflow_for_pdf_direct
from qml4var.adam import adam_optimizer_loop
from finance import bs_put_price, estimate_price_from_trained_pqc, estimate_price_ibp

# ── Black-Scholes parameters ──────────────────────────────────────────────────
BS_S0 = 100.0
BS_R = 0.1
BS_T = 1.0
BS_SIGMA = 0.25
STRIKES = [90.0, 100.0, 110.0]
TRAIN_INTERVAL = (-np.pi, np.pi)

# ── Shared hyperparameters ────────────────────────────────────────────────────
EPOCHS = 300
N_REPS = 10
BETA1 = 0.9
BETA2 = 0.999
PRINT_STEP = 50
INTEGRATION_POINTS = 100  # grid size for ∫PDF² in the QDML loss
K_TERMS = 12  # Fourier harmonics for option pricing

# ── Method configurations ─────────────────────────────────────────────────────
METHOD_CONFIGS: dict[int, dict] = {
    1: {
        "name": "method_I",
        "datasets": [250, 500, 1000, 2500],
        "architectures": [(6, 6), (7, 7), (8, 8)],  # (n_qubits, n_layers)
        "lr": 0.005,
        "alpha_0": 0.9,  # H¹ PDF-value weight (paper Table 1)
        "alpha_1": 0.1,  # H¹ PDF-derivative weight (paper Table 1)
        "n_test": 100,
        "pricing": "pdf_fourier",
        "use_real_method_I": True,  # train on analytical PDF labels (paper Sec. 3.2.1)
        # base_frecuency=1.0 → circuit Fourier period 2π, domain [-π, π].
        # For the PDF (Method I) this works well: the lognormal PDF is smooth and
        # decays to near-zero at the boundaries of [-π, π], so Gibbs oscillations
        # are negligible (unlike the CDF in Method II which has a step-function edge).
        # Paper Figs 2-3 show the difference is minor for the PDF case.
        "base_frecuency": 1.0,
        "eval_interval": (-np.pi, np.pi),  # same as training domain (no extended evaluation needed)
    },
    2: {
        "name": "method_II",
        "datasets": [1000, 2500, 5000, 10000],
        "architectures": [(4, 4), (5, 5), (6, 6)],
        "lr": 0.1,
        "alpha_0": 0.2,
        "alpha_1": 0.8,
        "n_test": 1000,
        "pricing": "ibp",
        # base_frecuency=0.5: half-frequency for CDF training — smoother IBP (paper Fig. 3, Sec. 3.2).
        # The encoding becomes RX(0.5*x + π/4), covering [-π/4, 3π/4] instead of [-π/2, π/2].
        "base_frecuency": 0.5,
        # shift_feature=π/4 breaks the parity symmetry of the Z⊗...⊗Z observable.
        # Without this, RX(0.5*x) + Z⊗Z observable gives f(-x) = f(x), so the circuit
        # cannot represent an asymmetric CDF and loss_boundary becomes self-contradictory
        # (F(a)=0 and F(b)=1 can't both be satisfied if the output is even).
        "shift_feature": np.pi / 4.0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_dataset(K: float, n_data: int, n_test: int, seed: int):
    """
    Generate training (u-space, empirical CDF) and test (grid, analytical CDF) sets.

    The circuit operates on log-moneyness x = log(S_T/K) rescaled to [-pi, pi].

    Returns
    -------
    train_x  : (n_data, 1) in [-pi, pi]
    train_y  : (n_data, 1) empirical CDF - 0.5
    test_x   : (n_test,  1) regular grid in [-pi, pi]
    test_y   : (n_test,  1) analytical BS CDF - 0.5
    x_min_raw, x_max_raw : bounds for inverse rescaling (used in pricing)
    """
    _, u_t, x_min_raw, x_max_raw = simulate_black_scholes_data_rescaled(
        S0_=BS_S0,
        r_=BS_R,
        T_=BS_T,
        sigma_=BS_SIGMA,
        K_=K,
        n_points=n_data,
        seed=seed,
    )
    train_x = u_t  # (n_data, 1)
    train_y = empirical_cdf(u_t).reshape(-1, 1)  # (n_data, 1) in [0, 1]

    u_test = np.linspace(-np.pi, np.pi, n_test).reshape(-1, 1)
    x_raw_test = inverse_rescaling_u_to_xt(u_test.reshape(-1), x_min_raw, x_max_raw)
    # P(X ≤ x) = P(S_T ≤ K·e^x) where X = log(S_T/K)
    s_t_test = K * np.exp(x_raw_test)
    test_y = bs_cdf(
        s_t_test,
        s_0=BS_S0,
        risk_free_rate=BS_R,
        volatility=BS_SIGMA,
        maturity=BS_T,
    ).reshape(-1, 1)  # in [0, 1]

    return train_x, train_y, u_test, test_y, x_min_raw, x_max_raw


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment
# ─────────────────────────────────────────────────────────────────────────────


def run_single(
    *,
    method_id: int,
    K: float,
    n_qubits: int,
    n_layers: int,
    n_data: int,
    rep: int,
    device: str,
    results_dir: pathlib.Path,
) -> dict:
    """Train one configuration and return a result dict."""
    cfg = METHOD_CONFIGS[method_id]
    use_real = cfg.get("use_real_method_I", False)

    # Deterministic seed: spread across K, data size, and rep
    seed = rep * 7919 + int(K) * 97 + n_data

    # ── 1. Data ───────────────────────────────────────────────────────────────
    if use_real:
        # Bifurcation 1: analytical grid + PDF labels (paper Sec. 3.2.1)
        train_x, pdf_labels, pdf_deriv_labels, x_min_raw, x_max_raw = generate_method_I_data(
            S0=BS_S0, K=K, r=BS_R, T=BS_T, sigma=BS_SIGMA, n_points=n_data
        )
        train_y = pdf_labels.reshape(-1, 1)
        test_x, test_pdf_labels, _, _, _ = generate_method_I_data(
            S0=BS_S0, K=K, r=BS_R, T=BS_T, sigma=BS_SIGMA, n_points=cfg["n_test"]
        )
        test_y = test_pdf_labels.reshape(-1, 1)
    else:
        train_x, train_y, test_x, test_y, x_min_raw, x_max_raw = _build_dataset(
            K=K,
            n_data=n_data,
            n_test=cfg["n_test"],
            seed=seed,
        )

    # ── 2. Circuit ────────────────────────────────────────────────────────────
    # Seed global numpy RNG for weight initialisation (data seed already used above)
    np.random.seed(seed + 42)
    circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
        features_number=1,
        n_qubits_by_feature=n_qubits,
        n_layers=n_layers,
        base_frecuency=[cfg.get("base_frecuency", 1.0)],
        shift_feature=[cfg.get("shift_feature", 0.0)],
        torch_device=device,
    )
    weights_dict = init_weights(weights_names)

    # ── 3. Workflow config ────────────────────────────────────────────────────
    workflow_cfg: dict = dict(
        circuit_fn=circuit_fn,
        torch_device=device,
        loss_weights=[cfg["alpha_0"], cfg["alpha_1"]],
        minval=[-np.pi],
        maxval=[np.pi],
        points=INTEGRATION_POINTS,
        features_names=features_names,
    )

    # ── 4. Closures for adam_optimizer_loop ───────────────────────────────────
    if use_real:
        # Bifurcation 2: H¹ supervised loss on PDF (paper eq. 12, Sec. 3.2.1)
        _circuit_fn = workflow_cfg["circuit_fn"]
        _device = workflow_cfg.get("torch_device", "cpu")
        _alpha_0 = cfg["alpha_0"]
        _alpha_1 = cfg["alpha_1"]

        def loss_fn(w):
            w_t = torch.tensor(
                list(w.values()) if isinstance(w, dict) else list(w),
                dtype=torch.float64,
            )
            return method_I_h1_loss(
                w_t, train_x, pdf_labels, pdf_deriv_labels,
                circuit_fn=_circuit_fn, device=_device,
                alpha_0=_alpha_0, alpha_1=_alpha_1, create_graph=False,
            ).item()

        def metric_fn(w):
            preds = workflow_for_pdf_direct(w, test_x, **workflow_cfg)["y_predict_pdf"]
            return float(np.mean((preds - test_pdf_labels) ** 2))

        def gradient_fn(w, bx, by):
            # bx = train_x batch, by = pdf_labels batch; pdf_deriv_labels captured
            def _loss_torch(w_t, bx_, by_):
                return method_I_h1_loss(
                    w_t, bx_, by_, pdf_deriv_labels,
                    circuit_fn=_circuit_fn, device=_device,
                    alpha_0=_alpha_0, alpha_1=_alpha_1, create_graph=True,
                )
            return torch_gradient(list(w), bx, by, _loss_torch)
    else:
        def loss_fn(w):
            return qdml_loss_workflow(w, train_x, train_y, **workflow_cfg)

        def metric_fn(w):
            return mse_workflow(w, test_x, test_y, **workflow_cfg)

        def gradient_fn(w, bx, by):
            def _loss_torch(w_t, bx_, by_):
                return qdml_loss_workflow(w_t, bx_, by_, **workflow_cfg)
            return torch_gradient(list(w), bx, by, _loss_torch)

    # ── 5. Output folder ──────────────────────────────────────────────────────
    run_name = f"M{method_id}_K{int(K)}_arch{n_qubits}x{n_layers}_data{n_data}_rep{rep:02d}"
    run_folder = results_dir / cfg["name"] / run_name
    run_folder.mkdir(parents=True, exist_ok=True)

    (run_folder / "config.json").write_text(
        json.dumps(
            dict(
                method=method_id,
                K=K,
                n_qubits=n_qubits,
                n_layers=n_layers,
                n_data=n_data,
                rep=rep,
                seed=seed,
                lr=cfg["lr"],
                alpha_0=cfg["alpha_0"],
                alpha_1=cfg["alpha_1"],
                epochs=EPOCHS,
                device=device,
                S0=BS_S0,
                r=BS_R,
                T=BS_T,
                sigma=BS_SIGMA,
            ),
            indent=2,
        )
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    final_weights = adam_optimizer_loop(
        weights_dict=weights_dict,
        loss_function=loss_fn,
        metric_function=metric_fn,
        gradient_function=gradient_fn,
        batch_generator=[(train_x, train_y)],  # full batch, list → re-iterable
        epochs=EPOCHS,
        learning_rate=cfg["lr"],
        beta1=BETA1,
        beta2=BETA2,
        tolerance=-1e30,  # never trigger early stopping
        n_counts_tolerance=EPOCHS + 1,
        print_step=PRINT_STEP,
        file_to_save=str(run_folder / "evolution.csv"),
        progress_bar=True,
        progress_desc=run_name,
        progress_leave=False,
    )

    # ── 7. Save final weights ─────────────────────────────────────────────────
    np.save(run_folder / "final_weights.npy", np.array(final_weights))

    # ── 8. Option pricing ─────────────────────────────────────────────────────
    artifacts = {"workflow_cfg": workflow_cfg}
    _base_kwargs = dict(
        weights=final_weights,
        artifacts=artifacts,
        K_=K,
        x_min_raw=x_min_raw,
        x_max_raw=x_max_raw,
        train_interval=TRAIN_INTERVAL,
        risk_free_rate=BS_R,
        delta_t=BS_T,
        k_terms=K_TERMS,
    )
    try:
        if cfg["pricing"] == "pdf_fourier":
            # Method I: eval_interval controls the circuit domain used for Fourier extraction.
            # With base_frecuency=0.5 the full circuit domain is [-2π, 2π].
            pricing_eval_interval = cfg.get("eval_interval", TRAIN_INTERVAL)
            est_price = float(estimate_price_from_trained_pqc(
                **_base_kwargs, eval_interval=pricing_eval_interval
            ))
        else:
            # Method II: estimate_price_ibp handles its own domain extension internally.
            est_price = float(estimate_price_ibp(**_base_kwargs))
    except Exception as exc:
        print(f"  [pricing error] {exc}")
        est_price = float("nan")

    bs_price = bs_put_price(S0_=BS_S0, K_=K, r_=BS_R, sigma_=BS_SIGMA, T_=BS_T)
    final_mse = metric_fn(final_weights)
    abs_err = abs(est_price - bs_price) if np.isfinite(est_price) else float("nan")
    rel_err = abs_err / abs(bs_price) if np.isfinite(abs_err) and bs_price != 0 else float("nan")

    result = dict(
        method=method_id,
        K=K,
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_data=n_data,
        rep=rep,
        seed=seed,
        bs_price=bs_price,
        estimated_price=est_price,
        abs_error=abs_err,
        rel_error=rel_err,
        final_mse=final_mse,
    )
    pd.DataFrame([result]).to_csv(run_folder / "result.csv", sep=";", index=False)

    print(
        f"  → BS={bs_price:.4f}  Est={est_price:.4f}  AbsErr={abs_err:.4f}  MSE={final_mse:.2e}",
        flush=True,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="QML option pricing experiments — Methods I and II",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device: 'cpu' or 'cuda'  (default: cpu)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=int,
        choices=[1, 2],
        default=[1, 2],
        help="Methods to run (default: 1 2)",
    )
    parser.add_argument(
        "--strikes",
        nargs="+",
        type=float,
        default=STRIKES,
        help="Strike prices (default: 90 100 110)",
    )
    parser.add_argument(
        "--results_dir",
        default="results/experiments",
        help="Root output directory (default: results/experiments)",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=N_REPS,
        help=f"Repetitions per configuration (default: {N_REPS})",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Filter architectures to run, format NxL e.g. '7x7 5x5'. "
            "If omitted all architectures in the method config are used."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Filter dataset sizes to run, e.g. '2500 5000'. "
            "If omitted all dataset sizes in the method config are used."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the experiment list without training",
    )
    args = parser.parse_args()

    # Parse --architectures filter into a set of (n_qubits, n_layers) tuples
    arch_filter: set | None = None
    if args.architectures is not None:
        arch_filter = set()
        for spec in args.architectures:
            parts = spec.lower().split("x")
            if len(parts) != 2:
                raise ValueError(f"Invalid architecture spec '{spec}', expected NxL (e.g. 7x7)")
            arch_filter.add((int(parts[0]), int(parts[1])))

    dataset_filter: set | None = None if args.datasets is None else set(args.datasets)

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build experiment grid
    experiments: list[dict] = []
    for method_id in sorted(set(args.methods)):
        cfg = METHOD_CONFIGS[method_id]
        for K in args.strikes:
            for n_qubits, n_layers in cfg["architectures"]:
                if arch_filter is not None and (n_qubits, n_layers) not in arch_filter:
                    continue
                for n_data in cfg["datasets"]:
                    if dataset_filter is not None and n_data not in dataset_filter:
                        continue
                    for rep in range(args.n_reps):
                        experiments.append(
                            dict(
                                method_id=method_id,
                                K=K,
                                n_qubits=n_qubits,
                                n_layers=n_layers,
                                n_data=n_data,
                                rep=rep,
                            )
                        )

    n_total = len(experiments)
    print(f"Total experiments : {n_total}")
    print(f"Device            : {args.device}")
    print(f"Results dir       : {results_dir.resolve()}")

    if args.dry_run:
        for i, e in enumerate(experiments):
            print(
                f"  [{i + 1:4d}/{n_total}] "
                f"M{e['method_id']} K={e['K']} "
                f"arch={e['n_qubits']}x{e['n_layers']} "
                f"data={e['n_data']} rep={e['rep']}"
            )
        return

    master_csv = results_dir / "master_results.csv"
    all_results: list[dict] = []
    t_start = datetime.now()

    for i, exp in enumerate(experiments):
        print(
            f"\n[{i + 1}/{n_total}] "
            f"M{exp['method_id']} | K={exp['K']} | "
            f"arch={exp['n_qubits']}x{exp['n_layers']} | "
            f"data={exp['n_data']} | rep={exp['rep']}",
            flush=True,
        )
        try:
            result = run_single(**exp, device=args.device, results_dir=results_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            traceback.print_exc()
            result = {**exp, "method": exp["method_id"], "error": str(exc)}

        all_results.append(result)
        # Incremental save after every run
        pd.DataFrame(all_results).to_csv(master_csv, sep=";", index=False)

    elapsed = (datetime.now() - t_start).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"Completed {n_total} experiments in {elapsed / 3600:.2f} h")
    print(f"Master results: {master_csv.resolve()}")


if __name__ == "__main__":
    main()
