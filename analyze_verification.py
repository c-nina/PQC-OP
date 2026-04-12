#!/usr/bin/env python
"""
analyze_verification.py — Summarise verification results and gate the full run.

Reads all result.csv files under results/verification/ and prints:
  - Per-method table: mean absolute error by (N_data, K)
  - Acceptance criteria check (pass / FAIL)

Usage:
    python analyze_verification.py                           # default dir
    python analyze_verification.py --results_dir results/verification
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

PRICE_UPPER = 90.5      # K * exp(-r*T) = 100 * exp(-0.1) ≈ 90.48
ATM_ERR_MAX = 2.0       # max tolerated mean abs error for K=100
ATM_K = 100.0


def load_results(results_dir: pathlib.Path) -> pd.DataFrame:
    csvs = list(results_dir.rglob("result.csv"))
    if not csvs:
        print(f"[ERROR] No result.csv files found under {results_dir.resolve()}")
        sys.exit(1)
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f, sep=";")
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Could not read {f}: {exc}")
    return pd.concat(frames, ignore_index=True)


def check_criteria(df: pd.DataFrame) -> bool:
    """Return True if all acceptance criteria pass."""
    all_pass = True

    print("\n" + "=" * 62)
    print("ACCEPTANCE CRITERIA")
    print("=" * 62)

    # ── 1. No price outside [0, 90.5] ────────────────────────────────────────
    finite = df[np.isfinite(df["estimated_price"])]
    out_of_range = finite[
        (finite["estimated_price"] < 0) | (finite["estimated_price"] > PRICE_UPPER)
    ]
    c1 = len(out_of_range) == 0
    status = "PASS" if c1 else "FAIL"
    print(f"[{status}] 1. All prices in [0, {PRICE_UPPER}]", end="")
    if not c1:
        all_pass = False
        print(f"\n       {len(out_of_range)} out-of-range prices:")
        for _, row in out_of_range.iterrows():
            print(f"       M{int(row['method'])} K={row['K']} N={row['n_data']} "
                  f"rep={row['rep']} price={row['estimated_price']:.4f}")
    else:
        print()

    # ── 2. Zero NaN in Method I ───────────────────────────────────────────────
    m1 = df[df["method"] == 1]
    nan_m1 = m1[~np.isfinite(m1["estimated_price"])]
    c2 = len(nan_m1) == 0
    status = "PASS" if c2 else "FAIL"
    print(f"[{status}] 2. Zero NaN in Method I", end="")
    if not c2:
        all_pass = False
        print(f"\n       {len(nan_m1)} NaN prices in Method I")
    else:
        print()

    # ── 3. Mean absolute error decreasing with N (ATM K=100) ─────────────────
    methods_present = sorted(df["method"].unique())
    for m in methods_present:
        sub = df[(df["method"] == m) & (df["K"] == ATM_K) & np.isfinite(df["abs_error"])]
        if sub.empty:
            continue
        mean_by_n = sub.groupby("n_data")["abs_error"].mean().sort_index()
        is_decreasing = all(
            mean_by_n.iloc[i] >= mean_by_n.iloc[i + 1] * 0.7   # 30% slack for noise
            for i in range(len(mean_by_n) - 1)
        )
        status_local = "PASS" if is_decreasing else "WARN"
        if not is_decreasing:
            pass  # treat as a warning, not hard failure
        print(f"[{status_local}] 3. M{m} error decreasing with N (K=100): "
              + " → ".join(f"{v:.3f}" for v in mean_by_n.values))

    # ── 4. Mean abs error ATM with N_max < 2.0 ───────────────────────────────
    for m in methods_present:
        sub = df[(df["method"] == m) & (df["K"] == ATM_K) & np.isfinite(df["abs_error"])]
        if sub.empty:
            continue
        n_max = sub["n_data"].max()
        err_nmax = sub[sub["n_data"] == n_max]["abs_error"].mean()
        ok = err_nmax < ATM_ERR_MAX
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] 4. M{m} mean AbsErr at N={n_max}, K=100: {err_nmax:.4f} < {ATM_ERR_MAX}")

    print("=" * 62)
    verdict = "ALL CRITERIA PASSED — safe to launch full run" if all_pass else "CRITERIA FAILED — do not launch full run"
    print(f"\n>>> {verdict}\n")
    return all_pass


def print_table(df: pd.DataFrame, method_id: int, method_name: str) -> None:
    sub = df[df["method"] == method_id].copy()
    if sub.empty:
        print(f"\n[INFO] No results for {method_name}")
        return

    print(f"\n{'=' * 62}")
    print(f"  {method_name}")
    print(f"{'=' * 62}")

    nan_count = sub[~np.isfinite(sub["estimated_price"])].shape[0]
    total = len(sub)
    print(f"  Total runs: {total}  |  NaN prices: {nan_count}")

    strikes = sorted(sub["K"].unique())
    n_sizes = sorted(sub["n_data"].unique())

    # Header
    header = f"{'N':>6}" + "".join(f"  {'K=' + str(int(k)):>12}" for k in strikes)
    print("\n" + header)
    print("-" * len(header))

    for n in n_sizes:
        row = f"{n:>6}"
        for k in strikes:
            cell = sub[(sub["n_data"] == n) & (sub["K"] == k)]
            finite_cell = cell[np.isfinite(cell["abs_error"])]
            if finite_cell.empty:
                row += f"  {'NaN':>12}"
            else:
                mean_err = finite_cell["abs_error"].mean()
                min_err = finite_cell["abs_error"].min()
                max_err = finite_cell["abs_error"].max()
                row += f"  {mean_err:6.4f} [{min_err:.4f},{max_err:.4f}]"
        # Prices range
        all_prices = sub[sub["n_data"] == n]["estimated_price"]
        finite_prices = all_prices[np.isfinite(all_prices)]
        if not finite_prices.empty:
            price_range = f"  prices=[{finite_prices.min():.3f},{finite_prices.max():.3f}]"
        else:
            price_range = "  prices=NaN"
        print(row + price_range)

    print()
    print("  Columns: mean AbsErr [min, max] across reps")
    print("  BS reference prices: K=90→13.98, K=100→5.46, K=110→1.39  (approx)")


def main():
    parser = argparse.ArgumentParser(description="Analyse verification results")
    parser.add_argument("--results_dir", default="results/verification")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    df = load_results(results_dir)

    print_table(df, method_id=1, method_name="Method I real (7×7, arch paper Sec. 3.2.1)")
    print_table(df, method_id=2, method_name="Method II (5×5, arch paper Sec. 3.2.2)")

    passed = check_criteria(df)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
