"""
Utilities for saving training runs.

A run folder is created once per experiment with a timestamp:

    results/2026-03-25_14-30-00/
    ├── config.json                     ← dataset + pqc + optimizer merged
    ├── evolution.csv                   ← per print_step: epoch, loss, mse, weights
    ├── summary.csv                     ← per checkpoint: epoch, train_loss, test_mse, option_price
    └── checkpoints/
        ├── epoch_0000_predictions.csv  ← x, cdf_pred, pdf_pred on dense grid
        ├── epoch_0050_predictions.csv
        └── ...

Usage
-----
    from qml4var.results import create_run_folder, save_config, make_checkpoint_fn

    folder = create_run_folder("results")
    save_config(folder, dataset=data_cfg, pqc=pqc_cfg, optimizer=opt_cfg)

    checkpoint_fn = make_checkpoint_fn(
        folder=folder,
        x_grid=np.linspace(minval, maxval, 200).reshape(-1, 1),
        workflow_cfg=workflow_cfg,          # dict with circuit_fn, minval, maxval, …
        price_fn=lambda w: estimate_price_from_trained_pqc(w, ...),  # optional
    )

    adam_optimizer_loop(
        ...,
        checkpoint_fn=checkpoint_fn,
        checkpoint_step=50,
    )
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Folder and config helpers
# ---------------------------------------------------------------------------


def create_run_folder(base_path: str = "results") -> str:
    """
    Create a timestamped folder for one training run.

    Parameters
    ----------
    base_path : str
        Parent directory (created if it does not exist).

    Returns
    -------
    str
        Absolute path to the new run folder, e.g.
        ``results/2026-03-25_14-30-00``.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = pathlib.Path(base_path) / timestamp
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "checkpoints").mkdir(exist_ok=True)
    return str(folder)


def save_config(folder: str, **configs: Any) -> None:
    """
    Save all configuration dicts as a single ``config.json``.

    Parameters
    ----------
    folder : str
        Run folder returned by :func:`create_run_folder`.
    **configs : dict
        Named config sections, e.g. ``dataset={...}, pqc={...}, optimizer={...}``.

    Example
    -------
    >>> save_config(folder, dataset=data_cfg, pqc=pqc_cfg, optimizer=opt_cfg)
    """
    config_path = pathlib.Path(folder) / "config.json"
    config_path.write_text(json.dumps(configs, indent=2, default=str))


# ---------------------------------------------------------------------------
# Checkpoint function factory
# ---------------------------------------------------------------------------


def make_checkpoint_fn(
    folder: str,
    x_grid: np.ndarray,
    workflow_cfg: dict,
    price_fn: Optional[Callable] = None,
) -> Callable:
    """
    Build a checkpoint callable to pass to :func:`~qml4var.adam.adam_optimizer_loop`.

    At each checkpoint the following is saved:

    * A row appended to ``summary.csv`` with epoch, train_loss, test_mse,
      and option_price (if *price_fn* is provided).
    * A file ``checkpoints/epoch_{N:04d}_predictions.csv`` with columns
      ``x``, ``cdf_pred``, ``pdf_pred`` evaluated on *x_grid*.

    Parameters
    ----------
    folder : str
        Run folder returned by :func:`create_run_folder`.
    x_grid : np.ndarray, shape (N, n_features)
        Dense evaluation grid for CDF/PDF predictions.
    workflow_cfg : dict
        Keyword arguments forwarded to ``workflow_for_cdf`` and
        ``workflow_for_pdf`` (must contain at least ``circuit_fn``).
    price_fn : callable, optional
        ``price_fn(weights) -> float``.  If provided, the estimated option
        price is included in ``summary.csv``.

    Returns
    -------
    callable
        ``checkpoint(weights, epoch, train_loss, test_mse)`` — matches the
        signature expected by ``adam_optimizer_loop``.
    """
    from qml4var.workflows import workflow_for_cdf, workflow_for_pdf

    folder_path = pathlib.Path(folder)
    summary_path = folder_path / "summary.csv"

    # Write header on first call flag
    _state = {"header_written": False}

    def checkpoint(
        weights: Any,
        epoch: int,
        train_loss: float,
        test_mse: Optional[float] = None,
    ) -> None:
        # --- scalar summary ---
        row: dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_mse": test_mse,
            "option_price": np.nan,
        }
        if price_fn is not None:
            try:
                row["option_price"] = price_fn(weights)
            except Exception as exc:
                print(f"[checkpoint] price_fn failed at epoch {epoch}: {exc}")

        pdf_summary = pd.DataFrame([row])
        write_mode = "w" if not _state["header_written"] else "a"
        pdf_summary.to_csv(
            summary_path,
            sep=";",
            index=False,
            mode=write_mode,
            header=not _state["header_written"],
        )
        _state["header_written"] = True

        # --- CDF / PDF predictions on the dense grid ---
        x_arr = np.asarray(x_grid)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

        cdf_preds = workflow_for_cdf(weights, x_arr, **workflow_cfg)["y_predict_cdf"]
        pdf_preds = workflow_for_pdf(weights, x_arr, **workflow_cfg)["y_predict_pdf"]

        pred_df = pd.DataFrame(
            {
                "x": x_arr[:, 0],
                "cdf_pred": np.asarray(cdf_preds).reshape(-1),
                "pdf_pred": np.asarray(pdf_preds).reshape(-1),
            }
        )
        pred_path = folder_path / "checkpoints" / f"epoch_{epoch:04d}_predictions.csv"
        pred_df.to_csv(pred_path, sep=";", index=False)

    return checkpoint
