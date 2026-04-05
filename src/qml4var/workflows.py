"""
PennyLane-based workflows for PQC evaluation and training.

Replaces the myQLM plugin-stack architecture. Each workflow function
accepts a `circuit_fn` (PennyLane QNode returned by hardware_efficient_ansatz)
via kwargs instead of the old `pqc`, `observable`, and `qpu_info` keys.

All public functions preserve their original signatures so that existing
notebook code requires minimal changes (only the workflow_cfg dict changes).
"""

from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from torch.func import vmap as _vmap

from qml4var.losses import mse

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weights_to_tensor(weights: Union[list, dict, torch.Tensor], device: str):
    """Convert weights (list or dict or tensor) to a float64 torch tensor."""
    if isinstance(weights, torch.Tensor):
        return weights
    values = list(weights.values()) if isinstance(weights, dict) else list(weights)
    return torch.tensor(values, dtype=torch.float64, device=torch.device(device))


# ---------------------------------------------------------------------------
# Single-sample evaluation (no gradient tracking on weights)
# ---------------------------------------------------------------------------


def cdf_workflow(weights: Union[list, dict, torch.Tensor], x_sample: np.ndarray, **kwargs: Any):
    """
    Evaluate CDF for one sample.

    Parameters
    ----------
    weights : list, dict, or torch.Tensor
        PQC weights.
    x_sample : np.array, shape (n_features,) or (1, n_features)
        Raw input feature sample.
    kwargs : dict
        Must contain:
        - circuit_fn : PennyLane QNode from hardware_efficient_ansatz
        Optional:
        - torch_device : str (default "cpu")

    Returns
    -------
    float
        CDF value for this sample.
    """
    circuit_fn = kwargs["circuit_fn"]
    device = kwargs.get("torch_device", "cpu")

    w_t = _weights_to_tensor(weights, device)
    x_flat = np.asarray(x_sample).reshape(-1)
    x_t = torch.tensor(x_flat, dtype=torch.float64, device=torch.device(device))

    with torch.no_grad():
        result = (circuit_fn(w_t, x_t) + 1.0) / 2.0  # map [-1,1] → [0,1]
    return result.item()


def pdf_workflow(weights: Union[list, dict, torch.Tensor], x_sample: np.ndarray, **kwargs: Any):
    """
    Evaluate PDF for one sample via autograd: PDF = d(CDF)/d(raw_feature).

    Parameters
    ----------
    Same as cdf_workflow.

    Returns
    -------
    float
        PDF value (derivative of CDF w.r.t. raw input feature).
    """
    circuit_fn = kwargs["circuit_fn"]
    device = kwargs.get("torch_device", "cpu")

    w_t = _weights_to_tensor(weights, device)
    if isinstance(w_t, torch.Tensor) and w_t.requires_grad:
        w_t = w_t.detach()

    x_flat = np.asarray(x_sample).reshape(-1)
    x_t = torch.tensor(x_flat, dtype=torch.float64, device=torch.device(device), requires_grad=True)
    cdf = (circuit_fn(w_t, x_t) + 1.0) / 2.0  # map [-1,1] → [0,1]; PDF = 0.5 * d(circuit)/dx
    cdf.backward()
    return x_t.grad.sum().item()


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------


def workflow_execution(
    weights: Union[list, dict, torch.Tensor], data_x: np.ndarray, workflow: Callable, dask_client: Optional[Any] = None
):
    """
    Execute a workflow function for every sample in data_x.

    Parameters
    ----------
    weights : list/dict/tensor
    data_x : np.array, shape (N, n_features)
    workflow : callable, workflow(weights, x_sample) -> float
    dask_client : optional

    Returns
    -------
    list or list of dask futures
    """
    if dask_client is None:
        return [workflow(weights, x_) for x_ in data_x]
    return dask_client.map(workflow, *([weights] * data_x.shape[0], data_x))


def workflow_for_cdf(
    weights: Union[list, dict, torch.Tensor], data_x: np.ndarray, dask_client: Optional[Any] = None, **kwargs: Any
):
    """
    Compute CDF predictions for a dataset (batched via qml.vmap).

    Returns
    -------
    dict with key 'y_predict_cdf' : np.array shape (N,)
    """
    if dask_client is not None:
        def cdf_fn(w, x):
            return cdf_workflow(w, x, **kwargs)
        preds = dask_client.gather(workflow_execution(weights, data_x, cdf_fn, dask_client=dask_client))
        return {"y_predict_cdf": np.array(preds)}

    circuit_fn = kwargs["circuit_fn"]
    device = kwargs.get("torch_device", "cpu")
    w_t = _weights_to_tensor(weights, device)
    data_x_arr = np.asarray(data_x)
    if data_x_arr.ndim == 1:
        data_x_arr = data_x_arr.reshape(-1, 1)
    x_batch = torch.tensor(data_x_arr, dtype=torch.float64, device=torch.device(device))
    with torch.no_grad():
        raw = _vmap(circuit_fn, in_dims=(None, 0))(w_t, x_batch)
        preds = ((raw + 1.0) / 2.0).detach().cpu().numpy().reshape(-1)  # map [-1,1] → [0,1]
    return {"y_predict_cdf": preds}


def workflow_for_pdf(
    weights: Union[list, dict, torch.Tensor], data_x: np.ndarray, dask_client: Optional[Any] = None, **kwargs: Any
):
    """
    Compute PDF predictions for a dataset (batched via qml.vmap).

    Returns
    -------
    dict with key 'y_predict_pdf' : np.array shape (N,)
    """
    if dask_client is not None:
        def pdf_fn(w, x):
            return pdf_workflow(w, x, **kwargs)
        preds = dask_client.gather(workflow_execution(weights, data_x, pdf_fn, dask_client=dask_client))
        return {"y_predict_pdf": np.array(preds)}

    circuit_fn = kwargs["circuit_fn"]
    device = kwargs.get("torch_device", "cpu")
    w_t = _weights_to_tensor(weights, device)
    if isinstance(w_t, torch.Tensor) and w_t.requires_grad:
        w_t = w_t.detach()
    data_x_arr = np.asarray(data_x)
    if data_x_arr.ndim == 1:
        data_x_arr = data_x_arr.reshape(-1, 1)
    x_batch = torch.tensor(data_x_arr, dtype=torch.float64, device=torch.device(device), requires_grad=True)
    cdf_batch = (_vmap(circuit_fn, in_dims=(None, 0))(w_t, x_batch) + 1.0) / 2.0  # map to [0,1]
    pdf_grads = torch.autograd.grad(cdf_batch.sum(), x_batch)[0]  # (N, n_features), = 0.5*d(circuit)/dx
    preds = pdf_grads.sum(dim=1).detach().cpu().numpy().reshape(-1)
    return {"y_predict_pdf": preds}


def mse_workflow(
    weights: Union[list, dict, torch.Tensor],
    data_x: np.ndarray,
    data_y: np.ndarray,
    dask_client: Optional[Any] = None,
    **kwargs: Any,
):
    """MSE of CDF predictions (numpy, for metric evaluation)."""
    out = workflow_for_cdf(weights, data_x, dask_client=dask_client, **kwargs)
    return mse(data_y, out["y_predict_cdf"])


def dft_from_trained_pqc(
    weights: Union[list, dict, torch.Tensor],
    minval: float = -2.0 * np.pi,
    maxval: float = 2.0 * np.pi,
    points: int = 256,
    prediction: str = "cdf",
    dask_client: Optional[Any] = None,
    **kwargs: Any,
):
    """
    Compute DFT coefficients from direct evaluations of a trained PQC.

    Parameters
    ----------
    weights : list/dict/tensor
    minval, maxval : float
        Evaluation interval bounds.
    points : int
        Number of grid points.
    prediction : str
        "cdf" or "pdf".
    kwargs : must contain circuit_fn and features_names.

    Returns
    -------
    dict with x_domain, y_predict, k_values, c_k.
    """
    if points < 2:
        raise ValueError("points must be >= 2")
    if maxval <= minval:
        raise ValueError("maxval must be greater than minval")
    features_names = kwargs.get("features_names")
    if features_names is None:
        raise ValueError("features_names must be provided in kwargs")
    if len(features_names) != 1:
        raise ValueError("dft_from_trained_pqc currently supports only 1 feature")

    x_domain = np.linspace(minval, maxval, points, endpoint=False)
    data_x = x_domain.reshape(-1, 1)

    if prediction == "cdf":
        y_predict = workflow_for_cdf(weights, data_x, dask_client=dask_client, **kwargs)["y_predict_cdf"]
    elif prediction == "pdf":
        y_predict = workflow_for_pdf(weights, data_x, dask_client=dask_client, **kwargs)["y_predict_pdf"]
    else:
        raise ValueError("prediction must be 'cdf' or 'pdf'")

    y_predict = np.asarray(y_predict).reshape(-1)
    c_k = np.fft.fft(y_predict) / points
    k_values = np.fft.fftfreq(points, d=1.0 / points).astype(int)
    order = np.argsort(k_values)

    return {
        "x_domain": x_domain,
        "y_predict": y_predict,
        "k_values": k_values[order],
        "c_k": c_k[order],
    }
