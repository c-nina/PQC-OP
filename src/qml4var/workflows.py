"""
PennyLane-based workflows for PQC evaluation and training.

Replaces the myQLM plugin-stack architecture. Each workflow function
accepts a `circuit_fn` (PennyLane QNode returned by hardware_efficient_ansatz)
via kwargs instead of the old `pqc`, `observable`, and `qpu_info` keys.

All public functions preserve their original signatures so that existing
notebook code requires minimal changes (only the workflow_cfg dict changes).
"""

from itertools import product
from typing import Any, Callable, Optional, Union

import numpy as np
import torch

from qml4var.data_utils import empirical_cdf
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


def _trapz_torch(y_tensor: torch.Tensor, x_tensor: torch.Tensor):
    """Trapezoidal integration on torch tensors (differentiable)."""
    if hasattr(torch, "trapezoid"):
        return torch.trapezoid(y_tensor, x_tensor)
    return torch.trapz(y_tensor, x_tensor)


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
        result = circuit_fn(w_t, x_t)
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
    cdf = circuit_fn(w_t, x_t)
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
    Compute CDF predictions for a dataset.

    Returns
    -------
    dict with key 'y_predict_cdf' : np.array shape (N,)
    """

    def cdf_fn(w, x):
        return cdf_workflow(w, x, **kwargs)

    preds = workflow_execution(weights, data_x, cdf_fn, dask_client=dask_client)
    if dask_client is not None:
        preds = dask_client.gather(preds)
    return {"y_predict_cdf": np.array(preds)}


def workflow_for_pdf(
    weights: Union[list, dict, torch.Tensor], data_x: np.ndarray, dask_client: Optional[Any] = None, **kwargs: Any
):
    """
    Compute PDF predictions for a dataset.

    Returns
    -------
    dict with key 'y_predict_pdf' : np.array shape (N,)
    """

    def pdf_fn(w, x):
        return pdf_workflow(w, x, **kwargs)

    preds = workflow_execution(weights, data_x, pdf_fn, dask_client=dask_client)
    if dask_client is not None:
        preds = dask_client.gather(preds)
    return {"y_predict_pdf": np.array(preds)}


# ---------------------------------------------------------------------------
# Differentiable loss (used by torch_gradient)
# ---------------------------------------------------------------------------


def _qdml_loss_torch(
    weights_t: torch.Tensor,
    data_x: np.ndarray,
    data_y: np.ndarray,
    circuit_fn: Callable,
    device: str,
    loss_weights: list,
    minval: Union[float, list],
    maxval: Union[float, list],
    points: int,
    create_graph: bool = False,
):
    """
    Compute QDML loss as a torch scalar.

    Parameters
    ----------
    create_graph : bool
        If True, build the second-order computation graph so that the outer
        backward pass (d(loss)/d(weights_t)) propagates through the PDF term.
        Set to True only when weights_t.requires_grad=True (training gradient).
        Set to False for monitoring evaluations.
    """
    torch_device = torch.device(device)
    data_x_arr = np.asarray(data_x)
    if data_x_arr.ndim == 1:
        data_x_arr = data_x_arr.reshape(-1, 1)

    alpha_0, alpha_1 = float(loss_weights[0]), float(loss_weights[1])

    # --- CDF predictions ---
    cdf_list = []
    for x_i in data_x_arr:
        x_t = torch.tensor(x_i, dtype=torch.float64, device=torch_device)
        cdf_list.append(circuit_fn(weights_t, x_t))
    cdf_preds = torch.stack(cdf_list).reshape(-1, 1)

    labels_t = torch.tensor(np.asarray(data_y).reshape(-1, 1), dtype=torch.float64, device=torch_device)
    loss_cdf = torch.mean((cdf_preds - labels_t) ** 2)

    # --- PDF predictions (d(CDF)/dx, needs create_graph for outer backward) ---
    pdf_list = []
    for x_i in data_x_arr:
        x_t = torch.tensor(x_i, dtype=torch.float64, device=torch_device, requires_grad=True)
        cdf_i = circuit_fn(weights_t, x_t)
        pdf_i = torch.autograd.grad(cdf_i, x_t, create_graph=create_graph)[0]
        pdf_list.append(pdf_i.sum())
    pdf_preds = torch.stack(pdf_list).reshape(-1, 1)

    mean_pdf = torch.mean(pdf_preds)

    # --- Integral of PDF² over the domain ---
    x_integral = np.linspace(
        np.asarray(minval).reshape(-1), np.asarray(maxval).reshape(-1), int(points)
    )  # shape (points, n_features)
    domain_x = np.array(list(product(*[x_integral[:, i] for i in range(x_integral.shape[1])])))

    pdf_sq_list = []
    for x_i in domain_x:
        x_t = torch.tensor(x_i, dtype=torch.float64, device=torch_device, requires_grad=True)
        cdf_i = circuit_fn(weights_t, x_t)
        pdf_i = torch.autograd.grad(cdf_i, x_t, create_graph=create_graph)[0]
        pdf_sq_list.append((pdf_i.sum()) ** 2)

    pdf_sq_tensor = torch.stack(pdf_sq_list)

    if domain_x.shape[1] == 1:
        x_dom_t = torch.tensor(domain_x[:, 0], dtype=torch.float64, device=torch_device)
        integral = _trapz_torch(pdf_sq_tensor, x_dom_t)
    else:
        # Monte Carlo for higher dimensions
        factor = float(np.prod(domain_x.max(axis=0) - domain_x.min(axis=0)) / domain_x.shape[0])
        integral = pdf_sq_tensor.sum() * factor

    return alpha_0 * loss_cdf + alpha_1 * (-2.0 * mean_pdf + integral)


# ---------------------------------------------------------------------------
# High-level workflow functions (same API as before)
# ---------------------------------------------------------------------------


def qdml_loss_workflow(
    weights: Union[list, dict, torch.Tensor],
    data_x: np.ndarray,
    data_y: np.ndarray,
    dask_client: Optional[Any] = None,
    **kwargs: Any,
):
    """
    Compute QDML loss.

    When weights is a torch.Tensor with requires_grad=True (called from
    torch_gradient), returns a differentiable scalar.
    Otherwise, returns a plain float for monitoring.

    Parameters
    ----------
    weights : list, dict, or torch.Tensor
    data_x : np.array
    data_y : np.array
    dask_client : ignored (kept for API compatibility)
    kwargs : must contain circuit_fn, torch_device, loss_weights,
             minval, maxval, points.
    """
    circuit_fn = kwargs["circuit_fn"]
    device = kwargs.get("torch_device", "cpu")
    loss_weights = kwargs.get("loss_weights", [1.0, 5.0])
    minval = kwargs.get("minval")
    maxval = kwargs.get("maxval")
    points = kwargs.get("points")

    if isinstance(weights, torch.Tensor):
        # Called from torch_gradient: need create_graph=True for PDF term
        return _qdml_loss_torch(
            weights, data_x, data_y, circuit_fn, device, loss_weights, minval, maxval, points, create_graph=True
        )
    # Called for monitoring: create_graph=False, return plain float
    weights_t = _weights_to_tensor(weights, device)
    return _qdml_loss_torch(
        weights_t, data_x, data_y, circuit_fn, device, loss_weights, minval, maxval, points, create_graph=False
    ).item()


def unsupervised_qdml_loss_workflow(
    weights: Union[list, dict, torch.Tensor],
    data_x: np.ndarray,
    dask_client: Optional[Any] = None,
    empirical_shift: float = -0.5,
    **kwargs: Any,
):
    """
    Unsupervised QDML loss: labels built from empirical CDF of data_x.

    Parameters
    ----------
    Same as qdml_loss_workflow, minus data_y.
    empirical_shift : float
        Constant shift added to empirical CDF labels. Default -0.5.
    """
    data_x = np.asarray(data_x)
    if data_x.ndim == 1:
        data_x = data_x.reshape(-1, 1)
    data_y = empirical_cdf(data_x).reshape(-1, 1) + empirical_shift
    return qdml_loss_workflow(weights, data_x, data_y, dask_client=dask_client, **kwargs)


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
