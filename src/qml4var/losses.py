"""
Functions for computing losses and gradients.
"""

from itertools import product
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from torch.func import vmap as _vmap

from qml4var.data_utils import empirical_cdf

# ---------------------------------------------------------------------------
# Loss functions (numpy — for metric evaluation, not differentiated)
# ---------------------------------------------------------------------------


def loss_function_qdml(
    labels: np.ndarray,
    predict_cdf: np.ndarray,
    predict_pdf: np.ndarray,
    integral: float,
    loss_weights: Optional[List[float]] = None,
):
    """
    QDML loss (numpy, for metric evaluation — not differentiated).

    L = alpha_0 * MSE(CDF) + alpha_1 * (-2*mean(PDF) + integral(PDF²))
    """
    if loss_weights is None:
        loss_weights = [1.0, 5.0]
    alpha_0, alpha_1 = loss_weights[0], loss_weights[1]
    if predict_cdf.shape != labels.shape:
        raise ValueError("predict_cdf and labels have different shape!")
    error_ = predict_cdf - labels
    loss_1 = np.mean(error_**2)
    if predict_pdf.shape != labels.shape:
        raise ValueError("predict_pdf and labels have different shape!")
    mean = -2 * np.mean(predict_pdf)
    return alpha_0 * loss_1 + alpha_1 * (mean + integral)


def mse(labels: np.ndarray, prediction: np.ndarray):
    """Mean Squared Error (numpy)."""
    error_ = prediction - labels.reshape(prediction.shape)
    return np.mean(error_**2)


# ---------------------------------------------------------------------------
# Differentiable loss (torch)
# ---------------------------------------------------------------------------


def _trapz_torch(y_tensor: torch.Tensor, x_tensor: torch.Tensor):
    """Trapezoidal integration on torch tensors (differentiable)."""
    if hasattr(torch, "trapezoid"):
        return torch.trapezoid(y_tensor, x_tensor)
    return torch.trapz(y_tensor, x_tensor)


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

    # Batched circuit: evaluates all samples in one GPU call instead of a Python loop.
    # grad(sum_i CDF(w, x_i), x)[i] = dCDF(w,x_i)/dx_i  (samples are independent).
    batched_circuit = _vmap(circuit_fn, in_dims=(None, 0))

    # --- CDF predictions: map circuit output [-1,1] → [0,1] ---
    x_cdf = torch.tensor(data_x_arr, dtype=torch.float64, device=torch_device)
    cdf_preds = ((batched_circuit(weights_t, x_cdf) + 1.0) / 2.0).reshape(-1, 1)

    labels_t = torch.tensor(np.asarray(data_y).reshape(-1, 1), dtype=torch.float64, device=torch_device)
    loss_cdf = torch.mean((cdf_preds - labels_t) ** 2)

    # --- PDF predictions: d(CDF)/dx — autograd through the mapped output gives 0.5*d(circuit)/dx ---
    x_pdf = torch.tensor(data_x_arr, dtype=torch.float64, device=torch_device, requires_grad=True)
    cdf_for_pdf = (batched_circuit(weights_t, x_pdf) + 1.0) / 2.0  # (N,) in [0,1]
    pdf_grads = torch.autograd.grad(cdf_for_pdf.sum(), x_pdf, create_graph=create_graph)[0]  # (N, n_features)
    mean_pdf = pdf_grads.sum(dim=1).mean()

    # --- Integral of PDF² over the domain (batched) ---
    x_integral = np.linspace(
        np.asarray(minval).reshape(-1), np.asarray(maxval).reshape(-1), int(points)
    )  # shape (points, n_features)
    domain_x = np.array(list(product(*[x_integral[:, i] for i in range(x_integral.shape[1])])))

    x_int = torch.tensor(domain_x, dtype=torch.float64, device=torch_device, requires_grad=True)
    cdf_int = (batched_circuit(weights_t, x_int) + 1.0) / 2.0  # (M,) in [0,1]
    pdf_int_grads = torch.autograd.grad(cdf_int.sum(), x_int, create_graph=create_graph)[0]  # (M, n_features)
    pdf_sq_tensor = pdf_int_grads.sum(dim=1) ** 2  # (M,)

    if domain_x.shape[1] == 1:
        x_dom_t = torch.tensor(domain_x[:, 0], dtype=torch.float64, device=torch_device)
        integral = _trapz_torch(pdf_sq_tensor, x_dom_t)
    else:
        # Monte Carlo for higher dimensions
        factor = float(np.prod(domain_x.max(axis=0) - domain_x.min(axis=0)) / domain_x.shape[0])
        integral = pdf_sq_tensor.sum() * factor

    # --- Boundary constraint: CDF(a) ≈ 0, CDF(b) ≈ 1 (paper Sec. 3.2.2) ---
    # Evaluate at fixed domain endpoints (minval, maxval) instead of empirical data
    # min/max so the anchoring is always at the true training domain boundaries [-π, π].
    # This is especially important when shift_feature ≠ 0: the empirical min/max of the
    # training batch and the intended domain endpoints may diverge slightly, but the IBP
    # formula always integrates over [-π, π], so we anchor there.
    n_feat = data_x_arr.shape[1]
    x_bound_a = torch.tensor(
        np.asarray(minval).reshape(1, n_feat), dtype=torch.float64, device=torch_device
    )
    x_bound_b = torch.tensor(
        np.asarray(maxval).reshape(1, n_feat), dtype=torch.float64, device=torch_device
    )
    cdf_at_min = (batched_circuit(weights_t, x_bound_a) + 1.0) / 2.0  # (1,)
    cdf_at_max = (batched_circuit(weights_t, x_bound_b) + 1.0) / 2.0  # (1,)
    loss_boundary = cdf_at_min[0] ** 2 + (cdf_at_max[0] - 1.0) ** 2

    return alpha_0 * loss_cdf + alpha_1 * (-2.0 * mean_pdf + integral) + loss_boundary


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
    weights_t = torch.tensor(
        list(weights.values()) if isinstance(weights, dict) else list(weights),
        dtype=torch.float64,
        device=torch.device(device),
    )
    return _qdml_loss_torch(
        weights_t, data_x, data_y, circuit_fn, device, loss_weights, minval, maxval, points, create_graph=False
    ).item()


def unsupervised_qdml_loss_workflow(
    weights: Union[list, dict, torch.Tensor],
    data_x: np.ndarray,
    dask_client: Optional[Any] = None,
    **kwargs: Any,
):
    """
    Unsupervised QDML loss (Method II): labels built from empirical CDF of data_x.

    Labels are in [0,1] (no shift). The circuit output is mapped [-1,1]→[0,1]
    inside qdml_loss_workflow, so both sides are consistent.

    Parameters
    ----------
    Same as qdml_loss_workflow, minus data_y.
    """
    data_x = np.asarray(data_x)
    if data_x.ndim == 1:
        data_x = data_x.reshape(-1, 1)
    data_y = empirical_cdf(data_x).reshape(-1, 1)  # [0,1], no shift
    return qdml_loss_workflow(weights, data_x, data_y, dask_client=dask_client, **kwargs)


# ---------------------------------------------------------------------------
# Method I — H¹ supervised loss (paper Sec. 3.2.1, eq. 12)
# ---------------------------------------------------------------------------


def method_I_h1_loss(
    weights_t: torch.Tensor,
    data_x: np.ndarray,
    pdf_labels: np.ndarray,
    pdf_deriv_labels: np.ndarray,
    circuit_fn: Callable,
    device: str,
    alpha_0: float = 0.9,
    alpha_1: float = 0.1,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    H¹ supervised loss for Method I (paper eq. 12, Sec. 3.2.1).

    The circuit approximates the PDF f(x) directly. Labels are analytical
    evaluations of the target PDF and its derivative.

    L = alpha_0 * (1/I) * Σ(f*(xi) - f(xi))²
      + alpha_1 * (1/I) * Σ(df*/dx(xi) - df/dx(xi))²

    Default weights from paper Table 1: alpha_0=0.9, alpha_1=0.1.

    Parameters
    ----------
    data_x : np.ndarray, shape (I,) or (I, 1)
        Training points in rescaled space [-2π, 2π].
    pdf_labels : np.ndarray, shape (I,)
        f*(xi) — analytical PDF evaluated at data_x.
    pdf_deriv_labels : np.ndarray, shape (I,)
        df*/dx(xi) — analytical PDF derivative at data_x.
    create_graph : bool
        Set True when called from torch_gradient (training).
    """
    torch_device = torch.device(device)
    data_x_arr = np.asarray(data_x).reshape(-1, 1)

    x_t = torch.tensor(data_x_arr, dtype=torch.float64, device=torch_device, requires_grad=True)
    pdf_t = torch.tensor(pdf_labels.reshape(-1), dtype=torch.float64, device=torch_device)
    dpdf_t = torch.tensor(pdf_deriv_labels.reshape(-1), dtype=torch.float64, device=torch_device)

    batched = _vmap(circuit_fn, in_dims=(None, 0))

    # Circuit approximates PDF directly (no [-1,1]→[0,1] mapping for PDF)
    pdf_pred = batched(weights_t, x_t).reshape(-1)

    # Term 1: MSE on PDF values (alpha_0 = 0.9)
    loss_values = torch.mean((pdf_t - pdf_pred) ** 2)

    # Term 2: MSE on derivatives w.r.t. input x (alpha_1 = 0.1)
    # create_graph=True so the gradient can propagate back to weights_t
    pdf_deriv_pred = torch.autograd.grad(
        pdf_pred.sum(), x_t, create_graph=create_graph
    )[0].reshape(-1)
    loss_derivs = torch.mean((dpdf_t - pdf_deriv_pred) ** 2)

    return alpha_0 * loss_values + alpha_1 * loss_derivs


# ---------------------------------------------------------------------------
# PyTorch gradient (replaces numeric_gradient)
# ---------------------------------------------------------------------------


def torch_gradient(weights: list, data_x: np.ndarray, data_y: np.ndarray, loss_fn: Callable):
    """
    Compute the gradient of loss_fn w.r.t. weights using PyTorch autograd
    (backpropagation through the PennyLane QNode).

    This replaces numeric_gradient: one backward pass replaces 2*N_weights
    forward passes.

    Parameters
    ----------
    weights : list of float
        Current model weights.
    data_x : np.array
        Feature batch.
    data_y : np.array
        Label batch.
    loss_fn : callable
        loss_fn(weights_tensor, data_x, data_y) -> torch scalar.

    Returns
    -------
    list of float
        Gradient vector with same length as weights.
    """
    weights_t = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    loss = loss_fn(weights_t, data_x, data_y)
    loss.backward()
    return weights_t.grad.tolist()


# ---------------------------------------------------------------------------
# Legacy numeric gradient (kept for reference / SPSA notebooks)
# ---------------------------------------------------------------------------


def numeric_gradient(weights: list, data_x: np.ndarray, data_y: np.ndarray, loss: Callable):
    """
    Finite-difference gradient (legacy). Prefer torch_gradient for speed.

    Parameters
    ----------
    weights : list of float
    data_x : np.array
    data_y : np.array
    loss : callable
        loss(weights, data_x, data_y) -> float

    Returns
    -------
    list of float
    """
    import copy

    gradient_i = []
    epsilon = 1.0e-7
    for i, weight in enumerate(weights):
        new_weights = copy.deepcopy(weights)
        new_weights[i] = weight + epsilon
        loss_plus = loss(new_weights, data_x, data_y)
        new_weights = copy.deepcopy(weights)
        new_weights[i] = weight - epsilon
        loss_minus = loss(new_weights, data_x, data_y)
        gradient_i.append((loss_plus - loss_minus) / (2.0 * epsilon))
    return gradient_i
