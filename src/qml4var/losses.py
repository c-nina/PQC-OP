"""
Functions for computing losses and gradients.
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Numerical helpers (device-agnostic, kept for metric evaluation)
# ---------------------------------------------------------------------------


def _trapz_compat(y, x):
    """Compatibility wrapper for NumPy trapezoidal integration."""
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trapz_fn(y=y, x=x)


def compute_integral(y_array, x_array, dask_client=None):
    """
    Numerical integral of y_array over x_array (trapezoidal or Monte Carlo).

    Parameters
    ----------
    y_array : np.array
        y values. Shape (n,).
    x_array : np.array
        Domain. Shape (n, n_features).
    dask_client : optional
        Dask client for distributed execution.

    Returns
    -------
    float or future
    """
    if x_array.shape[1] == 1:
        if dask_client is None:
            return _trapz_compat(y=y_array, x=x_array[:, 0])
        return dask_client.submit(_trapz_compat, y_array, x_array[:, 0])

    if x_array.shape[1] == 2:
        if dask_client is None:
            x_domain, y_domain = np.meshgrid(
                np.unique(x_array[:, 0]), np.unique(x_array[:, 1])
            )
            y_array_ = y_array.reshape(x_domain.shape)
            return _trapz_compat(_trapz_compat(y=y_array_, x=x_domain), y_domain[:, 0])
        factor = np.prod(x_array.max(axis=0) - x_array.min(axis=0)) / len(y_array)
        return dask_client.submit(lambda x: np.sum(x) * factor, y_array)

    # Monte Carlo
    factor = np.prod(x_array.max(axis=0) - x_array.min(axis=0)) / y_array.size
    if dask_client is None:
        return np.sum(y_array) * factor
    factor = np.prod(x_array.max(axis=0) - x_array.min(axis=0)) / len(y_array)
    return dask_client.submit(lambda x: np.sum(x) * factor, y_array)


def trapezoidal_rule(x_domain, y_range):
    """Trapezoidal integration."""
    dx = np.diff(x_domain)
    return np.dot((y_range[:-1] + y_range[1:]) / 2, dx)


def loss_function_qdml(labels, predict_cdf, predict_pdf, integral, loss_weights=None):
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
    loss_1 = np.mean(error_ ** 2)
    if predict_pdf.shape != labels.shape:
        raise ValueError("predict_pdf and labels have different shape!")
    mean = -2 * np.mean(predict_pdf)
    return alpha_0 * loss_1 + alpha_1 * (mean + integral)


def mse(labels, prediction):
    """Mean Squared Error (numpy)."""
    error_ = prediction - labels.reshape(prediction.shape)
    return np.mean(error_ ** 2)


# ---------------------------------------------------------------------------
# PyTorch gradient (replaces numeric_gradient)
# ---------------------------------------------------------------------------

def torch_gradient(weights, data_x, data_y, loss_fn):
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

def numeric_gradient(weights, data_x, data_y, loss):
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
