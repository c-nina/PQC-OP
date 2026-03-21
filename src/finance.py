"""Utility routines for CrisTFM experiments (PennyLane backend)."""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _weights_to_tensor(weights, device):
    if isinstance(weights, torch.Tensor):
        return weights
    values = list(weights.values()) if isinstance(weights, dict) else list(weights)
    return torch.tensor(values, dtype=torch.float64, device=torch.device(device))


# ---------------------------------------------------------------------------
# Single-sample workflow functions (PennyLane replacements)
# ---------------------------------------------------------------------------

def cdf_workflow_cris(weights, x_sample, **kwargs):
    """
    Compute CDF for one sample.

    Parameters
    ----------
    weights : list, dict, or torch.Tensor
    x_sample : np.array, shape (n_features,)
    kwargs : must contain circuit_fn; optional torch_device.

    Returns
    -------
    float
    """
    circuit_fn = kwargs["circuit_fn"]
    device = kwargs.get("torch_device", "cpu")

    w_t = _weights_to_tensor(weights, device)
    x_flat = np.asarray(x_sample).reshape(-1)
    x_t = torch.tensor(x_flat, dtype=torch.float64, device=torch.device(device))

    with torch.no_grad():
        result = circuit_fn(w_t, x_t)
    return result.item()


def pdf_workflow_cris(weights, x_sample, **kwargs):
    """
    Compute PDF = d(CDF)/d(raw_feature) for one sample.

    Parameters
    ----------
    Same as cdf_workflow_cris.

    Returns
    -------
    float
    """
    circuit_fn = kwargs["circuit_fn"]
    device = kwargs.get("torch_device", "cpu")

    w_t = _weights_to_tensor(weights, device)
    if isinstance(w_t, torch.Tensor) and w_t.requires_grad:
        w_t = w_t.detach()

    x_flat = np.asarray(x_sample).reshape(-1)
    x_t = torch.tensor(
        x_flat, dtype=torch.float64, device=torch.device(device), requires_grad=True
    )
    cdf = circuit_fn(w_t, x_t)
    cdf.backward()
    return x_t.grad.sum().item()


def pdf_derivative_workflow_cris(weights, x_sample, **kwargs):
    """
    Compute d(PDF)/dx = d²(CDF)/dx² for one sample via second-order autograd.

    Parameters
    ----------
    Same as cdf_workflow_cris.

    Returns
    -------
    float
    """
    circuit_fn = kwargs["circuit_fn"]
    device = kwargs.get("torch_device", "cpu")

    w_t = _weights_to_tensor(weights, device)
    if isinstance(w_t, torch.Tensor) and w_t.requires_grad:
        w_t = w_t.detach()

    x_flat = np.asarray(x_sample).reshape(-1)
    x_t = torch.tensor(
        x_flat, dtype=torch.float64, device=torch.device(device), requires_grad=True
    )
    cdf = circuit_fn(w_t, x_t)
    pdf = torch.autograd.grad(cdf, x_t, create_graph=True)[0]
    pdf_deriv = torch.autograd.grad(pdf.sum(), x_t)[0]
    return pdf_deriv.sum().item()


def workflow_execution_cris(weights, data_x, workflow, dask_client=None):
    """Execute one workflow for all input samples."""
    if dask_client is None:
        return [workflow(weights, x_) for x_ in data_x]
    return dask_client.map(workflow, *([weights] * data_x.shape[0], data_x))


def workflow_for_pdf_and_derivative_cris(
        weights,
        data_x,
        labels_pdf=None,
        labels_pdf_derivative=None,
        dask_client=None,
        **kwargs):
    """
    Compute PDF and d(PDF)/dx predictions for a full dataset.

    Parameters
    ----------
    weights : list/dict/tensor
    data_x : np.array
    labels_pdf, labels_pdf_derivative : np.array or None
    dask_client : optional
    kwargs : must contain circuit_fn; optional torch_device.

    Returns
    -------
    dict with predict_pdf, predict_pdf_derivative, and optionally labels.
    """
    def pdf_fn(w, x):
        return pdf_workflow_cris(w, x, **kwargs)

    def pdf_deriv_fn(w, x):
        return pdf_derivative_workflow_cris(w, x, **kwargs)

    predict_pdf = workflow_execution_cris(
        weights, data_x, pdf_fn, dask_client=dask_client
    )
    predict_pdf_derivative = workflow_execution_cris(
        weights, data_x, pdf_deriv_fn, dask_client=dask_client
    )

    if dask_client is None:
        predict_pdf = np.asarray(predict_pdf).reshape(-1, 1)
        predict_pdf_derivative = np.asarray(predict_pdf_derivative).reshape(-1, 1)
    else:
        predict_pdf = np.asarray(dask_client.gather(predict_pdf)).reshape(-1, 1)
        predict_pdf_derivative = np.asarray(
            dask_client.gather(predict_pdf_derivative)
        ).reshape(-1, 1)

    output = {
        "predict_pdf": predict_pdf,
        "predict_pdf_derivative": predict_pdf_derivative,
    }
    if labels_pdf is not None:
        output["labels_pdf"] = labels_pdf
    if labels_pdf_derivative is not None:
        output["labels_pdf_derivative"] = labels_pdf_derivative
    return output


# ---------------------------------------------------------------------------
# Fourier analysis (device-independent, unchanged)
# ---------------------------------------------------------------------------

def complex_fourier_coefficients(x_domain, y_predict, k_values, interval=None):
    """
    Compute complex Fourier coefficients c_k from sampled function values.

    Mathematical definition
    -----------------------
    For a function f(x) in a truncated interval [a, b], with L = b - a:

        c_k = (1 / L) * integral_a^b f(x) * exp(-i * 2*pi*k*(x-a)/L) dx

    The integral is approximated with the trapezoidal rule on `x_domain`.
    """
    x_domain = np.asarray(x_domain).reshape(-1)
    if x_domain.size < 2:
        raise ValueError("x_domain must contain at least two points")

    y_predict = np.asarray(y_predict)
    if y_predict.ndim == 2 and y_predict.shape[1] == 1:
        y_predict = y_predict[:, 0]
    y_predict = y_predict.reshape(-1)
    if y_predict.shape[0] != x_domain.shape[0]:
        raise ValueError("x_domain and y_predict must have the same number of samples")

    if np.isscalar(k_values):
        k_max = int(k_values)
        if k_max < 0:
            raise ValueError("If k_values is scalar, it must be non-negative")
        k_values = np.arange(-k_max, k_max + 1, dtype=int)
    else:
        k_values = np.asarray(k_values, dtype=int).reshape(-1)
        if k_values.size == 0:
            raise ValueError("k_values can not be empty")

    if interval is None:
        a, b = float(np.min(x_domain)), float(np.max(x_domain))
    else:
        a, b = float(interval[0]), float(interval[1])
        if b <= a:
            raise ValueError("interval must satisfy b > a")

    order = np.argsort(x_domain)
    x_sorted, y_sorted = x_domain[order], y_predict[order]

    if np.any(np.diff(x_sorted) == 0.0):
        raise ValueError("x_domain has repeated points")

    length = b - a
    phase_argument = (x_sorted - a) / length
    exponent = np.exp(-1.0j * 2.0 * np.pi * np.outer(k_values, phase_argument))
    integrand = exponent * y_sorted[np.newaxis, :]

    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    c_k = trapz_fn(integrand, x_sorted, axis=1) / length
    return k_values, c_k


def fourier_series_from_coefficients(x_domain, k_values, c_k, interval=None):
    """Reconstruct f(x) from complex Fourier coefficients."""
    x_domain = np.asarray(x_domain).reshape(-1)
    k_values = np.asarray(k_values, dtype=int).reshape(-1)
    c_k = np.asarray(c_k, dtype=complex).reshape(-1)
    if k_values.shape[0] != c_k.shape[0]:
        raise ValueError("k_values and c_k must have the same length")

    if interval is None:
        a, b = float(np.min(x_domain)), float(np.max(x_domain))
    else:
        a, b = float(interval[0]), float(interval[1])
        if b <= a:
            raise ValueError("interval must satisfy b > a")

    length = b - a
    phase_argument = (x_domain - a) / length
    exponent = np.exp(1.0j * 2.0 * np.pi * np.outer(k_values, phase_argument))
    return np.dot(c_k, exponent)


def ak_bk_from_complex_coefficients(k_values, c_k, k_max=None):
    """
    Compute A_k^f and B_k^f from complex exponential Fourier coefficients c_k.

        A_k^f = c_k + c_-k
        B_k^f = -i * (c_k - c_-k)
    """
    k_values = np.asarray(k_values, dtype=int).reshape(-1)
    c_k = np.asarray(c_k, dtype=complex).reshape(-1)
    if k_values.shape[0] != c_k.shape[0]:
        raise ValueError("k_values and c_k must have the same length")
    if k_values.size == 0:
        raise ValueError("k_values can not be empty")

    coeff_map = {int(k): c for k, c in zip(k_values, c_k, strict=False)}
    k_available = int(np.max(np.abs(k_values)))
    k_max = k_available if k_max is None else int(k_max)
    if k_max < 0:
        raise ValueError("k_max must be non-negative")

    k_non_negative = np.arange(0, k_max + 1, dtype=int)
    a_k_f = np.zeros(k_non_negative.shape[0], dtype=complex)
    b_k_f = np.zeros(k_non_negative.shape[0], dtype=complex)

    for idx, k in enumerate(k_non_negative):
        if k not in coeff_map or -k not in coeff_map:
            raise ValueError(f"Missing c_k or c_-k for k={k}")
        c_pos, c_neg = coeff_map[k], coeff_map[-k]
        a_k_f[idx] = c_pos + c_neg
        b_k_f[idx] = -1.0j * (c_pos - c_neg)

    return k_non_negative, a_k_f, b_k_f


def fourier_price_v_t0(a, b, risk_free_rate, delta_t, a_k_f, b_k_f,
                       c_k_payoff, d_k_payoff):
    """
    Compute V(t0, x) from Fourier coefficients:

        V(t0, x) ~= 0.5*(b-a)*exp(-r*delta_t) *
                   (A_0^f*C_0/2 + sum_{k=1}^K (A_k^f*C_k + B_k^f*D_k))
    """
    a_k_f = np.asarray(a_k_f, dtype=complex).reshape(-1)
    b_k_f = np.asarray(b_k_f, dtype=complex).reshape(-1)
    c_k_payoff = np.asarray(c_k_payoff, dtype=complex).reshape(-1)
    d_k_payoff = np.asarray(d_k_payoff, dtype=complex).reshape(-1)

    if not (
        a_k_f.shape[0] == b_k_f.shape[0]
        == c_k_payoff.shape[0] == d_k_payoff.shape[0]
    ):
        raise ValueError("All coefficient arrays must have the same length")
    if a_k_f.size == 0:
        raise ValueError("Coefficient arrays can not be empty")
    if b <= a:
        raise ValueError("Interval must satisfy b > a")

    series_term = 0.5 * a_k_f[0] * c_k_payoff[0]
    if a_k_f.size > 1:
        series_term += np.sum(
            a_k_f[1:] * c_k_payoff[1:] + b_k_f[1:] * d_k_payoff[1:]
        )
    return 0.5 * (b - a) * np.exp(-risk_free_rate * delta_t) * series_term


# ---------------------------------------------------------------------------
# Loss functions (device-independent, unchanged)
# ---------------------------------------------------------------------------

def loss_function_pdf_and_derivative(
        labels_pdf,
        predict_pdf,
        predict_pdf_derivative,
        labels_pdf_derivative=None,
        integral_pdf_sq=0.0,
        loss_weights=(0.9, 0.1, 0.0)):
    """
    L = alpha_pdf * E_pdf + alpha_derivative * E_der + alpha_integral * I
    """
    if predict_pdf.shape != labels_pdf.shape:
        raise ValueError("predict_pdf and labels_pdf have different shape")

    pdf_error = np.mean((predict_pdf - labels_pdf) ** 2)

    if labels_pdf_derivative is None:
        derivative_error = np.mean(predict_pdf_derivative ** 2)
    else:
        if predict_pdf_derivative.shape != labels_pdf_derivative.shape:
            raise ValueError(
                "predict_pdf_derivative and labels_pdf_derivative have different shape"
            )
        derivative_error = np.mean(
            (predict_pdf_derivative - labels_pdf_derivative) ** 2
        )

    alpha_pdf, alpha_derivative, alpha_integral = loss_weights
    return (
        alpha_pdf * pdf_error
        + alpha_derivative * derivative_error
        + alpha_integral * float(integral_pdf_sq)
    )


def loss_function_qdml(labels, predict_cdf, predict_pdf, integral):
    """Legacy QDML loss (numpy, for reference)."""
    alpha_0 = 0
    alpha_1 = 0.5
    if predict_cdf.shape != labels.shape:
        raise ValueError("predict_cdf and labels have different shape!!")
    error_ = predict_cdf - labels
    loss_1 = np.mean(error_ ** 2)
    if predict_pdf.shape != labels.shape:
        raise ValueError("predict_pdf and labels have different shape!!")
    mean = -2 * np.mean(predict_pdf)
    return alpha_0 * loss_1 + alpha_1 * (mean + integral)
