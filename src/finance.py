"""Utility routines for CrisTFM experiments (PennyLane backend)."""

from typing import Any, Callable, Optional, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _weights_to_tensor(weights: Union[list, dict, torch.Tensor], device: str):
    if isinstance(weights, torch.Tensor):
        return weights
    values = list(weights.values()) if isinstance(weights, dict) else list(weights)
    return torch.tensor(values, dtype=torch.float64, device=torch.device(device))


# ---------------------------------------------------------------------------
# Single-sample workflow functions (PennyLane replacements)
# ---------------------------------------------------------------------------


def cdf_workflow_cris(weights: Union[list, dict, torch.Tensor], x_sample: np.ndarray, **kwargs: Any):
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


def pdf_workflow_cris(weights: Union[list, dict, torch.Tensor], x_sample: np.ndarray, **kwargs: Any):
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
    x_t = torch.tensor(x_flat, dtype=torch.float64, device=torch.device(device), requires_grad=True)
    cdf = circuit_fn(w_t, x_t)
    cdf.backward()
    return x_t.grad.sum().item()


def pdf_derivative_workflow_cris(weights: Union[list, dict, torch.Tensor], x_sample: np.ndarray, **kwargs: Any):
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
    x_t = torch.tensor(x_flat, dtype=torch.float64, device=torch.device(device), requires_grad=True)
    cdf = circuit_fn(w_t, x_t)
    pdf = torch.autograd.grad(cdf, x_t, create_graph=True)[0]
    pdf_deriv = torch.autograd.grad(pdf.sum(), x_t)[0]
    return pdf_deriv.sum().item()


def workflow_execution_cris(
    weights: Union[list, dict, torch.Tensor], data_x: np.ndarray, workflow: Callable, dask_client: Optional[Any] = None
):
    """Execute one workflow for all input samples."""
    if dask_client is None:
        return [workflow(weights, x_) for x_ in data_x]
    return dask_client.map(workflow, *([weights] * data_x.shape[0], data_x))


def workflow_for_pdf_and_derivative_cris(
    weights: Union[list, dict, torch.Tensor],
    data_x: np.ndarray,
    labels_pdf: Optional[np.ndarray] = None,
    labels_pdf_derivative: Optional[np.ndarray] = None,
    dask_client: Optional[Any] = None,
    **kwargs: Any,
):
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

    predict_pdf = workflow_execution_cris(weights, data_x, pdf_fn, dask_client=dask_client)
    predict_pdf_derivative = workflow_execution_cris(weights, data_x, pdf_deriv_fn, dask_client=dask_client)

    if dask_client is None:
        predict_pdf = np.asarray(predict_pdf).reshape(-1, 1)
        predict_pdf_derivative = np.asarray(predict_pdf_derivative).reshape(-1, 1)
    else:
        predict_pdf = np.asarray(dask_client.gather(predict_pdf)).reshape(-1, 1)
        predict_pdf_derivative = np.asarray(dask_client.gather(predict_pdf_derivative)).reshape(-1, 1)

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


def complex_fourier_coefficients(
    x_domain: np.ndarray, y_predict: np.ndarray, k_values: Union[int, np.ndarray], interval: Optional[tuple] = None
):
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


def fourier_series_from_coefficients(
    x_domain: np.ndarray, k_values: np.ndarray, c_k: np.ndarray, interval: Optional[tuple] = None
):
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


def ak_bk_from_complex_coefficients(k_values: np.ndarray, c_k: np.ndarray, k_max: Optional[int] = None):
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


def payoff_derivative_fourier_coefficients(
    x_domain: np.ndarray,
    h_prime: np.ndarray,
    k_max: int,
    a_ext: float,
    L_ext: float,
) -> tuple:
    """
    Fourier coefficients of h'(x) on a sub-interval using extended period L_ext.

    Used for the anti-Gibbs treatment in Method II (CDF-based) pricing.
    Both C_k and D_k use the same reference point a_ext and period L_ext,
    which must match those used for the CDF coefficients A_k^F, B_k^F.

        C_k = (2 / L_ext) * integral h'(x) * cos(2*pi*k*(x - a_ext) / L_ext) dx
        D_k = (2 / L_ext) * integral h'(x) * sin(2*pi*k*(x - a_ext) / L_ext) dx

    Parameters
    ----------
    x_domain : array, grid points inside the sub-interval [x1, x2] ⊆ [a, b]
    h_prime  : array, values of h'(x) at x_domain
    k_max    : int, highest Fourier harmonic (coefficients 0 … k_max)
    a_ext    : float, left boundary of extended interval = (3a - b) / 2
    L_ext    : float, extended period = 2*(b - a)

    Returns
    -------
    C_k, D_k : arrays of shape (k_max + 1,)
    """
    x_domain = np.asarray(x_domain, dtype=float).reshape(-1)
    h_prime = np.asarray(h_prime, dtype=float).reshape(-1)
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    k_arr = np.arange(k_max + 1, dtype=float)
    C_k = np.zeros(k_max + 1)
    D_k = np.zeros(k_max + 1)

    if x_domain.size < 2:
        return C_k, D_k

    # phase[k, i] = 2*pi*k*(x_i - a_ext) / L_ext
    phase = 2.0 * np.pi * np.outer(k_arr, (x_domain - a_ext) / L_ext)  # (K+1, N)

    C_k = (2.0 / L_ext) * trapz_fn(h_prime[np.newaxis, :] * np.cos(phase), x_domain, axis=1)
    D_k = (2.0 / L_ext) * trapz_fn(h_prime[np.newaxis, :] * np.sin(phase), x_domain, axis=1)
    D_k[0] = 0.0  # by definition (sin(0) = 0)

    return C_k, D_k


def fourier_price_v_t0_ibp(
    a: float,
    b: float,
    risk_free_rate: float,
    delta_t: float,
    a_k_F: np.ndarray,
    b_k_F: np.ndarray,
    c_k_a: np.ndarray,
    d_k_a: np.ndarray,
    c_k_b: np.ndarray,
    d_k_b: np.ndarray,
    h_at_a: float,
    F_at_a: float,
    h_at_b: float,
    F_at_b: float,
) -> float:
    """
    Option price via integration by parts with payoff-derivative split at c.

    Derivation
    ----------
    V = e^{-rT} * integral_a^b h(x) f(x) dx
      = e^{-rT} * [h(b)F(b) - h(a)F(a) - integral_a^b h'(x) F(x) dx]

    The last integral is split at c (where h' is discontinuous) and each
    piece is evaluated via Parseval's theorem using Fourier series with
    extended period L_ext = 2*(b-a):

      integral_a^c h'(x)F(x)dx = (L_ext/2) * [A0^F*C0^a/2 + sum(Ak^F*Ck^a + Bk^F*Dk^a)]
      integral_c^b h'(x)F(x)dx = (L_ext/2) * [A0^F*C0^b/2 + sum(Ak^F*Ck^b + Bk^F*Dk^b)]

    where L_ext/2 = b - a.

    Parameters
    ----------
    a, b             : float, training domain bounds
    risk_free_rate   : float
    delta_t          : float, time to maturity
    a_k_F, b_k_F     : CDF Fourier coefficients (computed with period L_ext)
    c_k_a, d_k_a     : h' Fourier coefficients on [a, c] (period L_ext)
    c_k_b, d_k_b     : h' Fourier coefficients on [c, b] (period L_ext)
    h_at_a, F_at_a   : payoff and CDF at left boundary
    h_at_b, F_at_b   : payoff and CDF at right boundary

    Returns
    -------
    float : estimated option price V(t0)
    """
    a_k_F = np.asarray(a_k_F, dtype=complex).reshape(-1)
    b_k_F = np.asarray(b_k_F, dtype=complex).reshape(-1)
    c_k_a = np.asarray(c_k_a, dtype=float).reshape(-1)
    d_k_a = np.asarray(d_k_a, dtype=float).reshape(-1)
    c_k_b = np.asarray(c_k_b, dtype=float).reshape(-1)
    d_k_b = np.asarray(d_k_b, dtype=float).reshape(-1)

    if not (len(a_k_F) == len(b_k_F) == len(c_k_a) == len(d_k_a) == len(c_k_b) == len(d_k_b)):
        raise ValueError("All coefficient arrays must have the same length")

    boundary = float(h_at_b) * float(F_at_b) - float(h_at_a) * float(F_at_a)

    # L_ext / 2 = b - a  (the Parseval prefactor)
    L_ext_half = float(b) - float(a)

    series_a = 0.5 * a_k_F[0] * c_k_a[0]
    series_b = 0.5 * a_k_F[0] * c_k_b[0]
    if len(a_k_F) > 1:
        series_a += np.sum(a_k_F[1:] * c_k_a[1:] + b_k_F[1:] * d_k_a[1:])
        series_b += np.sum(a_k_F[1:] * c_k_b[1:] + b_k_F[1:] * d_k_b[1:])

    discount = np.exp(-float(risk_free_rate) * float(delta_t))
    return float(np.real(discount * (boundary - L_ext_half * (series_a + series_b))))


def fourier_price_v_t0(
    a: float,
    b: float,
    risk_free_rate: float,
    delta_t: float,
    a_k_f: np.ndarray,
    b_k_f: np.ndarray,
    c_k_payoff: np.ndarray,
    d_k_payoff: np.ndarray,
):
    """
    Compute V(t0, x) from Fourier coefficients:

        V(t0, x) ~= 0.5*(b-a)*exp(-r*delta_t) *
                   (A_0^f*C_0/2 + sum_{k=1}^K (A_k^f*C_k + B_k^f*D_k))
    """
    a_k_f = np.asarray(a_k_f, dtype=complex).reshape(-1)
    b_k_f = np.asarray(b_k_f, dtype=complex).reshape(-1)
    c_k_payoff = np.asarray(c_k_payoff, dtype=complex).reshape(-1)
    d_k_payoff = np.asarray(d_k_payoff, dtype=complex).reshape(-1)

    if not (a_k_f.shape[0] == b_k_f.shape[0] == c_k_payoff.shape[0] == d_k_payoff.shape[0]):
        raise ValueError("All coefficient arrays must have the same length")
    if a_k_f.size == 0:
        raise ValueError("Coefficient arrays can not be empty")
    if b <= a:
        raise ValueError("Interval must satisfy b > a")

    series_term = 0.5 * a_k_f[0] * c_k_payoff[0]
    if a_k_f.size > 1:
        series_term += np.sum(a_k_f[1:] * c_k_payoff[1:] + b_k_f[1:] * d_k_payoff[1:])
    return 0.5 * (b - a) * np.exp(-risk_free_rate * delta_t) * series_term


# ---------------------------------------------------------------------------
# Loss functions (device-independent, unchanged)
# ---------------------------------------------------------------------------


def loss_function_pdf_and_derivative(
    labels_pdf: np.ndarray,
    predict_pdf: np.ndarray,
    predict_pdf_derivative: np.ndarray,
    labels_pdf_derivative: Optional[np.ndarray] = None,
    integral_pdf_sq: float = 0.0,
    loss_weights: tuple = (0.9, 0.1, 0.0),
):
    """
    L = alpha_pdf * E_pdf + alpha_derivative * E_der + alpha_integral * I
    """
    if predict_pdf.shape != labels_pdf.shape:
        raise ValueError("predict_pdf and labels_pdf have different shape")

    pdf_error = np.mean((predict_pdf - labels_pdf) ** 2)

    if labels_pdf_derivative is None:
        derivative_error = np.mean(predict_pdf_derivative**2)
    else:
        if predict_pdf_derivative.shape != labels_pdf_derivative.shape:
            raise ValueError("predict_pdf_derivative and labels_pdf_derivative have different shape")
        derivative_error = np.mean((predict_pdf_derivative - labels_pdf_derivative) ** 2)

    alpha_pdf, alpha_derivative, alpha_integral = loss_weights
    return alpha_pdf * pdf_error + alpha_derivative * derivative_error + alpha_integral * float(integral_pdf_sq)


def loss_function_qdml(labels: np.ndarray, predict_cdf: np.ndarray, predict_pdf: np.ndarray, integral: float):
    """Legacy QDML loss (numpy, for reference)."""
    alpha_0 = 0
    alpha_1 = 0.5
    if predict_cdf.shape != labels.shape:
        raise ValueError("predict_cdf and labels have different shape!!")
    error_ = predict_cdf - labels
    loss_1 = np.mean(error_**2)
    if predict_pdf.shape != labels.shape:
        raise ValueError("predict_pdf and labels have different shape!!")
    mean = -2 * np.mean(predict_pdf)
    return alpha_0 * loss_1 + alpha_1 * (mean + integral)
