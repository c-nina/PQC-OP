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
        b_k_f[idx] = 1.0j * (c_pos - c_neg)

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
# Black-Scholes analytical pricing
# ---------------------------------------------------------------------------


def bs_put_price(
    S0_: float,
    K_: float,
    r_: float,
    sigma_: float,
    T_: float,
) -> float:
    """
    Black-Scholes closed-form price for a European put option.

    Parameters
    ----------
    S0_   : initial stock price
    K_    : strike price
    r_    : risk-free rate
    sigma_: volatility
    T_    : time to maturity

    Returns
    -------
    float : put option price
    """
    from scipy.stats import norm

    d1 = (np.log(S0_ / K_) + (r_ + 0.5 * sigma_**2) * T_) / (sigma_ * np.sqrt(T_))
    d2 = d1 - sigma_ * np.sqrt(T_)
    return K_ * np.exp(-r_ * T_) * norm.cdf(-d2) - S0_ * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Post-training inference: estimate option price from a trained PQC
# ---------------------------------------------------------------------------


def estimate_price_from_trained_pqc(
    weights,
    artifacts: dict,
    K_: float,
    x_min_raw: float,
    x_max_raw: float,
    train_interval: tuple,
    risk_free_rate: float,
    delta_t: float,
    k_terms: int = 12,
    grid_points: int = 1024,
    dask_client=None,
    debug: bool = False,
    debug_label: str = "",
    eval_interval: tuple = None,
) -> float:
    """
    Method I — PDF-based Fourier pricing.

    Evaluates the trained PQC as a PDF on a dense grid, normalises it,
    computes Fourier coefficients, and recovers the put option price via
    the Fourier pricing formula (fourier_price_v_t0).

    Parameters
    ----------
    weights        : trained circuit weights (list, dict, or torch.Tensor)
    artifacts      : dict returned by build_mode_artifacts (contains workflow_cfg)
    K_             : strike price
    x_min_raw      : minimum log-moneyness boundary (left endpoint of physical domain)
    x_max_raw      : maximum log-moneyness boundary (right endpoint of physical domain)
    train_interval : (a, b) domain where training data lived, e.g. (-pi, pi)
    risk_free_rate : annualised risk-free rate
    delta_t        : time to maturity
    k_terms        : number of Fourier harmonics (default 12)
    grid_points    : number of evaluation points on the grid (default 1024)
    dask_client    : optional Dask client for distributed evaluation
    debug          : if True, print intermediate diagnostics
    debug_label    : label string for debug messages
    eval_interval  : (a, b) domain for circuit evaluation and Fourier extraction.
                     Defaults to train_interval.  For Method I (base_frecuency=0.5)
                     this should be (-2π, 2π) so the full circuit domain is used
                     and Gibbs oscillations at the data boundary are suppressed
                     (paper Sec. 3.2, Figs 2-3).

    Returns
    -------
    float : estimated put option price, or np.nan on failure
    """
    from qml4var.workflows import workflow_for_pdf_direct

    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    workflow_cfg = artifacts["workflow_cfg"]

    # eval_interval is the circuit domain used for Fourier extraction.
    # For Method I (base_frecuency=0.5): [-2π, 2π]; for Method II: [-π, π].
    a, b = eval_interval if eval_interval is not None else train_interval
    u_grid = np.linspace(a, b, grid_points).reshape(-1, 1)

    pdf_raw = workflow_for_pdf_direct(weights, u_grid, dask_client=dask_client, **workflow_cfg)["y_predict_pdf"].reshape(-1)
    pdf_pred = np.nan_to_num(pdf_raw, nan=0.0, posinf=0.0, neginf=0.0)
    pdf_pred = np.clip(pdf_pred, 0.0, None)

    area = trapz_fn(pdf_pred, u_grid[:, 0])

    if debug:
        print(
            f"[PRICE DEBUG] {debug_label} "
            f"pdf_min={np.min(pdf_pred):.6e} pdf_max={np.max(pdf_pred):.6e} area={area:.6e}"
        )

    if (not np.isfinite(area)) or np.isclose(area, 0.0):
        print(
            f"[WARN price] {debug_label} invalid area. "
            f"pdf_min={np.min(pdf_pred):.6e} pdf_max={np.max(pdf_pred):.6e} area={area}"
        )
        return np.nan

    pdf_pred = pdf_pred / area

    if np.allclose(pdf_pred, 0.0):
        print(f"[WARN price] {debug_label} normalized pdf collapsed to zeros")
        return np.nan

    k_vals, c_k = complex_fourier_coefficients(
        x_domain=u_grid[:, 0],
        y_predict=pdf_pred,
        k_values=k_terms,
        interval=(a, b),
    )
    _, A_k_f, B_k_f = ak_bk_from_complex_coefficients(k_vals, c_k, k_max=k_terms)

    # Generalised inverse mapping: u ∈ [a, b] (eval domain) → x ∈ [x_min_raw, x_max_raw].
    # For Method I: [a,b] = [-2π, 2π], [x_min, x_max] = analytical BS bounds [a_bs, b_bs].
    # For Method II: [a,b] = [-π, π],  [x_min, x_max] = empirical data range.
    u_min, u_max = float(a), float(b)
    x_raw_grid = x_min_raw + (u_grid[:, 0] - u_min) * (x_max_raw - x_min_raw) / (u_max - u_min)
    payoff = np.maximum(K_ * (1.0 - np.exp(x_raw_grid)), 0.0)

    L = b - a
    z = (u_grid[:, 0] - a) / L
    C_k = np.zeros(k_terms + 1, dtype=complex)
    D_k = np.zeros(k_terms + 1, dtype=complex)
    for k in range(k_terms + 1):
        angle = 2.0 * np.pi * k * z
        C_k[k] = (2.0 / L) * trapz_fn(payoff * np.cos(angle), u_grid[:, 0])
        D_k[k] = 0.0 if k == 0 else (2.0 / L) * trapz_fn(payoff * np.sin(angle), u_grid[:, 0])

    v_t0 = fourier_price_v_t0(
        a=a,
        b=b,
        risk_free_rate=risk_free_rate,
        delta_t=delta_t,
        a_k_f=A_k_f,
        b_k_f=B_k_f,
        c_k_payoff=C_k,
        d_k_payoff=D_k,
    )
    return float(np.real(v_t0))


def estimate_price_ibp(
    weights,
    artifacts: dict,
    K_: float,
    x_min_raw: float,
    x_max_raw: float,
    train_interval: tuple,
    risk_free_rate: float,
    delta_t: float,
    k_terms: int = 12,
    grid_points: int = 1024,
    dask_client=None,
    debug: bool = False,
    debug_label: str = "",
) -> float:
    """
    Method II — Integration-by-parts (IBP) pricing using CDF Fourier coefficients.

    Avoids differentiating the learned PDF; instead uses the CDF directly and
    applies the IBP identity:
        V = e^{-rT} [h(b)F(b) - h(a)F(a) - integral_a^b h'(x) F(x) dx]

    The CDF is extended to a doubled interval to suppress Gibbs oscillations,
    and the payoff derivative integral is split at the exercise boundary c
    (where log-moneyness = 0).

    Parameters
    ----------
    weights        : trained circuit weights
    artifacts      : dict returned by build_mode_artifacts
    K_             : strike price
    x_min_raw      : minimum log-moneyness from training data
    x_max_raw      : maximum log-moneyness from training data
    train_interval : (a, b) domain used during training, e.g. (-2pi, 2pi)
    risk_free_rate : annualised risk-free rate
    delta_t        : time to maturity
    k_terms        : number of Fourier harmonics (default 12)
    grid_points    : number of evaluation points on the grid (default 1024)
    dask_client    : optional Dask client for distributed evaluation
    debug          : if True, print intermediate diagnostics
    debug_label    : label string for debug messages

    Returns
    -------
    float : estimated put option price
    """
    from qml4var.data_utils import inverse_rescaling_u_to_xt
    from qml4var.workflows import workflow_for_cdf

    workflow_cfg = artifacts["workflow_cfg"]
    a, b = train_interval

    u_inner = np.linspace(a, b, grid_points).reshape(-1, 1)
    u_flat = u_inner[:, 0]

    cdf_raw = workflow_for_cdf(weights, u_inner, dask_client=dask_client, **workflow_cfg)["y_predict_cdf"].reshape(-1)
    cdf_inner = np.clip(cdf_raw, 0.0, 1.0)  # workflow_for_cdf already maps to [0,1]

    # Enforce theoretical boundary conditions F(a)=0, F(b)=1.
    cdf_inner[0] = 0.0
    cdf_inner[-1] = 1.0

    F_at_a = float(cdf_inner[0])
    F_at_b = float(cdf_inner[-1])

    if debug:
        print(
            f"[IBP DEBUG] {debug_label} F(a)={F_at_a:.4f} F(b)={F_at_b:.4f}"
            f" cdf_min={cdf_inner.min():.4f} cdf_max={cdf_inner.max():.4f}"
        )

    x_raw_a = inverse_rescaling_u_to_xt(a, x_min_raw, x_max_raw)
    x_raw_b = inverse_rescaling_u_to_xt(b, x_min_raw, x_max_raw)
    h_at_a = float(np.maximum(K_ * (1.0 - np.exp(x_raw_a)), 0.0))
    h_at_b = float(np.maximum(K_ * (1.0 - np.exp(x_raw_b)), 0.0))

    # Locate discontinuity c of h'(x) = d/dx max(K*(1-exp(x)),0) in u-space.
    # h'(x) = -K*exp(x) for x < 0,  h'(x) = 0 for x > 0.
    # Three cases depending on whether x=0 falls inside the empirical domain.
    if x_min_raw >= 0.0:
        # All log-moneyness >= 0: put is always OTM, payoff=0 everywhere → V=0.
        return 0.0
    # Normal case: x=0 strictly inside [x_min_raw, x_max_raw]; edge: all ITM → u_c = b+1.
    u_c = (
        b + 1.0
        if x_max_raw <= 0.0
        else 2.0 * np.pi * (0.0 - x_min_raw) / (x_max_raw - x_min_raw) - np.pi
    )

    x_raw_grid = inverse_rescaling_u_to_xt(u_flat, x_min_raw, x_max_raw)
    dx_du = (x_max_raw - x_min_raw) / (2.0 * np.pi)
    h_prime = np.where(u_flat < u_c, -K_ * np.exp(x_raw_grid) * dx_du, 0.0)

    # IBP: V = e^{-rT} * [h(b)·F(b) - h(a)·F(a) - ∫_a^b h'(u)·F(u) du]
    #
    # Previous implementation used the Parseval sum with k_terms=12 Fourier harmonics
    # to evaluate the integral.  This caused a systematic upward bias that grew with
    # architecture size:
    #   - The circuit with base_frecuency=0.5 can represent frequencies up to
    #     n_layers × n_qubits × 0.5 (≈8 for 4×4, ≈12.5 for 5×5, ≈18 for 6×6).
    #   - Larger circuits encode real CDF energy into harmonics above k=12.
    #   - The k_terms=12 truncation missed these terms, making the Parseval sum
    #     smaller than the true integral and inflating the price.
    #
    # Fix: direct trapezoidal integration on the 1024-point grid, splitting naturally
    # at u_c via h_prime=0 for u≥u_c.  The integrand h'(u)·F(u) is smooth on each
    # piece ([a, u_c] and [u_c, b]), so the trapezoidal rule gives O(h²) accuracy
    # with h=2π/1024, far better than k=12 Parseval and fully architecture-independent.
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    integral_hprime_F = trapz_fn(h_prime * cdf_inner, u_flat)
    boundary = h_at_b * F_at_b - h_at_a * F_at_a
    discount = np.exp(-float(risk_free_rate) * float(delta_t))
    return float(discount * (boundary - integral_hprime_F))
