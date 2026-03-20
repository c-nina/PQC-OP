"""Utility routines for CrisTFM experiments."""

import numpy as np
from qat.core import Batch

from QQuantLib.qpu.select_qpu import select_qpu
from QQuantLib.qml4var.plugins import SetParametersPlugin, pdfPluging, MyQPU


def _build_base_job(**kwargs):
    """Create the base parametric QLM job from the provided PQC and observable."""
    pqc = kwargs.get("pqc")
    observable = kwargs.get("observable")
    nbshots = kwargs.get("nbshots")
    if pqc is None or observable is None:
        raise ValueError("pqc and observable are mandatory keyword arguments")
    circuit = pqc.to_circ()
    return circuit.to_job(nbshots=nbshots, observable=observable)


def _stack_execution(weights, x_sample, stack, **kwargs):
    """Execute a configured stack for one sample."""
    weights_names = kwargs.get("weights_names")
    features_names = kwargs.get("features_names")
    if weights_names is None or features_names is None:
        raise ValueError("weights_names and features_names are mandatory")

    batch = Batch(jobs=[_build_base_job(**kwargs)])
    batch.meta_data = {
        "weights": weights_names,
        "features": features_names,
    }
    return stack(weights, x_sample).submit(batch)


def distribution_workflow(weights, x_sample, differentiation_parameters=None, **kwargs):
    """
    Evaluate CDF or any derivative chain of the CDF.

    Parameters
    ----------
    weights : numpy array
        PQC weights.
    x_sample : numpy array
        Feature sample.
    differentiation_parameters : list or None
        If None computes CDF. If list, computes sequential derivatives
        with respect to those parameters. For example:
        * PDF in 1D: ["x_0"]
        * PDF in dD: features_names
        * d(PDF)/dx_0 in dD: features_names + ["x_0"]
    kwargs : dict
        Must contain the same keys used in qml4var workflows:
        pqc, observable, weights_names, features_names, nbshots, qpu_info.
    """
    qpu_dict = kwargs.get("qpu_info")
    if qpu_dict is None:
        raise ValueError("qpu_info is mandatory for workflow execution")
    qpu = select_qpu(qpu_dict)

    if differentiation_parameters is None:
        stack = lambda w_, x_: SetParametersPlugin(w_, x_) | MyQPU(qpu)
    else:
        stack = lambda w_, x_: (
            pdfPluging(differentiation_parameters) |
            SetParametersPlugin(w_, x_) |
            MyQPU(qpu)
        )

    results = _stack_execution(weights, x_sample, stack, **kwargs)
    return results[0].value


def cdf_workflow_cris(weights, x_sample, **kwargs):
    """Compute CDF value for one sample."""
    return distribution_workflow(
        weights, x_sample, differentiation_parameters=None, **kwargs
    )


def pdf_workflow_cris(weights, x_sample, **kwargs):
    """Compute PDF value for one sample."""
    features_names = kwargs.get("features_names")
    if features_names is None:
        raise ValueError("features_names is mandatory for PDF computation")
    return distribution_workflow(
        weights, x_sample, differentiation_parameters=list(features_names), **kwargs
    )


def pdf_derivative_workflow_cris(weights, x_sample, **kwargs):
    """
    Compute derivative of PDF for one sample.

    By default it computes d(PDF)/dx_i for all features sequentially.
    Use `pdf_derivative_features` in kwargs to choose specific variables.
    """
    features_names = kwargs.get("features_names")
    if features_names is None:
        raise ValueError("features_names is mandatory for PDF derivative computation")

    derivative_features = kwargs.get("pdf_derivative_features", features_names)
    if derivative_features is None:
        derivative_features = features_names
    if isinstance(derivative_features, str):
        derivative_features = [derivative_features]

    diff_parameters = list(features_names) + list(derivative_features)
    return distribution_workflow(
        weights, x_sample, differentiation_parameters=diff_parameters, **kwargs
    )


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
    Compute PDF and derivative(PDF) predictions for a full dataset.
    """
    workflow_cfg = {
        "pqc": kwargs.get("pqc"),
        "observable": kwargs.get("observable"),
        "weights_names": kwargs.get("weights_names"),
        "features_names": kwargs.get("features_names"),
        "nbshots": kwargs.get("nbshots"),
        "qpu_info": kwargs.get("qpu_info"),
    }
    # Preserve default behavior in pdf_derivative_workflow_cris when this key
    # is not provided: derivative over all features.
    pdf_derivative_features = kwargs.get("pdf_derivative_features")
    if pdf_derivative_features is not None:
        workflow_cfg["pdf_derivative_features"] = pdf_derivative_features

    pdf_workflow_ = lambda w, x: pdf_workflow_cris(w, x, **workflow_cfg)
    pdf_derivative_workflow_ = lambda w, x: pdf_derivative_workflow_cris(w, x, **workflow_cfg)

    predict_pdf = workflow_execution_cris(weights, data_x, pdf_workflow_, dask_client=dask_client)
    predict_pdf_derivative = workflow_execution_cris(
        weights, data_x, pdf_derivative_workflow_, dask_client=dask_client
    )

    if dask_client is None:
        predict_pdf = np.asarray(predict_pdf).reshape((-1, 1))
        predict_pdf_derivative = np.asarray(predict_pdf_derivative).reshape((-1, 1))
    else:
        predict_pdf = np.asarray(dask_client.gather(predict_pdf)).reshape((-1, 1))
        predict_pdf_derivative = np.asarray(
            dask_client.gather(predict_pdf_derivative)
        ).reshape((-1, 1))

    output = {
        "predict_pdf": predict_pdf,
        "predict_pdf_derivative": predict_pdf_derivative,
    }
    if labels_pdf is not None:
        output["labels_pdf"] = labels_pdf
    if labels_pdf_derivative is not None:
        output["labels_pdf_derivative"] = labels_pdf_derivative
    return output


def complex_fourier_coefficients(
        x_domain,
        y_predict,
        k_values,
        interval=None):
    """
    Compute complex Fourier coefficients c_k from sampled function values.

    Mathematical definition
    -----------------------
    For a function f(x) in a truncated interval [a, b], with L = b - a:

        c_k = (1 / L) * integral_a^b f(x) * exp(-i * 2*pi*k*(x-a)/L) dx

    The integral is approximated with the trapezoidal rule on `x_domain`.

    Parameters
    ----------
    x_domain : numpy.ndarray
        1D array with the grid points where f(x) is sampled.
    y_predict : numpy.ndarray
        Sampled values of f(x). It can be shape (N,) or (N, 1).
        Typical use in this project: predicted PDF after training.
    k_values : numpy.ndarray or list or int
        Fourier modes to compute.
        If int K is provided, modes are generated as [-K, ..., K].
    interval : tuple[float, float] or None, optional
        Interval (a, b). If None, uses (min(x_domain), max(x_domain)).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple `(k_values, c_k)` where:
        - `k_values` is an integer array of modes.
        - `c_k` is a complex array with the corresponding coefficients.
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
        a = float(np.min(x_domain))
        b = float(np.max(x_domain))
    else:
        a, b = float(interval[0]), float(interval[1])
        if b <= a:
            raise ValueError("interval must satisfy b > a")

    order = np.argsort(x_domain)
    x_sorted = x_domain[order]
    y_sorted = y_predict[order]

    if np.any(np.diff(x_sorted) == 0.0):
        raise ValueError("x_domain has repeated points; Fourier integration requires unique x")

    length = b - a
    phase_argument = (x_sorted - a) / length
    exponent = np.exp(-1.0j * 2.0 * np.pi * np.outer(k_values, phase_argument))
    integrand = exponent * y_sorted[np.newaxis, :]

    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    c_k = trapz_fn(integrand, x_sorted, axis=1) / length
    return k_values, c_k


def fourier_series_from_coefficients(x_domain, k_values, c_k, interval=None):
    """
    Reconstruct f(x) from complex Fourier coefficients.

    Parameters
    ----------
    x_domain : numpy.ndarray
        Points where the Fourier series is evaluated.
    k_values : numpy.ndarray
        Integer modes used to compute coefficients.
    c_k : numpy.ndarray
        Complex Fourier coefficients.
    interval : tuple[float, float] or None, optional
        Interval (a, b). If None, uses (min(x_domain), max(x_domain)).

    Returns
    -------
    numpy.ndarray
        Complex reconstructed values of f(x) on `x_domain`.
    """
    x_domain = np.asarray(x_domain).reshape(-1)
    k_values = np.asarray(k_values, dtype=int).reshape(-1)
    c_k = np.asarray(c_k, dtype=complex).reshape(-1)
    if k_values.shape[0] != c_k.shape[0]:
        raise ValueError("k_values and c_k must have the same length")

    if interval is None:
        a = float(np.min(x_domain))
        b = float(np.max(x_domain))
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

    Identities used:

        A_k^f = c_k + c_-k
        B_k^f = -i * (c_k - c_-k)

    Parameters
    ----------
    k_values : numpy.ndarray
        Integer modes associated with `c_k`.
    c_k : numpy.ndarray
        Complex coefficients c_k.
    k_max : int or None, optional
        Maximum mode K for outputs. If None, uses max(abs(k_values)).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        `(k_non_negative, a_k_f, b_k_f)` where:
        - `k_non_negative` are modes [0, ..., K],
        - `a_k_f` are A_k^f values for those modes,
        - `b_k_f` are B_k^f values for those modes.
    """
    k_values = np.asarray(k_values, dtype=int).reshape(-1)
    c_k = np.asarray(c_k, dtype=complex).reshape(-1)
    if k_values.shape[0] != c_k.shape[0]:
        raise ValueError("k_values and c_k must have the same length")
    if k_values.size == 0:
        raise ValueError("k_values can not be empty")

    coeff_map = {int(k): c for k, c in zip(k_values, c_k)}
    k_available = int(np.max(np.abs(k_values)))
    if k_max is None:
        k_max = k_available
    k_max = int(k_max)
    if k_max < 0:
        raise ValueError("k_max must be non-negative")

    k_non_negative = np.arange(0, k_max + 1, dtype=int)
    a_k_f = np.zeros(k_non_negative.shape[0], dtype=complex)
    b_k_f = np.zeros(k_non_negative.shape[0], dtype=complex)

    for idx, k in enumerate(k_non_negative):
        if k not in coeff_map or -k not in coeff_map:
            raise ValueError(f"Missing c_k or c_-k for k={k}")
        c_pos = coeff_map[k]
        c_neg = coeff_map[-k]
        a_k_f[idx] = c_pos + c_neg
        b_k_f[idx] = -1.0j * (c_pos - c_neg)

    return k_non_negative, a_k_f, b_k_f


def fourier_price_v_t0(
        a,
        b,
        risk_free_rate,
        delta_t,
        a_k_f,
        b_k_f,
        c_k_payoff,
        d_k_payoff):
    """
    Compute V(t0, x) from Fourier coefficients using:

        V(t0, x) ~= 0.5 * (b - a) * exp(-r * delta_t) *
                   (A_0^f * C_0 / 2 + sum_{k=1}^K (A_k^f*C_k + B_k^f*D_k))

    Parameters
    ----------
    a : float
        Lower bound of truncation interval.
    b : float
        Upper bound of truncation interval.
    risk_free_rate : float
        Risk-free rate r.
    delta_t : float
        Time step in discount factor.
    a_k_f : numpy.ndarray
        Array with A_k^f for k=0..K.
    b_k_f : numpy.ndarray
        Array with B_k^f for k=0..K.
    c_k_payoff : numpy.ndarray
        Array with C_k for k=0..K.
    d_k_payoff : numpy.ndarray
        Array with D_k for k=0..K.

    Returns
    -------
    complex
        Estimated value V(t0, x). For real-valued consistent inputs, the
        imaginary part should be numerically close to zero.
    """
    a_k_f = np.asarray(a_k_f, dtype=complex).reshape(-1)
    b_k_f = np.asarray(b_k_f, dtype=complex).reshape(-1)
    c_k_payoff = np.asarray(c_k_payoff, dtype=complex).reshape(-1)
    d_k_payoff = np.asarray(d_k_payoff, dtype=complex).reshape(-1)

    if not (
        a_k_f.shape[0] == b_k_f.shape[0] ==
        c_k_payoff.shape[0] == d_k_payoff.shape[0]
    ):
        raise ValueError("a_k_f, b_k_f, c_k_payoff and d_k_payoff must have the same length")
    if a_k_f.size == 0:
        raise ValueError("Coefficient arrays can not be empty")
    if b <= a:
        raise ValueError("Interval must satisfy b > a")

    series_term = 0.5 * a_k_f[0] * c_k_payoff[0]
    if a_k_f.size > 1:
        series_term += np.sum(
            a_k_f[1:] * c_k_payoff[1:] + b_k_f[1:] * d_k_payoff[1:]
        )

    prefactor = 0.5 * (b - a) * np.exp(-risk_free_rate * delta_t)
    return prefactor * series_term


def loss_function_pdf_and_derivative(
        labels_pdf,
        predict_pdf,
        predict_pdf_derivative,
        labels_pdf_derivative=None,
        integral_pdf_sq=0.0,
        loss_weights=(0.9, 0.1, 0.0)):
    """
    Compute a loss using PDF prediction and derivative(PDF) prediction.

    Mathematical definition
    -----------------------
    Let N be the number of samples, y_i the PDF target, y_hat_i the predicted
    PDF, d_i the derivative(PDF) target and d_hat_i the predicted derivative.

    The implemented loss is:

        L = alpha_pdf * E_pdf + alpha_derivative * E_der + alpha_integral * I

    where:

        E_pdf = (1/N) * sum_i (y_hat_i - y_i)^2

    and:

        E_der = (1/N) * sum_i (d_hat_i - d_i)^2
        if labels_pdf_derivative is provided,

        E_der = (1/N) * sum_i (d_hat_i)^2
        if labels_pdf_derivative is None (smoothness regularization).

    I is `integral_pdf_sq`.

    Parameters
    ----------
    labels_pdf : numpy.ndarray
        Target PDF values with shape compatible with `predict_pdf`.
    predict_pdf : numpy.ndarray
        Predicted PDF values (this is the model output y_predict for PDF).
    predict_pdf_derivative : numpy.ndarray
        Predicted derivative(PDF) values.
    labels_pdf_derivative : numpy.ndarray or None, optional
        Target derivative(PDF) values. If None, derivative term becomes
        `mean(predict_pdf_derivative**2)`.
    integral_pdf_sq : float, optional
        Extra scalar regularization term (e.g., integral of PDF^2 on a domain).
    loss_weights : tuple[float, float, float], optional
        Weights `(alpha_pdf, alpha_derivative, alpha_integral)`.

    Returns
    -------
    float
        Scalar value of the loss.
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
        alpha_pdf * pdf_error +
        alpha_derivative * derivative_error +
        alpha_integral * integral_pdf_sq
    )


def loss_function_qdml(labels, predict_cdf, predict_pdf, integral):
    """
    Legacy QDML loss based on CDF and PDF terms.

    Parameters
    ----------
    labels : numpy.ndarray
        CDF labels.
    predict_cdf : numpy.ndarray
        Predicted CDF values.
    predict_pdf : numpy.ndarray
        Predicted PDF values.
    integral : float
        Integral regularization term (typically integral of PDF^2).

    Returns
    -------
    float
        Scalar loss value.
    """
    alpha_0 = 0
    alpha_1 = 0.5
    if predict_cdf.shape != labels.shape:
        raise ValueError("predict_cdf and labels have different shape!!")
    error_ = predict_cdf - labels
    loss_1 = np.mean(error_ ** 2)
    if predict_pdf.shape != labels.shape:
        raise ValueError("predict_pdf and labels have different shape!!")
    mean = -2 * np.mean(predict_pdf)
    loss_ = alpha_0 * loss_1 + alpha_1 * (mean + integral)
    return loss_
