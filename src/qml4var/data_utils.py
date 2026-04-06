"""
Functions for heliping to build the datasets
"""

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erf


def empirical_distribution_function_old(data_points: np.array):
    """
    Given an array of data points create the corresponding empirical
    distribution dunction
    Parameters
    ----------

    data_points : numpy array
        numpy array with data sampled

    Returns
    -------

    batch_ : QLM Batch
        QLM Batch with the jobs for computing graidents
    """
    n_sample = data_points.shape[0]
    distribution = np.zeros(n_sample)
    for m_ in range(n_sample):
        count = 0
        for n_ in list(range(m_)) + list(range(m_ + 1, n_sample)):
            check = np.all(data_points[m_] >= data_points[n_])
            if check:
                count = count + 1

        distribution[m_] = count / (n_sample - 1)
    return distribution


def empirical_cdf(data_points: np.ndarray):
    """
    Given an array of data points create the corresponding empirical
    distribution function
    Parameters
    ----------
    data_points : numpy array
        numpy array with data sampled
    Returns
    -------
    emp_cdf : numpy array
        numpy array with the empirical cdf of the input data
    """
    if len(data_points.shape) == 1:
        data_points = data_points.reshape((data_points.shape[0], 1))

    return np.array([np.sum(np.all(data_points <= x, axis=1)) for x in data_points]) / data_points.shape[0]


def bs_pdf(
    s_t: float, s_0: float = 1.0, risk_free_rate: float = 0.0, volatility: float = 0.5, maturity: float = 0.5, **kwargs
):
    """
    Black Scholes PDF
    """

    mean = (risk_free_rate - 0.5 * volatility * volatility) * maturity + np.log(s_0)
    factor = s_t * volatility * np.sqrt(2 * np.pi * maturity)
    exponent = -((np.log(s_t) - mean) ** 2) / (2 * volatility * volatility * maturity)
    return np.exp(exponent) / factor


def bs_cdf(
    s_t: float, s_0: float = 1.0, risk_free_rate: float = 0.0, volatility: float = 0.5, maturity: float = 0.5, **kwargs
):
    """
    Black Scholes PDF
    """
    mean = (risk_free_rate - 0.5 * volatility * volatility) * maturity + np.log(s_0)
    variance = volatility * volatility * maturity
    return 0.5 * (1 + erf((np.log(s_t) - mean) / (np.sqrt(2 * variance))))


def bs_samples(
    number_samples: int,
    s_0: float = 1.0,
    risk_free_rate: float = 0.0,
    volatility: float = 0.5,
    maturity: float = 0.5,
    **kwargs,
):
    """
    Black Scholes Samples
    """

    dW = np.random.randn(number_samples)
    return s_0 * np.exp(
        (risk_free_rate - 0.5 * volatility * volatility) * maturity + volatility * dW * np.sqrt(maturity)
    )


def saving_datasets(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, **kwargs: Any):
    """
    Saving Data sets
    """
    name_for_saving = kwargs.get("name_for_saving")
    if name_for_saving is not None:
        features = ["Features_{}".format(x_) for x_ in range(x_train.shape[1])]
        pdf_training = pd.DataFrame(x_train, columns=features)
        pdf_training["Labels"] = y_train
        pdf_testing = pd.DataFrame(x_test, columns=features)
        pdf_testing["Labels"] = y_test
        pdf_training.to_csv(name_for_saving + "_training.csv", sep=";", index=True)
        pdf_testing.to_csv(name_for_saving + "_testing.csv", sep=";", index=True)
        pathlib.Path(kwargs.get("folder_path") + "/data.json").write_text(json.dumps(kwargs))


def get_dataset(name_for_loading: str):
    # load Datasets
    pdf_training = pd.read_csv(name_for_loading + "_training.csv", sep=";", index_col=0)
    pdf_testing = pd.read_csv(name_for_loading + "_testing.csv", sep=";", index_col=0)
    feat = [col for col in pdf_training.columns if "Features" in col]
    x_train = pdf_training[feat].values
    y_train = pdf_training["Labels"].values
    y_train = y_train.reshape((-1, 1))
    x_test = pdf_testing[feat].values
    y_test = pdf_testing["Labels"].values
    y_test = y_test.reshape((-1, 1))
    return x_train, y_train, x_test, y_test


def simulate_black_scholes_data_rescaled(
    S0_: float,
    r_: float,
    T_: float,
    sigma_: float,
    K_: float,
    n_points: int,
    seed: int,
):
    """
    Simulate Black-Scholes log-moneyness samples and rescale to [-pi, pi].

    Parameters
    ----------
    S0_, r_, T_, sigma_ : Black-Scholes parameters
    K_                  : strike price
    n_points            : number of samples
    seed                : random seed

    Returns
    -------
    x_t       : raw log-moneyness array, shape (n_points, 1)
    u_t       : rescaled values in [-pi, pi], shape (n_points, 1)
    x_min_raw : minimum of x_t (used for inverse rescaling)
    x_max_raw : maximum of x_t (used for inverse rescaling)
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_points)
    s_t = S0_ * np.exp((r_ - 0.5 * sigma_**2) * T_ + sigma_ * np.sqrt(T_) * z)
    x_t = np.log(s_t / K_)

    x_min_raw = float(np.min(x_t))
    x_max_raw = float(np.max(x_t))
    if x_max_raw <= x_min_raw:
        x_max_raw = x_min_raw + 1.0e-8

    u_t = 2.0 * np.pi * (x_t - x_min_raw) / (x_max_raw - x_min_raw) - np.pi
    return x_t.reshape(-1, 1), u_t.reshape(-1, 1), x_min_raw, x_max_raw


def generate_method_I_labels(
    xs_rescaled: np.ndarray,
    mu: float,
    sigma: float,
    a: float,
    b: float,
) -> tuple:
    """
    Generate analytical PDF labels and their derivatives for Method I (paper Sec. 3.2.1).

    The PQC is trained in [-2π, 2π] with data rescaled to [-π, π]. This function
    computes the PDF and its derivative in the rescaled space given the corresponding
    Normal distribution parameters in the original log-moneyness space.

    Parameters
    ----------
    xs_rescaled : np.ndarray, shape (I,) or (I, 1)
        Training points in rescaled space [-2π, 2π].
    mu : float
        Mean of the Normal distribution in log-moneyness space.
        For Black-Scholes: mu = log(S0/K) + (r - σ²/2) * T
    sigma : float
        Std dev of the Normal distribution.
        For Black-Scholes: sigma = σ_BS * sqrt(T)
    a : float
        Left bound of the truncation interval in original log-moneyness space.
    b : float
        Right bound of the truncation interval.

    Returns
    -------
    pdf_vals : np.ndarray, shape (I,)
        f*(xs) in the rescaled space (accounts for change-of-variables Jacobian).
    pdf_deriv : np.ndarray, shape (I,)
        df*/du(xs) in the rescaled space.
    """
    from scipy.stats import norm

    xs = np.asarray(xs_rescaled).reshape(-1)

    # Inverse map: rescaled u ∈ [-2π, 2π] → original x ∈ [a, b]
    # u = 4π * (x - a) / (b - a) - 2π  →  x = (u + 2π)(b - a)/(4π) + a
    scale = (b - a) / (4.0 * np.pi)
    xs_original = xs * scale + (a + b) / 2.0

    # PDF and its derivative in original space
    pdf_original = norm.pdf(xs_original, loc=mu, scale=sigma)
    dpdf_original = -((xs_original - mu) / sigma ** 2) * pdf_original

    # Change of variables: f_rescaled(u) = f_original(x(u)) * |dx/du| = f_original * scale
    pdf_vals = pdf_original * scale
    # d(f_rescaled)/du = d(f_original)/dx * (dx/du)² = dpdf_original * scale²
    pdf_deriv = dpdf_original * scale ** 2

    return pdf_vals, pdf_deriv


def generate_method_I_data(
    S0: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    n_points: int,
) -> tuple:
    """
    Generate Method I real dataset with analytical bounds and PDF labels (paper Sec. 3.2.1).

    Unlike `simulate_black_scholes_data_rescaled`, this uses analytical bounds
    (mu ± 3*sigma_T) so they are fixed and reproducible regardless of seed.
    The training grid is a uniform grid in [-π, π] (not random samples).

    Parameters
    ----------
    S0, K, r, T, sigma : Black-Scholes parameters
    n_points            : number of grid points

    Returns
    -------
    grid          : np.ndarray, shape (n_points, 1), uniform grid in [-π, π]
    pdf_vals      : np.ndarray, shape (n_points,), PDF in rescaled space
    pdf_deriv     : np.ndarray, shape (n_points,), d(PDF)/du in rescaled space
    a             : float, left bound in log-moneyness space
    b             : float, right bound in log-moneyness space
    """
    # Analytical BS parameters for log(S_T/K) ~ N(mu, sigma_T^2)
    mu = np.log(S0 / K) + (r - 0.5 * sigma**2) * T
    sigma_T = sigma * np.sqrt(T)

    # Analytical bounds covering >99.7% of the distribution (paper Sec. 3.1.1)
    a = mu - 3.0 * sigma_T
    b = mu + 3.0 * sigma_T

    # Uniform grid in [-π, π] (training domain)
    grid = np.linspace(-np.pi, np.pi, n_points)

    # generate_method_I_labels expects xs_rescaled in [-2π, 2π].
    # Passing 2*grid (in [-2π, 2π]) ensures the inverse map xs_original ∈ [a, b].
    # The returned densities are in [-2π, 2π] space; we convert to [-π, π] space:
    #   f_{[-π,π]}(u) = f_{[-2π,2π]}(2u) * 2     (factor from d(2u)/du = 2)
    #   df_{[-π,π]}/du = df_{[-2π,2π]}/dv * 4     (factor from chain rule: 2² = 4)
    pdf_vals_2pi, pdf_deriv_2pi = generate_method_I_labels(2.0 * grid, mu, sigma_T, a, b)
    pdf_vals = pdf_vals_2pi * 2.0
    pdf_deriv = pdf_deriv_2pi * 4.0

    return grid.reshape(-1, 1), pdf_vals, pdf_deriv, a, b


def inverse_rescaling_u_to_xt(
    u_values,
    x_min_raw: float,
    x_max_raw: float,
):
    """
    Invert the [-pi, pi] rescaling back to raw log-moneyness.

    Parameters
    ----------
    u_values            : array of rescaled values in [-pi, pi]
    x_min_raw, x_max_raw: bounds returned by simulate_black_scholes_data_rescaled

    Returns
    -------
    numpy array of raw log-moneyness values
    """
    import numpy as np

    u_values = np.asarray(u_values)
    return x_min_raw + (u_values + np.pi) * (x_max_raw - x_min_raw) / (2.0 * np.pi)
