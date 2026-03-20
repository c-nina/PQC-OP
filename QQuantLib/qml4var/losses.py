"""
Functions for computing losses
"""

import copy
import os
from time import perf_counter
import numpy as np


def _env_flag(name, default=False):
    """
    Parse boolean-like environment variables.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _trapz_compat(y, x):
    """
    Compatibility wrapper for NumPy trapezoidal integration.
    NumPy >=2 may expose `trapezoid` while older versions use `trapz`.
    """
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trapz_fn(y=y, x=x)


def compute_integral(y_array, x_array, dask_client=None):
    """
    Function for computing numerical integral of inputs arrays.

    Parameters
    ----------
    y_array : np.array or dask futures:
        array or futures (only is dask_client is provided) with the
        y-values for integration.  For array expected shape is: shape(n)
    x_array : np array
        array with the x domain for integration: Shape(n, n_features)
    dask_client : DASK client
        DASK client to submit computation of the integral.
        y_array MUST BE a list of futures

    Returns
    -------
    integral : float or future
        float or future (if dask_client is provided) with the desired
        integral computation.

    Note
    ----
    Considerations:
        * if x_array has shape(n, 1): then numpy trapz is used for computing integral.
        * if x_array has shape(n, 2): and dask_client is None numpy trapz is used for computing the double integral.
        * if x_array has shape(n, 2): and dask_client is provided then MonteCarlo integration is used.
        * if x_array has shape(n, > 2): MonteCarlo integration is used
    """
    if x_array.shape[1] == 1:
        if dask_client is None:
            integral = _trapz_compat(y=y_array, x=x_array[:, 0])
        else:
            integral = dask_client.submit(_trapz_compat, y_array, x_array[:, 0])
    elif x_array.shape[1] == 2:
        if dask_client is None:
            x_domain, y_domain = np.meshgrid(np.unique(x_array[:, 0]), np.unique(x_array[:, 1]))
            y_array_ = y_array.reshape(x_domain.shape)
            integral = _trapz_compat(_trapz_compat(y=y_array_, x=x_domain), y_domain[:, 0])
        else:
            # MonteCarlo integration
            factor = np.prod(x_array.max(axis=0) - x_array.min(axis=0)) / len(y_array)
            integral = dask_client.submit(lambda x: np.sum(x) * factor, y_array)

    else:
        # MonteCarlo integration
        if dask_client is None:
            factor = np.prod(x_array.max(axis=0) - x_array.min(axis=0)) / y_array.size
            integral = np.sum(y_array) * factor
        else:
            factor = np.prod(x_array.max(axis=0) - x_array.min(axis=0)) / len(y_array)
            integral = dask_client.submit(lambda x: np.sum(x) * factor, y_array)
    return integral


def trapezoidal_rule(x_domain, y_range):
    """
    Computes the integral using the trapezoidal_rule

    Parameters
    ----------
    x_domain : numpy array
        domain for computing the integral
    y_range : numpy array
        range for computing the integral

    Returns
    -------
    integral : float
        integral computed using the trapezoidal rule
    """
    dx = np.diff(x_domain)
    integral = np.dot((y_range[:-1] + y_range[1:]) / 2, dx)
    return integral


def loss_function_qdml(labels, predict_cdf, predict_pdf, integral, loss_weights=[1.0, 5.0]):
    """
    Computes the function for Quantum Differential Machine Learning

    Parameters
    ----------
    labels : numpy array
        numpy array with the labels. Shape: (-1, 1)
    predict_cdf : numpy array
        numpy array with the predictions for the CDF. Shape: (-1, 1)
    predict_pdf : numpy array
        numpy array with the predictions for the PDF. Shape: (-1, 1)
    integral : float
        float with the integral of the square of the PDF in the desired
        domain
    loss_weights : list
        Weights for each part of the Loss function

    Returns
    -------
    loss_ : float
        evaluation of the loss function for QDML
    """

    alpha_0 = loss_weights[0]
    alpha_1 = loss_weights[1]
    # Loss Computation
    # Typical DL error
    if predict_cdf.shape != labels.shape:
        raise ValueError("predict_cdf and labels have different shape!!")
    error_ = predict_cdf - labels
    loss_1 = np.mean(error_**2)
    if predict_pdf.shape != labels.shape:
        raise ValueError("predict_pdf and labels have different shape!!")
    # print("\t loss_1 : {}".format(loss_1))
    mean = -2 * np.mean(predict_pdf)
    # print("\t mean : {}".format(mean))
    loss_ = alpha_0 * loss_1 + alpha_1 * (mean + integral)
    # print("\t loss: {}".format(loss))
    return loss_


# def loss_function_qdml_old(
#     labels, predict_cdf, predict_pdf,
#     x_quad, predict_quad, loss_weights=[1.0, 5.0]):
#     """
#     Computes the function for Quantum Differential Machine Learning
#     Parameters
#     ----------
#     labels : numpy array
#         numpy array with the labels. Shape: (-1, 1)
#     predict_cdf : numpy array
#         numpy array with the predictions for the CDF. Shape: (-1, 1)
#     predict_pdf : numpy array
#         numpy array with the predictions for the PDF. Shape: (-1, 1)
#     x_quad : numpy array
#         numpy array with the domain for computing the integral of the PDF
#         Shape: (-1, number of features)
#     predict_quad : numpy array
#         numpy array with the PDF for computing the integral of the PDF
#         Shape: (-1, 1)
#     Returns
#     -------
#     loss_ : float
#         evaluation of the loss function for QDML
#     """
#
#     alpha_0 = loss_weights[0]
#     alpha_1 = loss_weights[1]
#     # Loss Computation
#     #Typical DL error
#     if predict_cdf.shape != labels.shape:
#         raise ValueError("predict_cdf and labels have different shape!!")
#     error_ = (predict_cdf - labels)
#     loss_1 = np.mean(error_ ** 2)
#     if predict_pdf.shape != labels.shape:
#         raise ValueError("predict_pdf and labels have different shape!!")
#     #print("\t loss_1 : {}".format(loss_1))
#     mean = -2 * np.mean(predict_pdf)
#     #print("\t mean : {}".format(mean))
#     square_for_integral = predict_quad ** 2
#
#     if x_quad.shape[1] == 1:
#         # Typical 1-D trapezoidal integration
#         integral = np.trapz(y=square_for_integral[:, 0], x=x_quad[:, 0])
#     elif x_quad.shape[1] == 2:
#         # 2-D Trapezoidal integration
#         x_domain, y_domain = np.meshgrid(
#             np.unique(x_quad[:, 0]),
#             np.unique(x_quad[:, 1])
#         )
#         square_for_integral = square_for_integral.reshape(x_domain.shape)
#         integral = np.trapz(
#             np.trapz(y=square_for_integral, x=x_domain),
#             x=y_domain[:, 0]
#         )
#     else:
#         # MonteCarlo approach
#         integral = np.sum(square_for_integral) * np.prod(
#             x_quad.max(axis=0) - x_quad.min(axis=0)
#         ) / square_for_integral.size
#     #integral = trapezoidal_rule(x_quad[:, 0], predict_quad[:, 0] * predict_quad[:, 0])
#     #print("\t integral: {}".format(integral))
#     loss_ = alpha_0 * loss_1 + alpha_1 * (mean + integral)
#     #print("\t loss: {}".format(loss))
#     return loss_


def mse(labels, prediction):
    """
    Compute Mean Square Error

    Parameters
    ----------
    labels : numpy array
        numpy array with the labels
    prediction : numpy array
        numpy array with the predictions

    Returns
    -------
    mse_v : float
        MSE value
    """
    error_ = prediction - labels.reshape(prediction.shape)
    mse_v = np.mean(error_**2)
    return mse_v


def compute_loss(weights, produce_results, loss_function):
    """
    Workflow for computing loss:

    Parameters
    ----------
    weights : list
        list with the weights for the PQC
    produce_results : python function
        Function for producing mandatory inputs for computing loss
    loss_function : python function
        Function for computing loss function

    Returns
    -------
    loss_ : float
        Loss value
    """

    output_dict = produce_results(weights)
    loss_ = loss_function(**output_dict)
    return loss_


def numeric_gradient(weights, data_x, data_y, loss):
    """
    Compute the numeric gradient for some input loss function properly
    configured.

    Parameters
    ----------
    weights : numpy array
        Array with weights for PQC
    data_x : numpy array
        Array with dataset of the features
    data_y : numpy array
        Array with targets (labes) dataset
    loss : function
        function for computing the loss properly configured

    Returns
    -------
    gradient_i : numpy array
        Array with the gradients of the Loss Function
    """
    profile_grad = _env_flag("QML4VAR_PROFILE_GRAD", False)
    profile_first_only = _env_flag("QML4VAR_PROFILE_GRAD_FIRST_ONLY", True)
    should_profile = profile_grad and (
        (not profile_first_only) or (not getattr(numeric_gradient, "_profile_done", False))
    )
    if should_profile:
        grad_t0 = perf_counter()
        eval_seconds = []

    gradient_i = []
    epsilon = 1.0e-7
    for i, weight in enumerate(weights):
        # print(weight)
        new_weights = copy.deepcopy(weights)
        new_weights[i] = weight + epsilon
        eval_t0 = perf_counter() if should_profile else None
        loss_plus = loss(new_weights, data_x, data_y)
        if should_profile:
            eval_seconds.append(perf_counter() - eval_t0)
        # print(new_weights)
        # print(loss_plus)
        new_weights = copy.deepcopy(weights)
        new_weights[i] = weight - epsilon
        eval_t0 = perf_counter() if should_profile else None
        loss_minus = loss(new_weights, data_x, data_y)  # , input_x, input_y)
        if should_profile:
            eval_seconds.append(perf_counter() - eval_t0)
        # print(new_weights)
        # print(loss_minus)
        gradient_i = gradient_i + [(loss_plus - loss_minus) / (2.0 * epsilon)]
        # print(gradient_i)

    if should_profile:
        total_grad_s = perf_counter() - grad_t0
        mean_eval_s = float(np.mean(eval_seconds)) if len(eval_seconds) > 0 else 0.0
        max_eval_s = float(np.max(eval_seconds)) if len(eval_seconds) > 0 else 0.0
        print(
            "[PROFILE GRAD] weights={} loss_evals={} eval_mean_s={:.3f} "
            "eval_max_s={:.3f} total_grad_s={:.3f}".format(
                len(weights),
                len(eval_seconds),
                mean_eval_s,
                max_eval_s,
                total_grad_s,
            )
        )
        numeric_gradient._profile_done = True

    return gradient_i
