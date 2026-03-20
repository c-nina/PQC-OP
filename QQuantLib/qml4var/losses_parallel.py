"""
Parallel helpers for loss/gradient computations.

This module is intentionally additive and does not alter existing losses.py.
"""

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed


def numeric_gradient_parallel(
    weights,
    data_x,
    data_y,
    loss,
    epsilon=1.0e-7,
    parallel=True,
    max_workers=4,
):
    """
    Compute numeric gradient with optional threaded parallelism.

    Parameters
    ----------
    weights : array-like
        Current model weights.
    data_x : np.array
        Feature batch.
    data_y : np.array
        Label batch.
    loss : callable
        Function with signature loss(weights, data_x, data_y) -> scalar.
    epsilon : float
        Finite-difference perturbation.
    parallel : bool
        If True, compute each weight gradient pair concurrently.
    max_workers : int
        ThreadPoolExecutor worker count.

    Returns
    -------
    gradient_i : list[float]
        Numeric gradient vector.
    """

    if (not parallel) or int(max_workers) <= 1:
        gradient_i = []
        for i, weight in enumerate(weights):
            new_weights = copy.deepcopy(weights)
            new_weights[i] = weight + epsilon
            loss_plus = loss(new_weights, data_x, data_y)

            new_weights = copy.deepcopy(weights)
            new_weights[i] = weight - epsilon
            loss_minus = loss(new_weights, data_x, data_y)

            gradient_i.append((loss_plus - loss_minus) / (2.0 * epsilon))
        return gradient_i

    def _evaluate_pair(i_weight_tuple):
        i, weight = i_weight_tuple
        plus_weights = copy.deepcopy(weights)
        plus_weights[i] = weight + epsilon
        loss_plus = loss(plus_weights, data_x, data_y)

        minus_weights = copy.deepcopy(weights)
        minus_weights[i] = weight - epsilon
        loss_minus = loss(minus_weights, data_x, data_y)

        return i, (loss_plus - loss_minus) / (2.0 * epsilon)

    gradient_i = [0.0] * len(weights)
    try:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
            futures = [executor.submit(_evaluate_pair, (i, w)) for i, w in enumerate(weights)]
            for future in as_completed(futures):
                i, grad_val = future.result()
                gradient_i[i] = grad_val
        return gradient_i
    except Exception:
        # Safe fallback to sequential path
        gradient_i = []
        for i, weight in enumerate(weights):
            new_weights = copy.deepcopy(weights)
            new_weights[i] = weight + epsilon
            loss_plus = loss(new_weights, data_x, data_y)

            new_weights = copy.deepcopy(weights)
            new_weights[i] = weight - epsilon
            loss_minus = loss(new_weights, data_x, data_y)

            gradient_i.append((loss_plus - loss_minus) / (2.0 * epsilon))
        return gradient_i
