"""
losses_parallel.py — API compatibility shim.

The threaded numeric_gradient_parallel is superseded by torch_gradient, which
computes the full gradient in a single backward pass via PyTorch autograd.

The wrapper below accepts the original `parallel` and `max_workers` kwargs
so that notebook 26 code does not need to change its call site.
"""

from QQuantLib.qml4var.losses import torch_gradient


def numeric_gradient_parallel(
    weights,
    data_x,
    data_y,
    loss_fn,
    parallel=None,
    max_workers=None,
    epsilon=None,
):
    """
    Compute gradient via PyTorch backprop (replaces threaded finite-difference).

    The `parallel`, `max_workers`, and `epsilon` arguments are accepted for
    API compatibility with notebook 26 but are ignored: backpropagation is
    used regardless.

    Parameters
    ----------
    weights : list of float
    data_x : np.array
    data_y : np.array
    loss_fn : callable  loss_fn(weights_tensor, data_x, data_y) -> torch scalar
    parallel, max_workers, epsilon : ignored

    Returns
    -------
    list of float
    """
    return torch_gradient(weights, data_x, data_y, loss_fn)


__all__ = ["numeric_gradient_parallel"]
