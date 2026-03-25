""" "
Adam
"""

import itertools
import os
from time import perf_counter
from typing import Any, Callable, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def _env_flag(name: str, default: bool = False):
    """
    Parse boolean-like environment variables.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int):
    """
    Parse integer environment variables.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def save_stuff(
    weights: list,
    weights_names: List[str],
    t_: int,
    loss_: float,
    metric_mse_: Optional[float] = None,
    file_to_save: Optional[str] = None,
):
    """
    Save stuff
    """
    pdf = pd.DataFrame(weights, index=weights_names).T
    pdf["t"] = t_
    pdf["loss"] = loss_
    pdf["metric_mse"] = metric_mse_
    if file_to_save is not None:
        pdf.to_csv(file_to_save, sep=";", index=True, mode="a", header=False)


def batch_generator(iterable: Iterable, batch_size: int = 1):
    iterable = iter(iterable)

    while True:
        batch = list(itertools.islice(iterable, batch_size))
        if len(batch) > 0:
            yield batch
        else:
            break


def initialize_adam(parameters: list):
    """
    Initialize the parameters of ADAM
    """

    v = np.zeros(len(parameters))
    s = np.zeros(len(parameters))

    return v, s


# Update parameters using Adam
def update_parameters_with_adam(
    x: np.ndarray,
    grads: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    t: int,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
):
    """
    Update the parameters of ADAM
    """
    s = beta1 * s + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * grads**2
    s_hat = s / (1.0 - beta1 ** (t + 1))
    v_hat = v / (1.0 - beta2 ** (t + 1))
    x = x - learning_rate * s_hat / (np.sqrt(v_hat) + epsilon)
    return x, s, v


def adam_optimizer_loop(
    weights_dict: dict,
    loss_function: Callable,
    metric_function: Optional[Callable],
    gradient_function: Callable,
    batch_generator: Iterable,
    initial_time: int = 0,
    **kwargs: Any,
):
    """
    Parameters
    ----------
    weights_dict : dict
        dictionary with the weights to fit
    loss_function : function
        function for computing the loss function
    metric_function : fuction
        function for computing the metric function
    gradient_function : function
        function for computing the gradient of the loss function
    batch_generator : function
        function for generating batches of the trainin data.
    initial_time : int
        Initial time step
    kwargs : keyword arguments
        arguments for configuring optimizer. For ADAM:

    store_folder : kwargs, str
        Folder for saving results. If None not saving
    epochs : kwargs, int
        Maximum number of iterations
    tolerance : kwargs, float
        Tolerance to achieve
    n_counts_tolerance : kwargs, int
        Number of times the tolerance should be achieved in consecutive
        iterations
    print_step : kwargs, int
        Print_step for printing evolution of training
    learning_rate : kwargs,float
        Learning_rate for ADAM
    beta1 : kwargs, float
        beta1 for ADAM
    beta2 : kwargs, float
        beta2 for ADAM
    progress_bar : kwargs, bool
        If True and tqdm is available, displays epoch progress bar
    progress_desc : kwargs, str
        Optional description for tqdm bar
    progress_leave : kwargs, bool
        Keep tqdm bar displayed after training
    profile_timing : kwargs, bool
        If True, prints lightweight timing breakdown by epoch.
        Can also be enabled with env var QML4VAR_PROFILE_TRAINING=1
    profile_first_n_epochs : kwargs, int
        Number of initial epochs to profile. Default: 1
        (or env var QML4VAR_PROFILE_EPOCHS)
    profile_once : kwargs, bool
        If True, profile only the first optimizer call in the process.
        Default: True (or env var QML4VAR_PROFILE_ONCE)
    profile_label : kwargs, str
        Optional label shown in profile lines.
    checkpoint_fn : kwargs, callable, optional
        ``checkpoint_fn(weights, epoch, train_loss, test_mse)`` called every
        *checkpoint_step* epochs and at epoch 0.  Build it with
        :func:`qml4var.results.make_checkpoint_fn`.
    checkpoint_step : kwargs, int, optional
        How often (in epochs) to call *checkpoint_fn*.  Default: 50.
    """
    # Get Weights
    weights = list(weights_dict.values())
    weights_names = list(weights_dict.keys())
    # Init Adam
    s_, v_ = initialize_adam(weights)  # .keys())
    # ADAM time parameter
    t_ = initial_time
    # Tolerance steps
    n_tol = 0

    # Deal with save Folder
    file_to_save = kwargs.get("file_to_save")
    # Checkpoint callback
    checkpoint_fn = kwargs.get("checkpoint_fn")
    checkpoint_step = int(kwargs.get("checkpoint_step", 50))
    # Configure Stop
    epochs = kwargs.get("epochs")
    tolerance = kwargs.get("tolerance")
    n_counts_tolerance = kwargs.get("n_counts_tolerance")
    # Configure printing info
    print_step = kwargs.get("print_step")
    # Configure Adam
    learning_rate = kwargs.get("learning_rate")
    beta1 = kwargs.get("beta1")
    beta2 = kwargs.get("beta2")
    # Configure optional progress bar
    progress_bar = bool(kwargs.get("progress_bar", False))
    progress_desc = kwargs.get("progress_desc", "ADAM epochs")
    progress_leave = bool(kwargs.get("progress_leave", False))

    # Optional lightweight timing profiler
    profile_timing = kwargs.get("profile_timing")
    if profile_timing is None:
        profile_timing = _env_flag("QML4VAR_PROFILE_TRAINING", False)
    profile_once = bool(kwargs.get("profile_once", _env_flag("QML4VAR_PROFILE_ONCE", True)))
    profile_first_n_epochs = int(kwargs.get("profile_first_n_epochs", _env_int("QML4VAR_PROFILE_EPOCHS", 1)))
    profile_label = kwargs.get("profile_label", os.getenv("QML4VAR_PROFILE_LABEL", ""))
    if profile_timing and profile_once and getattr(adam_optimizer_loop, "_profile_done", False):
        profile_timing = False

    pbar = None
    if progress_bar and _tqdm is not None and epochs is not None:
        pbar = _tqdm(
            total=max(int(epochs) - int(initial_time), 0),
            desc=progress_desc,
            leave=progress_leave,
            unit="epoch",
        )

    # Compute Initial Loss and Metric
    loss_0 = loss_function(weights)
    if metric_function is None:
        metric_mse_0 = None
    else:
        metric_mse_0 = metric_function(weights)
        print("Loss Function at t={}: {}".format(t_, loss_0))
        print("MSE at t={}: {}".format(t_, metric_mse_0))
        save_stuff(weights, weights_names, t_, loss_0, metric_mse_0, file_to_save)
        if checkpoint_fn is not None:
            checkpoint_fn(weights, t_, loss_0, metric_mse_0)

    converged = False
    start_t = t_
    for t_ in range(start_t, epochs):
        epoch_t0 = perf_counter()
        grad_seconds = 0.0
        update_seconds = 0.0
        loss_seconds = 0.0
        metric_seconds = 0.0
        n_batches = 0
        for batch in batch_generator:
            # Get the Batches
            batch_x = batch[0]
            batch_y = batch[1]
            # Compute gradient on batches
            grad_t0 = perf_counter()
            loss_gradient = np.array(gradient_function(weights, batch_x, batch_y))
            grad_seconds += perf_counter() - grad_t0
            # Update Weights
            update_t0 = perf_counter()
            weights, s_, v_ = update_parameters_with_adam(
                weights, loss_gradient, s_, v_, t_, learning_rate=learning_rate, beta1=beta1, beta2=beta2
            )
            update_seconds += perf_counter() - update_t0
            n_batches = n_batches + 1
        loss_t0 = perf_counter()
        loss_t = loss_function(weights)
        loss_seconds += perf_counter() - loss_t0
        delta = -(loss_t - loss_0)
        loss_0 = loss_t
        n_tol = n_tol + 1 if delta < tolerance else 0
        if t_ % print_step == 0:
            # Compute loss
            if metric_function is None:
                metric_mse_t = None
            else:
                metric_t0 = perf_counter()
                metric_mse_t = metric_function(weights)
                metric_seconds += perf_counter() - metric_t0
            print("\t MSE at t={}: {}".format(t_, metric_mse_t))
            print("\t Iteracion: {}. Loss: {}".format(t_, loss_t))
            save_stuff(weights, weights_names, t_, loss_0, metric_mse_t, file_to_save)
        if checkpoint_fn is not None and t_ % checkpoint_step == 0:
            metric_for_ckpt = metric_mse_t if t_ % print_step == 0 else (
                None if metric_function is None else metric_function(weights)
            )
            checkpoint_fn(weights, t_, loss_t, metric_for_ckpt)

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                "loss": "{:.3e}".format(loss_t),
                "n_tol": int(n_tol),
            })

        if profile_timing and (t_ - initial_time) < profile_first_n_epochs:
            epoch_total = perf_counter() - epoch_t0
            misc_seconds = max(
                epoch_total - grad_seconds - update_seconds - loss_seconds - metric_seconds,
                0.0,
            )
            label = " {}".format(profile_label) if profile_label else ""
            print(
                "[PROFILE{}] epoch={} batches={} grad_s={:.3f} "
                "update_s={:.3f} loss_s={:.3f} metric_s={:.3f} "
                "misc_s={:.3f} total_s={:.3f}".format(
                    label,
                    t_,
                    n_batches,
                    grad_seconds,
                    update_seconds,
                    loss_seconds,
                    metric_seconds,
                    misc_seconds,
                    epoch_total,
                )
            )

        if n_tol >= n_counts_tolerance:
            print("Achieved Convergence. Delta: {}".format(delta))
            metric_mse_t = None if metric_function is None else metric_function(weights)
            save_stuff(weights, weights_names, t_, loss_0, metric_mse_t, file_to_save)
            converged = True
            break
    if pbar is not None:
        pbar.close()

    if not converged:
        print("Maximum number of iterations achieved.")
    metric_mse_t = None if metric_function is None else metric_function(weights)
    save_stuff(weights, weights_names, t_, loss_t, metric_mse_t, file_to_save)
    if profile_timing and profile_once:
        adam_optimizer_loop._profile_done = True
    return weights
