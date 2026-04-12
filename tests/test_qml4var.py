"""
Tests for the qml4var module.
Run with: pytest tests/test_qml4var.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch

from qml4var.architectures import hardware_efficient_ansatz, init_weights, normalize_data
from qml4var.data_utils import bs_cdf, empirical_cdf, generate_method_I_data
from qml4var.losses import mse, qdml_loss_workflow, torch_gradient, unsupervised_qdml_loss_workflow
from qml4var.workflows import (
    cdf_workflow,
    dft_from_trained_pqc,
    pdf_workflow,
    workflow_for_cdf,
    workflow_for_pdf_direct,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_circuit():
    """1 feature, 2 qubits, 1 layer — fast for all tests."""
    circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
        features_number=1,
        n_qubits_by_feature=2,
        n_layers=1,
        base_frecuency=[1.0],
        shift_feature=[0.0],
    )
    weights = init_weights(weights_names)
    return circuit_fn, weights, weights_names, features_names


@pytest.fixture(scope="module")
def workflow_cfg(small_circuit):
    circuit_fn, weights, weights_names, features_names = small_circuit
    return {
        "circuit_fn": circuit_fn,
        "features_names": features_names,
        "torch_device": "cpu",
        "loss_weights": [1.0, 5.0],
        "minval": [-1.0],
        "maxval": [1.0],
        "points": 20,
    }


# ---------------------------------------------------------------------------
# architectures
# ---------------------------------------------------------------------------

def test_circuit_returns_scalar(small_circuit):
    circuit_fn, weights, *_ = small_circuit
    w_t = torch.tensor(list(weights.values()), dtype=torch.float64)
    x_t = torch.tensor([0.5], dtype=torch.float64)
    result = circuit_fn(w_t, x_t)
    assert result.ndim == 0


def test_circuit_output_bounded(small_circuit):
    """Expectation value of Z⊗Z...⊗Z is in [-1, 1]."""
    circuit_fn, weights, *_ = small_circuit
    w_t = torch.tensor(list(weights.values()), dtype=torch.float64)
    for x_val in [-1.0, 0.0, 1.0]:
        x_t = torch.tensor([x_val], dtype=torch.float64)
        result = circuit_fn(w_t, x_t).item()
        assert -1.0 <= result <= 1.0


def test_init_weights_keys(small_circuit):
    _, weights, weights_names, _ = small_circuit
    assert set(weights.keys()) == set(weights_names)


def test_normalize_data_known_values():
    slope, b0 = normalize_data(
        min_value=[0.0], max_value=[1.0],
        min_x=[-np.pi / 2], max_x=[np.pi / 2],
    )
    assert np.isclose(slope[0], np.pi, atol=1e-10)
    assert np.isclose(b0[0], -np.pi / 2, atol=1e-10)


# ---------------------------------------------------------------------------
# workflows — cdf / pdf
# ---------------------------------------------------------------------------

def test_cdf_in_range(small_circuit, workflow_cfg):
    _, weights, *_ = small_circuit
    val = cdf_workflow(weights, np.array([0.0]), **workflow_cfg)
    assert -1.0 <= val <= 1.0


def test_pdf_is_float(small_circuit, workflow_cfg):
    _, weights, *_ = small_circuit
    val = pdf_workflow(weights, np.array([0.0]), **workflow_cfg)
    assert isinstance(val, float)


def test_workflow_for_cdf_shape(small_circuit, workflow_cfg):
    _, weights, *_ = small_circuit
    data_x = np.linspace(-1.0, 1.0, 5).reshape(-1, 1)
    out = workflow_for_cdf(weights, data_x, **workflow_cfg)
    assert out["y_predict_cdf"].shape == (5,)


# ---------------------------------------------------------------------------
# losses — torch_gradient
# ---------------------------------------------------------------------------

def test_torch_gradient_shape(small_circuit, workflow_cfg):
    _, weights, weights_names, _ = small_circuit
    data_x = np.linspace(-0.5, 0.5, 4).reshape(-1, 1)
    data_y = np.random.uniform(0, 1, (4, 1))

    def loss_fn(w, x, y):
        return qdml_loss_workflow(w, x, y, **workflow_cfg)
    grad = torch_gradient(list(weights.values()), data_x, data_y, loss_fn)

    assert len(grad) == len(weights_names)
    assert all(isinstance(g, float) for g in grad)


def test_torch_gradient_nonzero(small_circuit, workflow_cfg):
    """Gradient should not be all zeros for a random init."""
    _, weights, *_ = small_circuit
    data_x = np.linspace(-0.5, 0.5, 4).reshape(-1, 1)
    data_y = np.random.uniform(0, 1, (4, 1))

    def loss_fn(w, x, y):
        return qdml_loss_workflow(w, x, y, **workflow_cfg)
    grad = torch_gradient(list(weights.values()), data_x, data_y, loss_fn)
    assert any(abs(g) > 1e-12 for g in grad)


# ---------------------------------------------------------------------------
# losses — qdml_loss_workflow
# ---------------------------------------------------------------------------

def test_qdml_loss_returns_float(small_circuit, workflow_cfg):
    _, weights, *_ = small_circuit
    data_x = np.linspace(-0.5, 0.5, 5).reshape(-1, 1)
    data_y = np.random.uniform(0, 1, (5, 1))
    loss = qdml_loss_workflow(weights, data_x, data_y, **workflow_cfg)
    assert isinstance(loss, float)


def test_unsupervised_loss_returns_float(small_circuit, workflow_cfg):
    _, weights, *_ = small_circuit
    data_x = np.linspace(-0.5, 0.5, 5).reshape(-1, 1)
    loss = unsupervised_qdml_loss_workflow(weights, data_x, **workflow_cfg)
    assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# dft_from_trained_pqc
# ---------------------------------------------------------------------------

def _dft_cfg(workflow_cfg):
    """workflow_cfg without minval/maxval/points — dft_from_trained_pqc takes them as positional kwargs."""
    return {k: v for k, v in workflow_cfg.items() if k not in ("minval", "maxval", "points")}


def test_dft_output_shape(small_circuit, workflow_cfg):
    _, weights, *_ = small_circuit
    result = dft_from_trained_pqc(
        weights, minval=-np.pi, maxval=np.pi, points=16,
        prediction="cdf", **_dft_cfg(workflow_cfg),
    )
    assert result["x_domain"].shape == (16,)
    assert result["y_predict"].shape == (16,)
    assert result["k_values"].shape == (16,)
    assert result["c_k"].shape == (16,)


def test_dft_k_values_sorted(small_circuit, workflow_cfg):
    _, weights, *_ = small_circuit
    result = dft_from_trained_pqc(
        weights, minval=-np.pi, maxval=np.pi, points=16,
        prediction="cdf", **_dft_cfg(workflow_cfg),
    )
    assert np.all(result["k_values"][:-1] <= result["k_values"][1:])


# ---------------------------------------------------------------------------
# data_utils
# ---------------------------------------------------------------------------

def test_empirical_cdf_bounds():
    data = np.random.randn(50).reshape(-1, 1)
    cdf = empirical_cdf(data)
    assert np.all(cdf >= 0.0)
    assert np.all(cdf <= 1.0)


def test_empirical_cdf_monotone():
    data = np.sort(np.random.randn(30)).reshape(-1, 1)
    cdf = empirical_cdf(data)
    assert np.all(np.diff(cdf) >= 0)


def test_bs_cdf_bounds():
    s_values = np.linspace(0.5, 2.0, 20)
    cdf_vals = np.array([bs_cdf(s) for s in s_values])
    assert np.all(cdf_vals >= 0.0)
    assert np.all(cdf_vals <= 1.0)


def test_bs_cdf_monotone():
    s_values = np.linspace(0.5, 2.0, 20)
    cdf_vals = np.array([bs_cdf(s) for s in s_values])
    assert np.all(np.diff(cdf_vals) >= 0)


def test_mse_zero_on_perfect_prediction():
    y = np.array([0.1, 0.5, 0.9])
    assert mse(y, y) == pytest.approx(0.0)


def test_mse_positive():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([1.0, 0.0])
    assert mse(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Method I real — new tests (paper Sec. 3.2.1)
# ---------------------------------------------------------------------------

def test_generate_method_I_data_pdf_integrates_to_one():
    """PDF labels from generate_method_I_data must integrate between 0.95 and 1.0."""
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    grid, pdf_vals, pdf_deriv, a, b = generate_method_I_data(
        S0=100.0, K=100.0, r=0.1, T=1.0, sigma=0.25, n_points=5000
    )
    integral = trapz_fn(pdf_vals, grid[:, 0])
    assert 0.95 <= integral <= 1.0, f"PDF integral={integral:.6f} not in [0.95, 1.0]"


def test_workflow_for_pdf_direct_bounded(small_circuit, workflow_cfg):
    """workflow_for_pdf_direct must return values in [-1, 1] (raw circuit output)."""
    _, weights, *_ = small_circuit
    data_x = np.linspace(-np.pi, np.pi, 20).reshape(-1, 1)
    out = workflow_for_pdf_direct(weights, data_x, **workflow_cfg)
    preds = out["y_predict_pdf"]
    assert preds.min() >= -1.0 - 1e-9
    assert preds.max() <= 1.0 + 1e-9


def test_method_I_real_price_in_physical_range():
    """Method I real smoke test: estimated price for K=100 must be in [0, K*exp(-rT)]."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import torch

    from finance import estimate_price_from_trained_pqc
    from qml4var.adam import adam_optimizer_loop
    from qml4var.architectures import hardware_efficient_ansatz, init_weights
    from qml4var.data_utils import generate_method_I_data
    from qml4var.losses import method_I_h1_loss, torch_gradient

    K = 100.0
    r = 0.1
    T = 1.0
    price_upper = K * np.exp(-r * T)  # ≈ 90.48

    circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
        features_number=1, n_qubits_by_feature=6, n_layers=6,
        base_frecuency=[1.0], shift_feature=[0.0],
    )
    weights_dict = init_weights(weights_names)
    cfg_wf = dict(circuit_fn=circuit_fn, torch_device="cpu", features_names=features_names)

    train_x, pdf_labels, pdf_deriv_labels, a, b = generate_method_I_data(
        S0=100.0, K=K, r=r, T=T, sigma=0.25, n_points=250
    )

    def loss_fn(w):
        w_t = torch.tensor(list(w.values()) if isinstance(w, dict) else list(w), dtype=torch.float64)
        return method_I_h1_loss(
            w_t, train_x, pdf_labels, pdf_deriv_labels,
            circuit_fn=circuit_fn, device="cpu", alpha_0=0.9, alpha_1=0.1, create_graph=False,
        ).item()

    def gradient_fn(w, bx, by):
        def _loss_torch(w_t, bx_, by_):
            return method_I_h1_loss(
                w_t, bx_, by_, pdf_deriv_labels,
                circuit_fn=circuit_fn, device="cpu", alpha_0=0.9, alpha_1=0.1, create_graph=True,
            )
        return torch_gradient(list(w), bx, by, _loss_torch)

    final_weights = adam_optimizer_loop(
        weights_dict=weights_dict,
        loss_function=loss_fn,
        metric_function=None,
        gradient_function=gradient_fn,
        batch_generator=[(train_x, pdf_labels.reshape(-1, 1))],
        epochs=50,
        learning_rate=0.005,
        beta1=0.9,
        beta2=0.999,
        tolerance=-1e30,
        n_counts_tolerance=100,
        print_step=100,
        file_to_save=None,
        progress_bar=False,
    )

    artifacts = {"workflow_cfg": cfg_wf}
    est_price = estimate_price_from_trained_pqc(
        weights=final_weights,
        artifacts=artifacts,
        K_=K,
        x_min_raw=a,
        x_max_raw=b,
        train_interval=(-np.pi, np.pi),
        risk_free_rate=r,
        delta_t=T,
        k_terms=12,
    )

    assert np.isfinite(est_price), f"Price is not finite: {est_price}"
    assert 0.0 <= est_price <= price_upper, (
        f"Price {est_price:.4f} not in [0, {price_upper:.4f}]"
    )
