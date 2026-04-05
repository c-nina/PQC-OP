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
from qml4var.data_utils import bs_cdf, empirical_cdf
from qml4var.losses import mse, torch_gradient, qdml_loss_workflow, unsupervised_qdml_loss_workflow
from qml4var.workflows import (
    cdf_workflow,
    dft_from_trained_pqc,
    pdf_workflow,
    workflow_for_cdf,
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
