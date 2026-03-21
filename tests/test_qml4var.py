"""
Basic smoke tests for the qml4var module.
Run with: pytest tests/test_qml4var.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from qml4var.architectures import hardware_efficient_ansatz, init_weights
from qml4var.workflows import cdf_workflow, pdf_workflow


def test_circuit_returns_scalar():
    circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
        features_number=1,
        n_qubits_by_feature=2,
        n_layers=1,
        base_frecuency=[1.0],
        shift_feature=[0.0],
    )
    weights = init_weights(weights_names)
    w_t = torch.tensor(list(weights.values()), dtype=torch.float64)
    x_t = torch.tensor([0.5], dtype=torch.float64)
    result = circuit_fn(w_t, x_t)
    assert result.ndim == 0


def test_cdf_in_range():
    circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
        features_number=1,
        n_qubits_by_feature=2,
        n_layers=1,
        base_frecuency=[1.0],
        shift_feature=[0.0],
    )
    weights = init_weights(weights_names)
    val = cdf_workflow(weights, np.array([0.0]), circuit_fn=circuit_fn)
    assert -1.0 <= val <= 1.0


def test_pdf_is_float():
    circuit_fn, weights_names, features_names = hardware_efficient_ansatz(
        features_number=1,
        n_qubits_by_feature=2,
        n_layers=1,
        base_frecuency=[1.0],
        shift_feature=[0.0],
    )
    weights = init_weights(weights_names)
    val = pdf_workflow(weights, np.array([0.0]), circuit_fn=circuit_fn)
    assert isinstance(val, float)
