"""
Architectures and Architecture class definition (PennyLane backend)
"""

from typing import Any, List, Optional

import numpy as np
import pennylane as qml
import torch


def hardware_efficient_ansatz(**kwargs: Any):
    """
    Create a hardware efficient ansatz as a PennyLane QNode.

    The circuit applies, for each layer:
      1. RX(normalized_feature) on each qubit  — data encoding
      2. RY(weight) on each qubit              — variational block
      3. CNOT entanglement chain (circular)

    Feature normalization (base_frecuency * x + shift_feature) is applied
    inside the circuit so callers pass raw feature values.

    Parameters
    ----------
    features_number : int
        Number of input features.
    n_qubits_by_feature : int
        Number of qubits used for each feature.
    n_layers : int
        Number of variational layers.
    base_frecuency : list of float
        Slope for feature normalization (one per feature).
    shift_feature : list of float
        Shift for feature normalization (one per feature).
    torch_device : str, optional
        PyTorch device string, e.g. "cpu" or "cuda". Default: "cpu".

    Returns
    -------
    circuit : callable
        PennyLane QNode with signature circuit(weights, raw_features) -> scalar.
        Both arguments must be torch.Tensor.
    weights_names : list of str
        Parameter names for the trainable weights.
    features_names : list of str
        Parameter names for the input features.
    """
    features_number = kwargs.get("features_number")
    n_qubits_by_feature = kwargs.get("n_qubits_by_feature")
    n_layers = kwargs.get("n_layers")
    base_frecuency = kwargs.get("base_frecuency", [1.0] * features_number)
    shift_feature = kwargs.get("shift_feature", [0.0] * features_number)

    n_qubits = n_qubits_by_feature * features_number

    features_names = ["feature_{}".format(i) for i in range(features_number)]

    weights_names = []
    for layer_ in range(n_layers):
        for input_ in range(features_number):
            for qubit_ in range(n_qubits_by_feature):
                weights_names.append("weights_{}_{}_{}".format(layer_, input_, qubit_))

    # Capture normalization constants as tensors
    bf = torch.tensor(base_frecuency, dtype=torch.float64)
    sf = torch.tensor(shift_feature, dtype=torch.float64)

    # default.qubit + backprop is required for two reasons:
    #   1. The PDF loss term needs create_graph=True (second-order differentiation:
    #      d²(circuit)/(dx·dw)), which adjoint diff does NOT support.
    #   2. qml.vmap (used for batched evaluation in the workflow) is compatible
    #      with default.qubit + backprop but not with lightning.qubit + adjoint.
    # GPU acceleration is achieved via torch_device="cuda" in workflow_cfg, which
    # moves all tensor/gradient operations (Adam, loss, backprop) to CUDA.
    # For circuits with ≤ 12 qubits, the state vector (≤ 4096 amplitudes) is
    # negligible in cost; the real gain comes from batching + GPU tensor ops.
    dev = qml.device("default.qubit", wires=n_qubits)
    print(f"[hardware_efficient_ansatz] device=default.qubit  diff_method=backprop  wires={n_qubits}")

    @qml.qnode(dev, diff_method="backprop", interface="torch")
    def circuit(weights, raw_features):
        """
        Parameters
        ----------
        weights : torch.Tensor, shape (n_layers * features_number * n_qubits_by_feature,)
        raw_features : torch.Tensor, shape (features_number,)
        """
        # Apply feature normalization inside the circuit (same as old QLM parametric expr)
        norm_features = bf.to(raw_features.device) * raw_features + sf.to(raw_features.device)

        for layer_ in range(n_layers):
            for input_ in range(features_number):
                base_w = layer_ * features_number * n_qubits_by_feature + input_ * n_qubits_by_feature
                for qubit_ in range(n_qubits_by_feature):
                    actual_qubit = input_ * n_qubits_by_feature + qubit_
                    qml.RX(norm_features[input_], wires=actual_qubit)
                for qubit_ in range(n_qubits_by_feature):
                    actual_qubit = input_ * n_qubits_by_feature + qubit_
                    qml.RY(weights[base_w + qubit_], wires=actual_qubit)

            # Circular entanglement layer
            for qubit_ in range(n_qubits - 1):
                qml.CNOT(wires=[qubit_, qubit_ + 1])
            if n_qubits > 1:
                qml.CNOT(wires=[n_qubits - 1, 0])

        # Z⊗Z⊗...⊗Z observable
        obs = qml.PauliZ(0)
        for q in range(1, n_qubits):
            obs = obs @ qml.PauliZ(q)
        return qml.expval(obs)

    return circuit, weights_names, features_names


def normalize_data(min_value: list, max_value: list, min_x: Optional[list] = None, max_x: Optional[list] = None):
    """
    Feature Normalization.

    Parameters
    ----------
    min_value : list
        Minimum value for all features.
    max_value : list
        Maximum value for all features.
    min_x : list, optional
        Minimum encoding value (rotation angle). Default: -pi.
    max_x : list, optional
        Maximum encoding value (rotation angle). Default: +pi.

    Returns
    -------
    slope : np.ndarray
    b0 : np.ndarray
    """
    max_value_ = np.array(max_value)
    min_value_ = np.array(min_value)
    min_x = np.array([-np.pi]) if min_x is None else np.array(min_x)
    max_x = np.array([+np.pi]) if max_x is None else np.array(max_x)
    slope = (max_x - min_x) / (max_value_ - min_value_)
    b0 = min_x - slope * min_value_
    return slope, b0


def z_observable(**kwargs: Any) -> qml.operation.Operator:
    """
    Return the Z⊗Z⊗...⊗Z observable for the given architecture.

    Parameters
    ----------
    features_number : int
    n_qubits_by_feature : int

    Returns
    -------
    qml.operation.Operator
    """
    features_number = kwargs.get("features_number", 1)
    n_qubits_by_feature = kwargs.get("n_qubits_by_feature", 1)
    n_qubits = features_number * n_qubits_by_feature
    obs = qml.PauliZ(0)
    for q in range(1, n_qubits):
        obs = obs @ qml.PauliZ(q)
    return obs


def init_weights(weights_names: List[str]):
    """
    Initialize PQC weights uniformly in [0, 1).

    Parameters
    ----------
    weights_names : list of str

    Returns
    -------
    dict mapping weight name -> float
    """
    return {v: np.random.uniform() for v in weights_names}
