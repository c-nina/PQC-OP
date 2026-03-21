"""
Parallel-oriented qml4var workflows (PennyLane backend).

With PennyLane + PyTorch backprop, gradient computation is done in a single
backward pass, making the old QPU-caching and ThreadPoolExecutor strategies
unnecessary. This module re-exports the base workflow functions under their
original "parallel" names for API compatibility with notebook 26.
"""

from QQuantLib.qml4var.myqlm_workflows import (
    cdf_workflow as cdf_workflow_parallel,
    pdf_workflow as pdf_workflow_parallel,
    workflow_for_cdf as workflow_for_cdf_parallel,
    workflow_for_pdf as workflow_for_pdf_parallel,
    qdml_loss_workflow as qdml_loss_workflow_parallel,
    unsupervised_qdml_loss_workflow as unsupervised_qdml_loss_workflow_parallel,
    mse_workflow as mse_workflow_parallel,
    dft_from_trained_pqc as dft_from_trained_pqc_parallel,
)

__all__ = [
    "cdf_workflow_parallel",
    "pdf_workflow_parallel",
    "workflow_for_cdf_parallel",
    "workflow_for_pdf_parallel",
    "qdml_loss_workflow_parallel",
    "unsupervised_qdml_loss_workflow_parallel",
    "mse_workflow_parallel",
    "dft_from_trained_pqc_parallel",
]
