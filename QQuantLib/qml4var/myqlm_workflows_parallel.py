"""
Parallel-oriented qml4var workflows.

This module keeps API-compatible helpers while adding:
- cached QPU selection to avoid rebuilding QPU objects repeatedly
- explicit parallel execution helper with dask map/submit fallback

The goal is speed improvements without modifying existing workflows.
"""

from functools import lru_cache
from itertools import product
import json

import numpy as np
from qat.core import Batch

from QQuantLib.qpu.select_qpu import select_qpu
from QQuantLib.qml4var.plugins import SetParametersPlugin, pdfPluging, MyQPU
from QQuantLib.qml4var.losses import loss_function_qdml, mse, compute_integral
from QQuantLib.qml4var.data_utils import empirical_cdf


def _qpu_key(qpu_info):
    """
    Build hashable cache key from qpu config dictionary.
    """
    if qpu_info is None:
        raise ValueError("qpu_info should be provided in kwargs")
    return json.dumps(qpu_info, sort_keys=True)


@lru_cache(maxsize=32)
def _get_cached_qpu(qpu_key):
    """
    Cached QPU factory.
    """
    return select_qpu(json.loads(qpu_key))


def stack_execution(weights, x_sample, stack, **kwargs):
    """
    Execute stack for one sample (same contract as base workflow module).
    """
    pqc = kwargs.get("pqc")
    observable = kwargs.get("observable")
    weights_names = kwargs.get("weights_names")
    features_names = kwargs.get("features_names")
    nbshots = kwargs.get("nbshots")

    circuit = pqc.to_circ()
    job = circuit.to_job(nbshots=nbshots, observable=observable)
    cdf_batch = Batch(jobs=[job])
    cdf_batch.meta_data = {
        "weights": weights_names,
        "features": features_names,
    }
    results = stack(weights, x_sample).submit(cdf_batch)
    return results


def cdf_workflow_parallel(weights, x_sample, **kwargs):
    """
    CDF workflow with cached QPU selection.
    """
    qpu_dict = kwargs.get("qpu_info")
    qpu = _get_cached_qpu(_qpu_key(qpu_dict))

    stack_cdf = lambda weights_, features_: SetParametersPlugin(weights_, features_) | MyQPU(qpu)
    workflow_cfg = {
        "pqc": kwargs.get("pqc"),
        "observable": kwargs.get("observable"),
        "weights_names": kwargs.get("weights_names"),
        "features_names": kwargs.get("features_names"),
        "nbshots": kwargs.get("nbshots"),
    }
    results = stack_execution(weights, x_sample, stack_cdf, **workflow_cfg)
    return results[0].value


def pdf_workflow_parallel(weights, x_sample, **kwargs):
    """
    PDF workflow with cached QPU selection.
    """
    qpu_dict = kwargs.get("qpu_info")
    qpu = _get_cached_qpu(_qpu_key(qpu_dict))

    features_names = kwargs.get("features_names")
    stack_pdf = lambda weights_, features_: (
        pdfPluging(features_names) | SetParametersPlugin(weights_, features_) | MyQPU(qpu)
    )
    workflow_cfg = {
        "pqc": kwargs.get("pqc"),
        "observable": kwargs.get("observable"),
        "weights_names": kwargs.get("weights_names"),
        "features_names": kwargs.get("features_names"),
        "nbshots": kwargs.get("nbshots"),
    }
    results = stack_execution(weights, x_sample, stack_pdf, **workflow_cfg)
    return results[0].value


def workflow_execution_parallel(weights, data_x, workflow, dask_client=None):
    """
    Execute workflow for all samples using optional dask map/submit fallback.
    """
    if dask_client is None:
        return [workflow(weights, x_) for x_ in data_x]

    try:
        return dask_client.map(workflow, *([weights] * data_x.shape[0], data_x))
    except Exception:
        return [dask_client.submit(workflow, weights, x_, pure=False) for x_ in data_x]


def workflow_for_cdf_parallel(weights, data_x, dask_client=None, **kwargs):
    """
    Compute CDF predictions for a dataset.
    """
    workflow_cfg = {
        "pqc": kwargs.get("pqc"),
        "observable": kwargs.get("observable"),
        "weights_names": kwargs.get("weights_names"),
        "features_names": kwargs.get("features_names"),
        "nbshots": kwargs.get("nbshots"),
        "qpu_info": kwargs.get("qpu_info"),
    }
    cdf_workflow_ = lambda w, x: cdf_workflow_parallel(w, x, **workflow_cfg)
    cdf_prediction = workflow_execution_parallel(weights, data_x, cdf_workflow_, dask_client=dask_client)
    if dask_client is None:
        cdf_prediction = np.array(cdf_prediction)
    else:
        cdf_prediction = np.array(dask_client.gather(cdf_prediction))
    return {"y_predict_cdf": cdf_prediction}


def workflow_for_pdf_parallel(weights, data_x, dask_client=None, **kwargs):
    """
    Compute PDF predictions for a dataset.
    """
    workflow_cfg = {
        "pqc": kwargs.get("pqc"),
        "observable": kwargs.get("observable"),
        "weights_names": kwargs.get("weights_names"),
        "features_names": kwargs.get("features_names"),
        "nbshots": kwargs.get("nbshots"),
        "qpu_info": kwargs.get("qpu_info"),
    }
    pdf_workflow_ = lambda w, x: pdf_workflow_parallel(w, x, **workflow_cfg)
    pdf_prediction = workflow_execution_parallel(weights, data_x, pdf_workflow_, dask_client=dask_client)
    if dask_client is None:
        pdf_prediction = np.array(pdf_prediction)
    else:
        pdf_prediction = np.array(dask_client.gather(pdf_prediction))
    return {"y_predict_pdf": pdf_prediction}


def workflow_for_qdml_parallel(weights, data_x, data_y, dask_client=None, **kwargs):
    """
    Build arrays required for qdml loss computation.
    """
    workflow_cfg = {
        "pqc": kwargs.get("pqc"),
        "observable": kwargs.get("observable"),
        "weights_names": kwargs.get("weights_names"),
        "features_names": kwargs.get("features_names"),
        "nbshots": kwargs.get("nbshots"),
        "qpu_info": kwargs.get("qpu_info"),
    }

    cdf_workflow_ = lambda w, x: cdf_workflow_parallel(w, x, **workflow_cfg)
    pdf_workflow_ = lambda w, x: pdf_workflow_parallel(w, x, **workflow_cfg)
    pdf_square_workflow_ = lambda w, x: pdf_workflow_parallel(w, x, **workflow_cfg) ** 2

    minval = kwargs.get("minval")
    maxval = kwargs.get("maxval")
    points = kwargs.get("points")

    x_integral = np.linspace(minval, maxval, points)
    domain_x = np.array(list(product(*[x_integral[:, i] for i in range(x_integral.shape[1])])))

    cdf_train = workflow_execution_parallel(weights, data_x, cdf_workflow_, dask_client=dask_client)
    pdf_train = workflow_execution_parallel(weights, data_x, pdf_workflow_, dask_client=dask_client)
    pdf_square = workflow_execution_parallel(weights, domain_x, pdf_square_workflow_, dask_client=dask_client)

    integral = compute_integral(pdf_square, domain_x, dask_client=dask_client)

    if dask_client is None:
        cdf_train = np.array(cdf_train)
        pdf_train = np.array(pdf_train)
    else:
        cdf_train = np.array(dask_client.gather(cdf_train))
        pdf_train = np.array(dask_client.gather(pdf_train))
        integral = dask_client.gather(integral)

    return {
        "y_predict_cdf": cdf_train.reshape((-1, 1)),
        "y_predict_pdf": pdf_train.reshape((-1, 1)),
        "integral": integral,
        "data_y": data_y,
    }


def qdml_loss_workflow_parallel(weights, data_x, data_y, dask_client=None, **kwargs):
    """
    Compute qdml loss using parallel-friendly workflow.
    """
    loss_weights = kwargs.get("loss_weights", [1.0, 5.0])
    output = workflow_for_qdml_parallel(weights, data_x, data_y, dask_client=dask_client, **kwargs)
    return loss_function_qdml(
        output.get("data_y"),
        output.get("y_predict_cdf"),
        output.get("y_predict_pdf"),
        output.get("integral"),
        loss_weights=loss_weights,
    )


def mse_workflow_parallel(weights, data_x, data_y, dask_client=None, **kwargs):
    """
    Compute MSE via parallel-friendly CDF workflow.
    """
    output = workflow_for_cdf_parallel(weights, data_x, dask_client=dask_client, **kwargs)
    return mse(data_y, output["y_predict_cdf"])


def unsupervised_qdml_loss_workflow_parallel(
    weights,
    data_x,
    dask_client=None,
    empirical_shift=-0.5,
    **kwargs,
):
    """
    Unsupervised qdml loss: labels from empirical CDF + shift.
    """
    data_x = np.asarray(data_x)
    if data_x.ndim == 1:
        data_x = data_x.reshape((-1, 1))
    data_y = empirical_cdf(data_x).reshape((-1, 1)) + empirical_shift
    return qdml_loss_workflow_parallel(weights, data_x, data_y, dask_client=dask_client, **kwargs)


def dft_from_trained_pqc_parallel(
    weights,
    minval=-2.0 * np.pi,
    maxval=2.0 * np.pi,
    points=256,
    prediction="cdf",
    dask_client=None,
    **kwargs,
):
    """
    Compute DFT coefficients from direct evaluations of a trained PQC.
    """
    if points < 2:
        raise ValueError("points must be >= 2")
    if maxval <= minval:
        raise ValueError("maxval must be greater than minval")

    features_names = kwargs.get("features_names")
    if features_names is None:
        raise ValueError("features_names should be provided in kwargs")
    if len(features_names) != 1:
        raise ValueError("dft_from_trained_pqc_parallel currently supports only 1 feature")

    x_domain = np.linspace(minval, maxval, points, endpoint=False)
    data_x = x_domain.reshape((-1, 1))

    if prediction == "cdf":
        y_predict = workflow_for_cdf_parallel(weights, data_x, dask_client=dask_client, **kwargs)[
            "y_predict_cdf"
        ]
    elif prediction == "pdf":
        y_predict = workflow_for_pdf_parallel(weights, data_x, dask_client=dask_client, **kwargs)[
            "y_predict_pdf"
        ]
    else:
        raise ValueError("prediction should be either 'cdf' or 'pdf'")

    y_predict = np.asarray(y_predict).reshape((-1,))
    c_k = np.fft.fft(y_predict) / points
    k_values = np.fft.fftfreq(points, d=1.0 / points).astype(int)
    order = np.argsort(k_values)

    return {
        "x_domain": x_domain,
        "y_predict": y_predict,
        "k_values": k_values[order],
        "c_k": c_k[order],
    }
