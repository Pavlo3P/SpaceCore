"""
Core kernels for :class:`spacecore.linop.SparseLinOp`.

Each direction is one SpMV, or one batched SpMV over the stacked right-hand sides,
with the metric mode fixed at construction. Metric helpers are imported lazily to
avoid an import cycle.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any

from ._rules import CoreKernelSet, register_core_kernels


class _SparseMode(Enum):
    """Private computation modes for sparse coordinate operators."""

    EUCLIDEAN_FLAT = auto()
    EUCLIDEAN_TENSOR = auto()
    WEIGHTED_FUSED = auto()
    GENERAL_METRIC = auto()


def sparse_apply_core(op: Any, x: Any) -> Any:
    # Flat or fused: one matvec. Tensor: reshape around it. Else the spaces' flatten/unflatten.
    if op._mode is _SparseMode.EUCLIDEAN_FLAT:
        return op._A @ x
    if op._mode is _SparseMode.EUCLIDEAN_TENSOR:
        return (op._A @ x.reshape((op._dom_size,))).reshape(op.cod.shape)
    if op._mode is _SparseMode.WEIGHTED_FUSED:
        return op._A @ x
    x1 = op.dom.flatten(x)
    y1 = op._A @ x1
    return op.cod.unflatten(y1)


def sparse_euclidean_rapply_core(op: Any, y: Any) -> Any:
    # Plain conjugate transpose A^H, no metric.
    if op._mode is _SparseMode.EUCLIDEAN_FLAT:
        return op._AH @ y
    if op._mode is _SparseMode.EUCLIDEAN_TENSOR:
        return (op._AH @ y.reshape((op._cod_size,))).reshape(op.dom.shape)
    y1 = op.cod.flatten(y)
    x1 = op._AH @ y1
    return op.dom.unflatten(x1)


def sparse_rapply_core(op: Any, y: Any) -> Any:
    # Euclidean: A^H. Diagonal weights: W_X^{-1} A^H W_Y y. Else wrap A^H with the metrics.
    if op._mode is _SparseMode.EUCLIDEAN_FLAT or op._mode is _SparseMode.EUCLIDEAN_TENSOR:
        return sparse_euclidean_rapply_core(op, y)
    if op._mode is _SparseMode.WEIGHTED_FUSED:
        return (op._AH @ (op._cod_weights * y)) / op._dom_weights
    from ...linop._metric import metric_rapply

    return metric_rapply(
        op.domain, op.codomain, lambda yy: sparse_euclidean_rapply_core(op, yy), y
    )


def sparse_vapply_core(op: Any, xs: Any) -> Any:
    # Sparse @ dense wants the batch as columns: (A X^T)^T.
    if op._mode is _SparseMode.EUCLIDEAN_FLAT:
        return (op._A @ xs.reshape((-1, op._dom_size)).T).T
    if op._mode is _SparseMode.EUCLIDEAN_TENSOR:
        lead = tuple(xs.shape[: len(xs.shape) - len(op.dom.shape)])
        xs2 = xs.reshape((-1, op._dom_size))
        ys2 = (op._A @ xs2.T).T
        return ys2.reshape(lead + tuple(op.cod.shape))
    if op._mode is _SparseMode.WEIGHTED_FUSED:
        return (op._A @ xs.reshape((-1, op._dom_size)).T).T
    xs_flat = op.domain.flatten_batch(xs)
    ys_flat = (op._A @ xs_flat.T).T
    return op.codomain.unflatten_batch(ys_flat)


def sparse_euclidean_rvapply_core(op: Any, ys: Any) -> Any:
    # (A^H Y^T)^T for a batch of rows.
    if op._mode is _SparseMode.EUCLIDEAN_FLAT:
        return (op._AH @ ys.reshape((-1, op._cod_size)).T).T
    if op._mode is _SparseMode.EUCLIDEAN_TENSOR:
        lead = tuple(ys.shape[: len(ys.shape) - len(op.cod.shape)])
        ys2 = ys.reshape((-1, op._cod_size))
        xs2 = (op._AH @ ys2.T).T
        return xs2.reshape(lead + tuple(op.dom.shape))
    ys_flat = op.codomain.flatten_batch(ys)
    xs_flat = (op._AH @ ys_flat.T).T
    return op.domain.unflatten_batch(xs_flat)


def sparse_rvapply_core(op: Any, ys: Any) -> Any:
    # Batched form of sparse_rapply_core.
    if op._mode is _SparseMode.EUCLIDEAN_FLAT or op._mode is _SparseMode.EUCLIDEAN_TENSOR:
        return sparse_euclidean_rvapply_core(op, ys)
    if op._mode is _SparseMode.WEIGHTED_FUSED:
        return (op._AH @ (ys * op._cod_weights).T).T / op._dom_weights
    from ...linop._metric import metric_rvapply

    return metric_rvapply(
        op.domain,
        op.codomain,
        lambda yy: sparse_euclidean_rapply_core(op, yy),
        lambda yy: sparse_euclidean_rvapply_core(op, yy),
        ys,
        opname=type(op).__name__,
        ops=op.ops,
    )


register_core_kernels(CoreKernelSet(
    "sparse", sparse_apply_core, sparse_rapply_core,
    sparse_vapply_core, sparse_rvapply_core,
    notes="Sparse matrix action; metric-aware adjoint via Riesz maps.",
))
