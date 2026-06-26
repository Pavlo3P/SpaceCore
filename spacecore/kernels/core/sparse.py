"""Concrete core kernels for :class:`spacecore.linop.SparseLinOp`.

A sparse coordinate operator stores a sparse matrix (and its Hermitian adjoint)
and applies them in Euclidean coordinates; the adjoint is metric-aware. The mode
is chosen once at construction (:class:`_SparseMode`); the per-apply cores branch
on ``op._mode`` plus the cached matrices and (for the weighted-fused mode) the
diagonal metric weights.

Metric helpers are imported lazily (general-metric path only), so this module has
no module-level dependency on :mod:`spacecore.linop` and no import cycle forms.

No ADR-016 dispatch spec is wired for ``SparseLinOp`` (the ``linop.matvec.sparse``
key stays reserved/inert). Every direction is already a *single* optimal backend
call: ``apply``/``rapply`` are one SpMV (``op._A @ x`` / ``op._AH @ y``) and
``vapply``/``rvapply`` are one batched SpMV over the stacked right-hand side
(``(op._A @ xs.reshape((-1, n)).T).T``), never an ``O(batch)`` Python loop. There
is no faster bit-exact path to route to, so a dispatch spec would add only the
``applicable``/walk overhead for no win — exactly the over-fitting ADR-016's
profitability rule forbids. (A future format- or device-aware SpMV variant, if
one is ever benchmarked to win, would register under that reserved key.)
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
    if op._mode is _SparseMode.EUCLIDEAN_FLAT:
        return op._AH @ y
    if op._mode is _SparseMode.EUCLIDEAN_TENSOR:
        return (op._AH @ y.reshape((op._cod_size,))).reshape(op.dom.shape)
    y1 = op.cod.flatten(y)
    x1 = op._AH @ y1
    return op.dom.unflatten(x1)


def sparse_rapply_core(op: Any, y: Any) -> Any:
    if op._mode is _SparseMode.EUCLIDEAN_FLAT or op._mode is _SparseMode.EUCLIDEAN_TENSOR:
        return sparse_euclidean_rapply_core(op, y)
    if op._mode is _SparseMode.WEIGHTED_FUSED:
        return (op._AH @ (op._cod_weights * y)) / op._dom_weights
    from ...linop._metric import metric_rapply

    return metric_rapply(
        op.domain, op.codomain, lambda yy: sparse_euclidean_rapply_core(op, yy), y
    )


def sparse_vapply_core(op: Any, xs: Any) -> Any:
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
