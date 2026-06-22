"""Concrete core kernels for :class:`spacecore.linop.DenseLinOp`.

A dense coordinate operator stores a flattened matrix and applies it directly in
Euclidean coordinates; its adjoint is metric-aware. The computation strategy is
chosen once at construction (:class:`_DenseMode`) from the operand spaces, so the
per-apply cores below only branch on the precomputed ``op._mode`` plus the cached
matrices (``op._A2``, ``op._A2H``, ``op._A2T``, ``op._weighted_A2H``).

The functions are duck-typed on the operator and import the metric helpers
*lazily* (only on the general-metric / Riesz path, which is the non-hot branch),
so this module has no module-level dependency on :mod:`spacecore.linop` and no
import cycle forms.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any

from ._rules import CoreKernelSet, register_core_kernels


class _DenseMode(Enum):
    """Private computation modes for dense coordinate operators."""

    EUCLIDEAN_FLAT = auto()
    EUCLIDEAN_TENSOR = auto()
    WEIGHTED_FUSED = auto()
    GENERAL_METRIC = auto()


def dense_apply_core(op: Any, x: Any) -> Any:
    if op._mode is _DenseMode.EUCLIDEAN_FLAT:
        return op._A2 @ x
    if op._mode is _DenseMode.EUCLIDEAN_TENSOR:
        return (op._A2 @ x.reshape((op._dom_size,))).reshape(op.cod.shape)
    if op._mode is _DenseMode.WEIGHTED_FUSED:
        return op._A2 @ x
    x1 = op.dom.flatten(x)
    y1 = op._A2 @ x1
    return op.cod.unflatten(y1)


def dense_euclidean_rapply_core(op: Any, y: Any) -> Any:
    if op._mode is _DenseMode.EUCLIDEAN_FLAT:
        return op._A2H @ y
    if op._mode is _DenseMode.EUCLIDEAN_TENSOR:
        return (op._A2H @ y.reshape((op._cod_size,))).reshape(op.dom.shape)
    y1 = op.cod.flatten(y)
    x1 = op._A2H @ y1
    return op.dom.unflatten(x1)


def dense_rapply_core(op: Any, y: Any) -> Any:
    if op._mode is _DenseMode.EUCLIDEAN_FLAT or op._mode is _DenseMode.EUCLIDEAN_TENSOR:
        return dense_euclidean_rapply_core(op, y)
    if op._mode is _DenseMode.WEIGHTED_FUSED:
        return op._weighted_A2H @ y
    from ...linop._metric import metric_rapply

    return metric_rapply(
        op.domain, op.codomain, lambda yy: dense_euclidean_rapply_core(op, yy), y
    )


def dense_vapply_core(op: Any, xs: Any) -> Any:
    if op._mode is _DenseMode.EUCLIDEAN_FLAT:
        return xs.reshape((-1, op._dom_size)) @ op._A2T
    if op._mode is _DenseMode.EUCLIDEAN_TENSOR:
        lead = tuple(xs.shape[: len(xs.shape) - len(op.dom.shape)])
        xs2 = xs.reshape((-1, op._dom_size))
        ys2 = xs2 @ op._A2T
        return ys2.reshape(lead + tuple(op.cod.shape))
    if op._mode is _DenseMode.WEIGHTED_FUSED:
        return xs.reshape((-1, op._dom_size)) @ op._A2T
    xs_flat = op.domain.flatten_batch(xs)
    ys_flat = xs_flat @ op._A2T
    return op.codomain.unflatten_batch(ys_flat)


def dense_euclidean_rvapply_core(op: Any, ys: Any) -> Any:
    if op._mode is _DenseMode.EUCLIDEAN_FLAT:
        return ys.reshape((-1, op._cod_size)) @ op._A2H.T
    if op._mode is _DenseMode.EUCLIDEAN_TENSOR:
        lead = tuple(ys.shape[: len(ys.shape) - len(op.cod.shape)])
        ys2 = ys.reshape((-1, op._cod_size))
        xs2 = ys2 @ op._A2H.T
        return xs2.reshape(lead + tuple(op.dom.shape))
    ys_flat = op.codomain.flatten_batch(ys)
    xs_flat = ys_flat @ op._A2H.T
    return op.domain.unflatten_batch(xs_flat)


def dense_rvapply_core(op: Any, ys: Any) -> Any:
    if op._mode is _DenseMode.EUCLIDEAN_FLAT or op._mode is _DenseMode.EUCLIDEAN_TENSOR:
        return dense_euclidean_rvapply_core(op, ys)
    if op._mode is _DenseMode.WEIGHTED_FUSED:
        return ys @ op._weighted_A2H.T
    from ...linop._metric import metric_rvapply

    return metric_rvapply(
        op.domain,
        op.codomain,
        lambda yy: dense_euclidean_rapply_core(op, yy),
        lambda yy: dense_euclidean_rvapply_core(op, yy),
        ys,
        opname=type(op).__name__,
        ops=op.ops,
    )


register_core_kernels(CoreKernelSet(
    "dense", dense_apply_core, dense_rapply_core, dense_vapply_core, dense_rvapply_core,
    notes="Flattened dense matrix action; metric-aware adjoint via Riesz maps.",
))
