"""Concrete core kernels for :class:`spacecore.linop.DiagonalLinOp`.

A diagonal operator multiplies coordinatewise by a stored diagonal; its adjoint
is metric-aware. As with the dense kernels, the computation mode is chosen once
at construction (:class:`_DiagonalMode`) and the per-apply cores branch on
``op._mode`` plus the cached diagonals.

Metric helpers are imported lazily (general-metric path only) so this module has
no module-level dependency on :mod:`spacecore.linop`. The vector-space type check
needs the concrete coordinate-space classes, imported from :mod:`spacecore.space`
(which does not depend on :mod:`spacecore.kernels`, so no cycle forms).
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any

from ...space import DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace
from ._rules import CoreKernelSet, register_core_kernels

_VECTOR_SPACES = (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace)


class _DiagonalMode(Enum):
    """Private computation modes for diagonal coordinate operators."""

    EUCLIDEAN = auto()
    WEIGHTED_FUSED = auto()
    GENERAL_METRIC = auto()


def diagonal_apply_core(op: Any, x: Any) -> Any:
    if op._mode is _DiagonalMode.EUCLIDEAN:
        return op.diagonal * x
    if op._mode is _DiagonalMode.WEIGHTED_FUSED:
        return op.diagonal * x
    if type(op.domain) in _VECTOR_SPACES:
        return op.diagonal * x
    x_flat = op.domain.flatten(x)
    y_flat = op._diag_flat * x_flat
    return op.codomain.unflatten(y_flat)


def diagonal_euclidean_rapply_core(op: Any, y: Any) -> Any:
    if op._mode is _DiagonalMode.EUCLIDEAN:
        return op._diag_adjoint * y
    if op._mode is _DiagonalMode.WEIGHTED_FUSED:
        return op._diag_adjoint * y
    if type(op.domain) in _VECTOR_SPACES:
        return op._diag_adjoint * y
    y_flat = op.codomain.flatten(y)
    x_flat = op._diag_adjoint_flat * y_flat
    return op.domain.unflatten(x_flat)


def diagonal_rapply_core(op: Any, y: Any) -> Any:
    if op._mode is _DiagonalMode.EUCLIDEAN:
        return diagonal_euclidean_rapply_core(op, y)
    if op._mode is _DiagonalMode.WEIGHTED_FUSED:
        return op._diag_adjoint * y
    from ...linop._metric import metric_rapply

    return metric_rapply(
        op.domain, op.codomain, lambda yy: diagonal_euclidean_rapply_core(op, yy), y
    )


def diagonal_vapply_core(op: Any, xs: Any) -> Any:
    if op._mode is _DiagonalMode.EUCLIDEAN:
        return op.diagonal * xs
    if op._mode is _DiagonalMode.WEIGHTED_FUSED:
        return op.diagonal * xs
    if type(op.domain) in _VECTOR_SPACES:
        return op.diagonal * xs
    xs_flat = op.domain.flatten_batch(xs)
    ys_flat = xs_flat * op._diag_flat
    return op.codomain.unflatten_batch(ys_flat)


def diagonal_euclidean_rvapply_core(op: Any, ys: Any) -> Any:
    if op._mode is _DiagonalMode.EUCLIDEAN:
        return op._diag_adjoint * ys
    if op._mode is _DiagonalMode.WEIGHTED_FUSED:
        return op._diag_adjoint * ys
    if type(op.domain) in _VECTOR_SPACES:
        return op._diag_adjoint * ys
    ys_flat = op.codomain.flatten_batch(ys)
    xs_flat = ys_flat * op._diag_adjoint_flat
    return op.domain.unflatten_batch(xs_flat)


def diagonal_rvapply_core(op: Any, ys: Any) -> Any:
    if op._mode is _DiagonalMode.EUCLIDEAN:
        return diagonal_euclidean_rvapply_core(op, ys)
    if op._mode is _DiagonalMode.WEIGHTED_FUSED:
        return op._diag_adjoint * ys
    from ...linop._metric import metric_rvapply

    return metric_rvapply(
        op.domain,
        op.codomain,
        lambda yy: diagonal_euclidean_rapply_core(op, yy),
        lambda yy: diagonal_euclidean_rvapply_core(op, yy),
        ys,
        opname=type(op).__name__,
        ops=op.ops,
    )


register_core_kernels(CoreKernelSet(
    "diagonal", diagonal_apply_core, diagonal_rapply_core,
    diagonal_vapply_core, diagonal_rvapply_core,
    notes="Coordinatewise diagonal action; metric-aware adjoint via Riesz maps.",
))
