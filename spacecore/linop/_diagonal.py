from __future__ import annotations

from enum import Enum, auto
from functools import cached_property
from math import prod
from typing import Any, cast

from ._base import LinOp, _output_checked_rvapply
from ._metric import (
    _VECTOR_SPACES, _metric_is_hermitian_by_basis, _requires_euclidean_or_riesz, metric_rapply,
    metric_rvapply,
)
from .._checks import CheckLevel
from ..contextual import Context
from ..space import (
    CoordinateSpace,
    DenseCoordinateSpace,
    WeightedInnerProduct,
)
from ..types import DenseArray
from ..contextual import resolve_context_priority


class _DiagonalMode(Enum):
    """Private computation modes for diagonal coordinate operators."""

    EUCLIDEAN = auto()
    WEIGHTED_FUSED = auto()
    GENERAL_METRIC = auto()



class DiagonalLinOp(LinOp[CoordinateSpace, CoordinateSpace]):
    r"""
    Coordinatewise map ``x -> diagonal * x`` on one space.

    The adjoint conjugates the diagonal on Euclidean spaces and is
    ``R_X^{-1} D^dagger R_X`` otherwise.

    Parameters
    ----------
    diagonal : DenseArray
        Diagonal of shape ``space.shape``.
    space : Space or None, optional
        Domain and codomain; a vector space of ``diagonal.shape`` when omitted.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``space`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> D = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0]), X, ctx)
    >>> D.apply(ctx.asarray([4.0, 5.0]))
    array([ 8., 15.])
    """

    _fields = ("diagonal", "space:domain", "ctx")
    _dynamic = ("diagonal",)

    def _apply_core(self, x: Any) -> Any:
        # Vector-shaped or fused: elementwise product. Else on flat coordinates.
        # EUCLIDEAN and WEIGHTED_FUSED imply a vector-space domain; check them first.
        if self._elementwise:
            return self.diagonal * x
        x_flat = self.domain.flatten(x)
        y_flat = self._diag_flat * x_flat
        return self.codomain.unflatten(y_flat)

    def _rapply_core(self, y: Any) -> Any:
        # Euclidean: conj diagonal. Fused: precomputed. Else wrap it with the metrics.
        if self._mode is _DiagonalMode.EUCLIDEAN:
            return self._euclidean_rapply_core(y)
        if self._mode is _DiagonalMode.WEIGHTED_FUSED:
            return self._diag_adjoint * y
        return metric_rapply(
            self.domain, self.codomain, self._euclidean_rapply_core, y
        )

    def _vapply_core(self, xs: Any) -> Any:
        # Elementwise, broadcasting over the leading batch axis.
        if self._elementwise:
            return self.diagonal * xs
        xs_flat = self.domain.flatten_batch(xs)
        ys_flat = xs_flat * self._diag_flat
        return self.codomain.unflatten_batch(ys_flat)

    def _rvapply_core(self, ys: Any) -> Any:
        # Batched form of _rapply_core.
        if self._mode is _DiagonalMode.EUCLIDEAN:
            return self._euclidean_rvapply_core(ys)
        if self._mode is _DiagonalMode.WEIGHTED_FUSED:
            return self._diag_adjoint * ys
        return metric_rvapply(self.domain, self.codomain, self._euclidean_rvapply_core, ys)

    def _euclidean_rapply_core(self, y: Any) -> Any:
        # The adjoint diagonal, elementwise or on flat coordinates.
        if self._elementwise:
            return self._diag_adjoint * y
        y_flat = self.codomain.flatten(y)
        x_flat = self._diag_adjoint_flat * y_flat
        return self.domain.unflatten(x_flat)

    def _euclidean_rvapply_core(self, ys: Any) -> Any:
        # Adjoint diagonal over the batch.
        if self._elementwise:
            return self._diag_adjoint * ys
        ys_flat = self.codomain.flatten_batch(ys)
        xs_flat = ys_flat * self._diag_adjoint_flat
        return self.domain.unflatten_batch(xs_flat)

    def __init__(
        self,
        diagonal: DenseArray,
        space: CoordinateSpace | None = None,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        # Infer a Euclidean space from the diagonal's shape if none is given.
        ctx = resolve_context_priority(ctx, space)
        ctx.assert_dense(diagonal)
        if space is None:
            space = DenseCoordinateSpace(tuple(diagonal.shape), ctx)
        _requires_euclidean_or_riesz(space, space, "DiagonalLinOp")
        super().__init__(space, space, ctx, check_level=check_level)
        expected = tuple(self.domain.shape)
        if tuple(diagonal.shape) != expected:
            raise TypeError(
                f"Expected diagonal.shape == space.shape == {expected}, got {diagonal.shape}"
            )
        # Cache flat and conjugated copies for the cores, then pick the strategy.
        self.diagonal = diagonal
        self._diag_flat = diagonal.reshape((prod(self.domain.shape),))
        dtype = self.ops.get_dtype(diagonal)
        self._diag_adjoint = (
            self.ops.conj(diagonal) if self.ops.is_complex_dtype(dtype) else diagonal
        )
        self._diag_adjoint_flat = self._diag_adjoint.reshape((prod(self.domain.shape),))
        self._elementwise = type(self.domain) in _VECTOR_SPACES
        self._mode = self._select_mode()

    def _select_mode(self) -> _DiagonalMode:
        """Select the diagonal computation mode once for this operator."""
        # Euclidean: multiply by the diagonal; weighted: fused path; else general Riesz maps.
        if self._elementwise and cast(Any, self.domain).is_euclidean:
            return _DiagonalMode.EUCLIDEAN
        if self._elementwise and type(cast(Any, self.domain).geometry) is WeightedInnerProduct:
            return _DiagonalMode.WEIGHTED_FUSED
        return _DiagonalMode.GENERAL_METRIC

    @cached_property
    def A(self) -> DenseArray:
        """Dense tensor representation of this diagonal operator."""
        return self.to_dense()

    rvapply = _output_checked_rvapply

    def to_matrix(self) -> DenseArray:
        """Return the flattened dense diagonal matrix representation."""
        return self.ops.diag(self._diag_flat)

    def to_dense(self) -> DenseArray:
        """Return a dense tensor representation of this diagonal operator."""
        matrix = self.to_matrix()
        return self.ops.reshape(matrix, tuple(self.codomain.shape) + tuple(self.domain.shape))

    def is_hermitian(self) -> bool | None:
        """
        Return whether this diagonal operator is structurally self-adjoint.

        Returns
        -------
        bool or None
            ``True`` or ``False`` when the structure can be checked, otherwise
            ``None``.
        """
        # Euclidean or diagonal-weighted: a diagonal commutes with the metric, so it is
        # Hermitian iff the diagonal is real; any other metric is tested by basis.
        domain = cast(Any, self.domain)
        if not (domain.is_euclidean or isinstance(getattr(domain, "geometry", None), WeightedInnerProduct)):
            return _metric_is_hermitian_by_basis(self)
        try:
            return bool(self.ops.allclose(self.diagonal, self._diag_adjoint))
        except Exception:
            return None
