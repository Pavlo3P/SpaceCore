from __future__ import annotations

from enum import Enum, auto
from functools import cached_property
from math import prod
from typing import Any

from ._base import LinOp
from ._metric import (
    _metric_is_hermitian_by_basis,
    _requires_euclidean_or_riesz,
    metric_rapply,
    metric_rvapply,
)
from .._batching import _check_batched
from ..backend import Context, jax_pytree_class
from ..space import Space, VectorSpace, WeightedInnerProduct
from ..types import DenseArray
from .._contextual import resolve_context_priority


class _DiagonalMode(Enum):
    """Private computation modes for diagonal coordinate operators."""

    EUCLIDEAN = auto()
    WEIGHTED_FUSED = auto()
    GENERAL_METRIC = auto()


@jax_pytree_class
class DiagonalLinOp(LinOp[Space, Space]):
    r"""
    Represent a coordinatewise diagonal linear operator.

    ``DiagonalLinOp(diagonal, space)`` maps ``x`` to ``diagonal * x`` in
    coordinates. The adjoint is metric-aware: Euclidean spaces use the complex
    conjugate of the diagonal, while non-Euclidean spaces use their Riesz maps
    as ``R_X^{-1} D^dagger R_X``.

    Parameters
    ----------
    diagonal : DenseArray
        Dense backend array with shape ``space.shape``.
    space : Space or None, optional
        Domain and codomain space. If omitted, a vector space is inferred from
        ``diagonal.shape``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``space``.

    Attributes
    ----------
    diagonal : DenseArray
        Stored diagonal values.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.VectorSpace((2,), ctx)
    >>> D = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0]), X, ctx)
    >>> D.apply(ctx.asarray([4.0, 5.0]))
    array([ 8., 15.])
    """

    def __init__(
        self,
        diagonal: DenseArray,
        space: Space | None = None,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = resolve_context_priority(ctx, space)
        ctx.assert_dense(diagonal)
        if space is None:
            space = VectorSpace(tuple(diagonal.shape), ctx)
        _requires_euclidean_or_riesz(space, space, "DiagonalLinOp")
        super().__init__(space, space, ctx)
        expected = tuple(self.domain.shape)
        if tuple(diagonal.shape) != expected:
            raise TypeError(f"Expected diagonal.shape == space.shape == {expected}, got {diagonal.shape}")
        self.diagonal = diagonal
        self._diag_flat = diagonal.reshape((prod(self.domain.shape),))
        dtype = self.ops.get_dtype(diagonal)
        self._diag_adjoint = (
            self.ops.conj(diagonal) if self.ops.is_complex_dtype(dtype) else diagonal
        )
        self._diag_adjoint_flat = self._diag_adjoint.reshape((prod(self.domain.shape),))
        self._mode = self._select_mode()

    def _select_mode(self) -> _DiagonalMode:
        """Select the diagonal computation mode once for this operator."""
        if type(self.domain) is VectorSpace and self.domain.is_euclidean:
            return _DiagonalMode.EUCLIDEAN
        if type(self.domain) is VectorSpace and type(self.domain.geometry) is WeightedInnerProduct:
            return _DiagonalMode.WEIGHTED_FUSED
        return _DiagonalMode.GENERAL_METRIC

    @cached_property
    def A(self) -> DenseArray:
        """Dense tensor representation of this diagonal operator."""
        return self.to_dense()

    def apply(self, x: DenseArray) -> DenseArray:
        """Apply the diagonal operator to ``x``."""
        checks = self._enable_checks
        if checks:
            self.domain._check_member(x)
        y = self._apply_core(x)
        if checks:
            self.codomain._check_member(y)
        return y

    def _apply_core(self, x: DenseArray) -> DenseArray:
        """Apply the diagonal operator without membership checks."""
        if self._mode is _DiagonalMode.EUCLIDEAN:
            return self.diagonal * x
        if self._mode is _DiagonalMode.WEIGHTED_FUSED:
            return self.diagonal * x
        if type(self.domain) is VectorSpace:
            return self.diagonal * x
        x_flat = self.domain.flatten(x)
        y_flat = self._diag_flat * x_flat
        return self.codomain.unflatten(y_flat)

    def rapply(self, y: DenseArray) -> DenseArray:
        """Apply the adjoint diagonal operator to ``y``."""
        checks = self._enable_checks
        if checks:
            self.codomain._check_member(y)
        x = self._rapply_core(y)
        if checks:
            self.domain._check_member(x)
        return x

    def _rapply_core(self, y: DenseArray) -> DenseArray:
        """Apply the metric adjoint without membership checks."""
        if self._mode is _DiagonalMode.EUCLIDEAN:
            return self._euclidean_rapply_core(y)
        if self._mode is _DiagonalMode.WEIGHTED_FUSED:
            return self._diag_adjoint * y
        return metric_rapply(self.domain, self.codomain, self._euclidean_rapply_core, y)

    def _euclidean_rapply_core(self, y: DenseArray) -> DenseArray:
        """Apply the Euclidean diagonal adjoint without membership checks."""
        if self._mode is _DiagonalMode.EUCLIDEAN:
            return self._diag_adjoint * y
        if self._mode is _DiagonalMode.WEIGHTED_FUSED:
            return self._diag_adjoint * y
        if type(self.domain) is VectorSpace:
            return self._diag_adjoint * y
        y_flat = self.codomain.flatten(y)
        x_flat = self._diag_adjoint_flat * y_flat
        return self.domain.unflatten(x_flat)

    def vapply(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        checks = self._enable_checks
        if checks:
            _check_batched(self.domain, xs)
        return self._vapply_core(xs)

    def _vapply_core(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis without membership checks."""
        if self._mode is _DiagonalMode.EUCLIDEAN:
            return self.diagonal * xs
        if self._mode is _DiagonalMode.WEIGHTED_FUSED:
            return self.diagonal * xs
        if type(self.domain) is VectorSpace:
            return self.diagonal * xs
        xs_flat = self.domain.flatten_batch(xs)
        ys_flat = xs_flat * self._diag_flat
        return self.codomain.unflatten_batch(ys_flat)

    def rvapply(self, ys: DenseArray) -> DenseArray:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
        checks = self._enable_checks
        if checks:
            _check_batched(self.codomain, ys)
        xs = self._rvapply_core(ys)
        if checks:
            _check_batched(self.domain, xs)
        return xs

    def _rvapply_core(self, ys: DenseArray) -> DenseArray:
        """Apply the metric adjoint over a leading batch axis without checks."""
        if self._mode is _DiagonalMode.EUCLIDEAN:
            return self._euclidean_rvapply_core(ys)
        if self._mode is _DiagonalMode.WEIGHTED_FUSED:
            return self._diag_adjoint * ys
        return metric_rvapply(
            self.domain,
            self.codomain,
            self._euclidean_rapply_core,
            self._euclidean_rvapply_core,
            ys,
            opname=type(self).__name__,
            ops=self.ops,
        )

    def _euclidean_rvapply_core(self, ys: DenseArray) -> DenseArray:
        """Apply the Euclidean diagonal adjoint over a leading batch axis."""
        if self._mode is _DiagonalMode.EUCLIDEAN:
            return self._diag_adjoint * ys
        if self._mode is _DiagonalMode.WEIGHTED_FUSED:
            return self._diag_adjoint * ys
        if type(self.domain) is VectorSpace:
            return self._diag_adjoint * ys
        ys_flat = self.codomain.flatten_batch(ys)
        xs_flat = ys_flat * self._diag_adjoint_flat
        return self.domain.unflatten_batch(xs_flat)

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
        if not self.domain.is_euclidean:
            return _metric_is_hermitian_by_basis(self)
        try:
            return bool(self.ops.allclose(self.diagonal, self._diag_adjoint))
        except Exception:
            return None

    def __eq__(self, other: Any) -> bool:
        """Return whether another diagonal operator has the same space and values."""
        if type(other) is type(self):
            return self.domain == other.domain and self.ops.allclose(self.diagonal, other.diagonal)
        return False

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        children = (self.diagonal,)
        aux = (self.domain, self.ctx, self._mode)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        if len(aux) == 3:
            domain, ctx, _mode = aux
        else:
            domain, ctx = aux
        return cls(children[0], domain, ctx)

    def _convert(self, new_ctx: Context) -> DiagonalLinOp:
        """Convert the stored diagonal and space to ``new_ctx``."""
        return DiagonalLinOp(
            new_ctx.asarray(self.diagonal),
            self.domain.convert(new_ctx),
            new_ctx,
        )
