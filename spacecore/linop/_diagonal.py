from __future__ import annotations

from functools import cached_property
from math import prod
from typing import Any, cast

from ._base import LinOp
from ._metric import _metric_is_hermitian_by_basis, _requires_euclidean_or_riesz
from .._checks import checked_method
from ..backend import Context, jax_pytree_class
from ..space import (
    CoordinateSpace,
    DenseCoordinateSpace,
    DenseVectorSpace,
    ElementwiseJordanSpace,
    WeightedInnerProduct,
)
from ..types import DenseArray
from .._contextual import resolve_context_priority
from ..kernels import core_kernels
from ..kernels.core.diagonal import _DiagonalMode


@core_kernels("diagonal")
@jax_pytree_class
class DiagonalLinOp(LinOp[CoordinateSpace, CoordinateSpace]):
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
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> D = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0]), X, ctx)
    >>> D.apply(ctx.asarray([4.0, 5.0]))
    array([ 8., 15.])
    """

    def __init__(
        self,
        diagonal: DenseArray,
        space: CoordinateSpace | None = None,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = resolve_context_priority(ctx, space)
        ctx.assert_dense(diagonal)
        if space is None:
            space = DenseCoordinateSpace(tuple(diagonal.shape), ctx)
        _requires_euclidean_or_riesz(space, space, "DiagonalLinOp")
        super().__init__(space, space, ctx)
        expected = tuple(self.domain.shape)
        if tuple(diagonal.shape) != expected:
            raise TypeError(
                f"Expected diagonal.shape == space.shape == {expected}, got {diagonal.shape}"
            )
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
        if (
            type(self.domain) in (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace)
        ) and cast(Any, self.domain).is_euclidean:
            return _DiagonalMode.EUCLIDEAN
        if (
            type(self.domain) in (DenseCoordinateSpace, DenseVectorSpace, ElementwiseJordanSpace)
        ) and type(cast(Any, self.domain).geometry) is WeightedInnerProduct:
            return _DiagonalMode.WEIGHTED_FUSED
        return _DiagonalMode.GENERAL_METRIC

    @cached_property
    def A(self) -> DenseArray:
        """Dense tensor representation of this diagonal operator."""
        return self.to_dense()

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: DenseArray) -> DenseArray:
        """Apply the diagonal operator to ``x``."""
        return self._apply_core(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: DenseArray) -> DenseArray:
        """Apply the adjoint diagonal operator to ``y``."""
        return self._rapply_core(y)

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        return self._vapply_core(xs)

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, ys: DenseArray) -> DenseArray:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
        return self._rvapply_core(ys)

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
        if not cast(Any, self.domain).is_euclidean:
            return _metric_is_hermitian_by_basis(self)
        try:
            return bool(self.ops.allclose(self.diagonal, self._diag_adjoint))
        except Exception:
            return None

    def __eq__(self, other: Any) -> bool:
        """Return whether another diagonal operator has the same space and values."""
        if not self._eq_backend_compatible(other):                  # Tier 1: backend
            return NotImplemented
        if self.domain != other.domain:                             # Tier 2: space before allclose
            return False
        return bool(self.ops.allclose(self.diagonal, other.diagonal, equal_nan=True))  # Tier 3

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        children = (self.diagonal,)
        aux = (self.domain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        domain, ctx = aux
        return cls(children[0], domain, ctx)

    def _convert(self, new_ctx: Context) -> DiagonalLinOp:
        """Convert the stored diagonal and space to ``new_ctx``."""
        return DiagonalLinOp(
            new_ctx.asarray(self.diagonal),
            self.domain.convert(new_ctx),
            new_ctx,
        )
