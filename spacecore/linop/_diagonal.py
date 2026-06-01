from __future__ import annotations

from functools import cached_property
from math import prod
from typing import Any

from ._base import LinOp
from ._metric import (
    _METRIC_BATCH_FALLBACK_ERRORS,
    _metric_is_hermitian_by_basis,
    _requires_euclidean_or_riesz,
    _warn_metric_batch_fallback,
)
from .._batching import _check_batched
from .._checks import checked_method
from ..backend import Context, jax_pytree_class
from ..space import Space, VectorSpace
from ..types import DenseArray
from .._contextual import resolve_context_priority


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
        self._vector_fast_path = type(self.domain) is VectorSpace

    @cached_property
    def A(self) -> DenseArray:
        """Dense tensor representation of this diagonal operator."""
        return self.to_dense()

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: DenseArray) -> DenseArray:
        """Apply the diagonal operator to ``x``."""
        if self._vector_fast_path:
            return self.diagonal * x
        x_flat = self.domain.flatten(x)
        y_flat = self._diag_flat * x_flat
        return self.codomain.unflatten(y_flat)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: DenseArray) -> DenseArray:
        """Apply the adjoint diagonal operator to ``y``."""
        if self.domain.is_euclidean:
            return self._euclidean_rapply_unchecked(y)
        yd = self.codomain.riesz(y)
        tmp = self._euclidean_rapply_unchecked(yd)
        return self.domain.riesz_inverse(tmp)

    def _euclidean_rapply_unchecked(self, y: DenseArray) -> DenseArray:
        """Apply the Euclidean diagonal adjoint without membership checks."""
        if self._vector_fast_path:
            return self._diag_adjoint * y
        y_flat = self.codomain.flatten(y)
        x_flat = self._diag_adjoint_flat * y_flat
        return self.domain.unflatten(x_flat)

    def vapply(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        if self._enable_checks:
            _check_batched(self.domain, xs)
        if self._vector_fast_path:
            return self.diagonal * xs
        xs_flat = self.domain.flatten_batch(xs)
        ys_flat = xs_flat * self._diag_flat
        return self.codomain.unflatten_batch(ys_flat)

    def rvapply(self, ys: DenseArray) -> DenseArray:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
        if self._enable_checks:
            _check_batched(self.codomain, ys)
        if not self.domain.is_euclidean:
            try:
                yd = self.codomain.riesz(ys)
                tmp = self._euclidean_rvapply_unchecked(yd)
                xs = self.domain.riesz_inverse(tmp)
                if self._enable_checks:
                    _check_batched(self.domain, xs)
                return xs
            except _METRIC_BATCH_FALLBACK_ERRORS as err:
                _warn_metric_batch_fallback(type(self).__name__, err)
                return self.ops.vmap(self.rapply, in_axes=0, out_axes=0)(ys)
        return self._euclidean_rvapply_unchecked(ys)

    def _euclidean_rvapply_unchecked(self, ys: DenseArray) -> DenseArray:
        """Apply the Euclidean diagonal adjoint over a leading batch axis."""
        if self._vector_fast_path:
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
