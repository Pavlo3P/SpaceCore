from __future__ import annotations

from functools import cached_property
from math import prod
from typing import Any

from ._base import LinOp
from .._checks import checked_method
from ..backend import Context, jax_pytree_class
from ..space import VectorSpace
from ..types import DenseArray
from .._contextual import resolve_context_priority


@jax_pytree_class
class DiagonalLinOp(LinOp[VectorSpace, VectorSpace]):
    r"""
    Represent a coordinatewise diagonal linear operator.

    ``DiagonalLinOp(diagonal, space)`` maps ``x`` to ``diagonal * x`` on a
    :class:`VectorSpace`. The adjoint uses the complex conjugate of the
    diagonal, so complex-valued diagonals follow the SpaceCore adjoint
    convention.

    Parameters
    ----------
    diagonal : DenseArray
        Dense backend array with shape ``space.shape``.
    space : VectorSpace or None, optional
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
        space: VectorSpace | None = None,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = resolve_context_priority(ctx, space)
        ctx.assert_dense(diagonal)
        if space is None:
            space = VectorSpace(tuple(diagonal.shape), ctx)
        super().__init__(space, space, ctx)
        expected = tuple(self.domain.shape)
        if tuple(diagonal.shape) != expected:
            raise TypeError(f"Expected diagonal.shape == space.shape == {expected}, got {diagonal.shape}")
        self.diagonal = diagonal
        dtype = self.ops.get_dtype(diagonal)
        self._diag_adjoint = (
            self.ops.conj(diagonal) if self.ops.is_complex_dtype(dtype) else diagonal
        )

    @cached_property
    def A(self) -> DenseArray:
        """Dense tensor representation of this diagonal operator."""
        return self.to_dense()

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: DenseArray) -> DenseArray:
        """Apply the diagonal operator to ``x``."""
        return self.diagonal * x

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: DenseArray) -> DenseArray:
        """Apply the adjoint diagonal operator to ``y``."""
        return self._diag_adjoint * y

    def _reshape_diagonal_for_batch(self, diagonal: DenseArray, batch_space: Any) -> DenseArray:
        """Broadcast diagonal values over a batch space."""
        batch_shape = tuple(getattr(batch_space, "batch_shape", ()))
        batch_axes = tuple(getattr(batch_space, "batch_axes", ()))
        total_ndim = len(self.domain.shape) + len(batch_shape)
        base_axes = [axis for axis in range(total_ndim) if axis not in batch_axes]
        shape = [1] * total_ndim
        for axis, dim in zip(base_axes, self.domain.shape, strict=True):
            shape[axis] = dim
        return self.ops.reshape(diagonal, tuple(shape))

    def vapply(self, xs: DenseArray, batch_space=None) -> DenseArray:
        """Apply this diagonal operator over a batch of domain elements."""
        in_space = self._input_batch_space(self.domain, xs, batch_space)
        if self._enable_checks:
            in_space._check_member(xs)
        diagonal = self._reshape_diagonal_for_batch(self.diagonal, in_space)
        ys = diagonal * xs
        if self._enable_checks:
            self._output_batch_space(self.codomain, in_space)._check_member(ys)
        return ys

    def rvapply(self, ys: DenseArray, batch_space=None) -> DenseArray:
        """Apply the adjoint over a batch of codomain elements."""
        in_space = self._input_batch_space(self.codomain, ys, batch_space)
        if self._enable_checks:
            in_space._check_member(ys)
        diagonal = self._reshape_diagonal_for_batch(self._diag_adjoint, in_space)
        xs = diagonal * ys
        if self._enable_checks:
            self._output_batch_space(self.domain, in_space)._check_member(xs)
        return xs

    def to_dense(self) -> DenseArray:
        """Return a dense tensor representation of this diagonal operator."""
        flat = self.diagonal.reshape((prod(self.domain.shape),))
        matrix = self.ops.diag(flat)
        return self.ops.reshape(matrix, tuple(self.codomain.shape) + tuple(self.domain.shape))

    def is_hermitian(self) -> bool | None:
        """
        Return whether this diagonal operator is structurally Hermitian.

        Returns
        -------
        bool
            ``True`` when the diagonal equals its complex conjugate.
        """
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
            VectorSpace(tuple(self.domain.shape), new_ctx),
            new_ctx,
        )
