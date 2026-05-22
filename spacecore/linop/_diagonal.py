from __future__ import annotations

from math import prod
from typing import Any

from ._base import LinOp
from ..backend import Context, jax_pytree_class
from ..space import VectorSpace
from ..types import DenseArray
from .._contextual.manager import ctx_manager


@jax_pytree_class
class DiagonalLinOp(LinOp[VectorSpace, VectorSpace]):
    """Coordinatewise diagonal linear operator on a vector space."""

    def __init__(
        self,
        diagonal: DenseArray,
        space: VectorSpace | None = None,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = ctx_manager.resolve_context_priority(ctx, space)
        ctx.assert_dense(diagonal)
        if space is None:
            space = VectorSpace(tuple(diagonal.shape), ctx)
        super().__init__(space, space, ctx)
        expected = tuple(self.domain.shape)
        if tuple(diagonal.shape) != expected:
            raise TypeError(f"Expected diagonal.shape == space.shape == {expected}, got {diagonal.shape}")
        self.diagonal = diagonal
        self._size = prod(self.domain.shape)
        self._is_flat = tuple(self.domain.shape) == (self._size,)
        self._diag_flat = diagonal if self._is_flat else diagonal.reshape((self._size,))
        dtype = self.ops.get_dtype(diagonal)
        self._is_complex = getattr(dtype, "kind", None) == "c" or str(dtype).startswith("torch.complex")
        self._diag_adjoint = self.ops.conj(diagonal) if self._is_complex else diagonal
        self._diag_adjoint_flat = (
            self._diag_adjoint if self._is_flat else self._diag_adjoint.reshape((self._size,))
        )

    @property
    def A(self) -> DenseArray:
        return self.to_dense()

    def apply(self, x: DenseArray) -> DenseArray:
        if self._enable_checks:
            self.domain._check_member(x)
        return self.diagonal * x

    def rapply(self, y: DenseArray) -> DenseArray:
        if self._enable_checks:
            self.codomain._check_member(y)
        return self._diag_adjoint * y

    def vapply(self, xs: DenseArray, batch_space=None) -> DenseArray:
        in_space = self._input_batch_space(self.domain, xs, batch_space)
        if self._enable_checks:
            in_space._check_member(xs)
        ys = self.diagonal * xs
        if self._enable_checks:
            self._output_batch_space(self.codomain, in_space)._check_member(ys)
        return ys

    def rvapply(self, ys: DenseArray, batch_space=None) -> DenseArray:
        in_space = self._input_batch_space(self.codomain, ys, batch_space)
        if self._enable_checks:
            in_space._check_member(ys)
        xs = self._diag_adjoint * ys
        if self._enable_checks:
            self._output_batch_space(self.domain, in_space)._check_member(xs)
        return xs

    def to_dense(self) -> DenseArray:
        matrix = self.ops.diag(self._diag_flat)
        return self.ops.reshape(matrix, tuple(self.codomain.shape) + tuple(self.domain.shape))

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.domain == other.domain and self.ops.allclose(self.diagonal, other.diagonal)
        return False

    def tree_flatten(self):
        children = (self.diagonal,)
        aux = (self.domain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, ctx = aux
        return cls(children[0], domain, ctx)

    def _convert(self, new_ctx: Context) -> DiagonalLinOp:
        return DiagonalLinOp(new_ctx.asarray(self.diagonal), self.domain.convert(new_ctx), new_ctx)
