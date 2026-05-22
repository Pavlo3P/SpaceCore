from __future__ import annotations

from functools import cached_property
from math import prod
from typing import Any

from ._base import LinOp, Domain, Codomain
from .._checks import checked_method
from ..space import VectorSpace
from ..types import DenseArray, SparseArray
from ..backend import jax_pytree_class, Context
from .._contextual import resolve_context_priority


@jax_pytree_class
class SparseLinOp(LinOp):
    """
    Sparse linear operator implementing the tensor map A : dom -> cod where
    conceptually A has shape cod.shape + dom.shape, but stored as a 2D sparse matrix:

        A2.shape == (prod(cod.shape), prod(dom.shape))

    apply:  y = A ⋅ x  (contract over dom axes)
    rapply: x = A^* ⋅ y  (contract over cod axes)
    """

    def __init__(self,
                 A: SparseArray,
                 dom: Domain,
                 cod: Codomain,
                 ctx: Context | str | None = None
                 ) -> None:
        ctx = resolve_context_priority(ctx, dom, cod)
        ctx.assert_sparse(A)  # Check if A is sparse array of ctx

        super(SparseLinOp, self).__init__(dom, cod, ctx)

        expected = (prod(self.cod.shape), prod(self.dom.shape))
        if tuple(A.shape) != expected:
            raise TypeError(f"Expected A.shape == (prod(cod.shape), prod(dom.shape)) == {expected}, got {A.shape}")

        self._A = A  # No dtype conversion
        self._cod_size = expected[0]
        self._dom_size = expected[1]
        dtype = self.ops.get_dtype(self.A)
        self._A_is_complex = getattr(dtype, "kind", None) == "c" or str(dtype).startswith("torch.complex")
        self._AT = self.A.T
        self._AH = self._AT.conj() if self._A_is_complex else self._AT
        self._dom_is_flat = tuple(self.dom.shape) == (self._dom_size,)
        self._cod_is_flat = tuple(self.cod.shape) == (self._cod_size,)
        self._dom_vector_fast_path = type(self.dom) is VectorSpace
        self._cod_vector_fast_path = type(self.cod) is VectorSpace

    @cached_property
    def A(self) -> SparseArray:
        """
        Stored sparse matrix representation of this operator.

        The returned sparse matrix has shape
        ``(prod(self.codomain.shape), prod(self.domain.shape))`` and is the
        same object supplied at construction.
        """
        return self._A

    @checked_method(in_space="dom", out_space="cod")
    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action: y = A ⋅ x with y in cod.shape.

        x must have shape dom.shape (dense).
        """
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: DenseArray) -> DenseArray:
        x1 = x if self._dom_is_flat else x.reshape((self._dom_size,))
        y1 = self.A @ x1   # (m,)
        if self._cod_vector_fast_path:
            return y1 if self._cod_is_flat else y1.reshape(self.cod.shape)
        return self.cod.unflatten(y1)

    @checked_method(in_space="cod", out_space="dom")
    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Adjoint action: x = A^* ⋅ y with x in dom.shape.

        y must have shape cod.shape (dense).
        """
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: DenseArray) -> DenseArray:
        y1 = y if self._cod_is_flat else y.reshape((self._cod_size,))
        x1 = self._AH @ y1

        if self._dom_vector_fast_path:
            return x1 if self._dom_is_flat else x1.reshape(self.dom.shape)
        return self.dom.unflatten(x1)

    @staticmethod
    def _batch_shape_from_input(value: DenseArray, base_ndim: int) -> tuple[int, ...]:
        shape = tuple(value.shape)
        return shape if base_ndim == 0 else shape[:-base_ndim]

    @staticmethod
    def _is_leading_batch(batch_space: Any) -> bool:
        if batch_space is None:
            return True
        batch_shape = tuple(getattr(batch_space, "batch_shape", ()))
        batch_axes = tuple(getattr(batch_space, "batch_axes", ()))
        return batch_axes == tuple(range(len(batch_shape)))

    @staticmethod
    def _batch_shape_from_space(batch_space: Any) -> tuple[int, ...]:
        return tuple(getattr(batch_space, "batch_shape"))

    def _vapply_unchecked_leading(
        self,
        xs: DenseArray,
        batch_shape: tuple[int, ...],
    ) -> DenseArray:
        xs2 = xs.reshape((-1, self._dom_size))
        ys2 = (self.A @ xs2.T).T
        if self._cod_vector_fast_path:
            if self._cod_is_flat and tuple(ys2.shape[:-1]) == batch_shape:
                return ys2
            return ys2.reshape(batch_shape + tuple(self.cod.shape))
        ys_flat = ys2.reshape(batch_shape + (self._cod_size,))
        return self.cod.batch(batch_shape, tuple(range(len(batch_shape)))).unflatten(ys_flat)

    def _rvapply_unchecked_leading(
        self,
        ys: DenseArray,
        batch_shape: tuple[int, ...],
    ) -> DenseArray:
        ys2 = ys.reshape((-1, self._cod_size))
        xs2 = (self._AH @ ys2.T).T
        if self._dom_vector_fast_path:
            if self._dom_is_flat and tuple(xs2.shape[:-1]) == batch_shape:
                return xs2
            return xs2.reshape(batch_shape + tuple(self.dom.shape))
        xs_flat = xs2.reshape(batch_shape + (self._dom_size,))
        return self.dom.batch(batch_shape, tuple(range(len(batch_shape)))).unflatten(xs_flat)

    def _vapply_unchecked(self, xs: DenseArray, batch_space=None) -> DenseArray:
        if not self._is_leading_batch(batch_space):
            return self._fallback_vapply(xs, batch_space)
        batch_shape = (
            self._batch_shape_from_input(xs, len(self.domain.shape))
            if batch_space is None
            else self._batch_shape_from_space(batch_space)
        )
        return self._vapply_unchecked_leading(xs, batch_shape)

    def _rvapply_unchecked(self, ys: DenseArray, batch_space=None) -> DenseArray:
        if not self._is_leading_batch(batch_space):
            return self._fallback_rvapply(ys, batch_space)
        batch_shape = (
            self._batch_shape_from_input(ys, len(self.codomain.shape))
            if batch_space is None
            else self._batch_shape_from_space(batch_space)
        )
        return self._rvapply_unchecked_leading(ys, batch_shape)

    def vapply(self, xs: DenseArray, batch_space=None) -> DenseArray:
        in_space = self._input_batch_space(self.domain, xs, batch_space)
        if tuple(getattr(in_space, "batch_axes", ())) != tuple(range(len(in_space.batch_shape))):
            return self._fallback_vapply(xs, batch_space)
        if self._enable_checks:
            in_space._check_member(xs)
        batch_shape = tuple(in_space.batch_shape)
        ys = self._vapply_unchecked_leading(xs, batch_shape)
        if self._enable_checks:
            self._output_batch_space(self.codomain, in_space)._check_member(ys)
        return ys

    def rvapply(self, ys: DenseArray, batch_space=None) -> DenseArray:
        in_space = self._input_batch_space(self.codomain, ys, batch_space)
        if tuple(getattr(in_space, "batch_axes", ())) != tuple(range(len(in_space.batch_shape))):
            return self._fallback_rvapply(ys, batch_space)
        if self._enable_checks:
            in_space._check_member(ys)
        batch_shape = tuple(in_space.batch_shape)
        xs = self._rvapply_unchecked_leading(ys, batch_shape)
        if self._enable_checks:
            self._output_batch_space(self.domain, in_space)._check_member(xs)
        return xs

    def to_dense(self) -> DenseArray:
        """
        Materialize the stored sparse matrix as a dense operator tensor.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        if hasattr(self.A, "toarray"):
            dense = self.A.toarray()
        elif hasattr(self.A, "todense"):
            dense = self.A.todense()
        elif hasattr(self.A, "to_dense"):
            dense = self.A.to_dense()
        else:
            dense = super().to_dense().reshape((self._cod_size, self._dom_size))
        return self.ops.reshape(dense, tuple(self.codomain.shape) + tuple(self.domain.shape))

    def __eq__(self, x: Any) -> bool:
        if type(x) is type(self):
            return (self.dom == x.dom
                and self.cod == x.cod
                and self.ops.allclose_sparse(self.A, x.A)
            )
        return False

    def tree_flatten(self):
        aux = (self.dom, self.cod, self.ctx)
        children = (self.A,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod, ctx = aux
        A = children[0]
        return cls(A, dom, cod, ctx)

    def _convert(self, new_ctx: Context) -> SparseLinOp:
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_A = new_ctx.assparse(self.A)
        return SparseLinOp(new_A, new_dom, new_cod, new_ctx)
