from __future__ import annotations

from math import prod
from typing import Any

from ._base import LinOp, Domain, Codomain
from ..space import VectorSpace
from ..types import DenseArray, SparseArray
from ..backend import jax_pytree_class, Context
from .._contextual.manager import ctx_manager


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
        ctx = ctx_manager.resolve_context_priority(ctx, dom, cod)
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
        if not self._enable_checks:
            self.apply = self._apply_unchecked
            self.rapply = self._rapply_unchecked

    @property
    def A(self) -> SparseArray:
        """
        Stored sparse matrix representation of this operator.

        The returned sparse matrix has shape
        ``(prod(self.codomain.shape), prod(self.domain.shape))`` and is the
        same object supplied at construction.
        """
        return self._A

    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action: y = A ⋅ x with y in cod.shape.

        x must have shape dom.shape (dense).
        """
        if self._enable_checks:
            self.dom._check_member(x)
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: DenseArray) -> DenseArray:
        x1 = x if self._dom_is_flat else x.reshape((self._dom_size,))
        y1 = self.A @ x1   # (m,)
        if self._cod_vector_fast_path:
            return y1 if self._cod_is_flat else y1.reshape(self.cod.shape)
        return self.cod.unflatten(y1)

    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Adjoint action: x = A^* ⋅ y with x in dom.shape.

        y must have shape cod.shape (dense).
        """
        if self._enable_checks:
            self.cod._check_member(y)
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: DenseArray) -> DenseArray:
        y1 = y if self._cod_is_flat else y.reshape((self._cod_size,))
        x1 = self._AH @ y1

        if self._dom_vector_fast_path:
            return x1 if self._dom_is_flat else x1.reshape(self.dom.shape)
        return self.dom.unflatten(x1)

    def vapply(self, xs: DenseArray, batch_space=None) -> DenseArray:
        in_space = self._input_batch_space(self.domain, xs, batch_space)
        if tuple(getattr(in_space, "batch_axes", ())) != tuple(range(len(in_space.batch_shape))):
            return self._fallback_vapply(xs, batch_space)
        if self._enable_checks:
            in_space._check_member(xs)
        batch_shape = tuple(in_space.batch_shape)
        xs2 = self.ops.reshape(xs, (-1, self._dom_size))
        ys_t = self.ops.sparse_matmul(self.A, self.ops.transpose(xs2))
        ys2 = self.ops.transpose(ys_t)
        ys_flat = self.ops.reshape(ys2, batch_shape + (self._cod_size,))
        if self._cod_vector_fast_path:
            ys = self.ops.reshape(ys2, batch_shape + tuple(self.cod.shape))
        else:
            ys = self.cod.batch(batch_shape, tuple(range(len(batch_shape)))).unflatten(ys_flat)
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
        ys2 = self.ops.reshape(ys, (-1, self._cod_size))
        xs_t = self.ops.sparse_matmul(self._AH, self.ops.transpose(ys2))
        xs2 = self.ops.transpose(xs_t)
        xs_flat = self.ops.reshape(xs2, batch_shape + (self._dom_size,))
        if self._dom_vector_fast_path:
            xs = self.ops.reshape(xs2, batch_shape + tuple(self.dom.shape))
        else:
            xs = self.dom.batch(batch_shape, tuple(range(len(batch_shape)))).unflatten(xs_flat)
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
