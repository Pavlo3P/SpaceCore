from __future__ import annotations

from math import prod
from typing import Any

from ._base import LinOp, Domain, Codomain
from ..space import VectorSpace
from ..types import DenseArray
from ..backend import jax_pytree_class, Context
from .._contextual.manager import ctx_manager


@jax_pytree_class
class DenseLinOp(LinOp[VectorSpace, VectorSpace]):
    """
    Dense linear operator defined by an array A with shape:

        A.shape == cod.shape + dom.shape

    apply:  y = A ⋅ x  (contract over dom axes)
    rapply: x = A^* ⋅ y  (contract over cod axes)
    """

    def __init__(self,
                 A: DenseArray,
                 dom: Domain,
                 cod: Codomain | None = None,
                 ctx: Context | str | None = None
                 ) -> None:
        ctx = ctx_manager.resolve_context_priority(ctx, dom, cod)
        ctx.assert_dense(A)  # Check if A is ndarray of ctx

        if cod is None:
            cod_shape_len = len(A.shape) - len(dom.shape)
            cod = VectorSpace(A.shape[:cod_shape_len], ctx)

        super(DenseLinOp, self).__init__(dom, cod, ctx)

        expected = tuple(self.cod.shape) + tuple(self.dom.shape)
        if tuple(A.shape) != expected:
            raise TypeError(f"Expected A.shape == cod.shape + dom.shape == {expected}, got {A.shape}")

        self._A = A  # No dtype conversion
        self._cod_size = prod(self.cod.shape)
        self._dom_size = prod(self.dom.shape)
        self._matrix_shape = (self._cod_size, self._dom_size)
        self._A2 = self.A.reshape(self._matrix_shape)
        dtype = self.ops.get_dtype(self.A)
        is_complex = getattr(dtype, "kind", None) == "c" or str(dtype).startswith("torch.complex")
        self._A2T = self._A2.T
        self._A2H = self._A2.T.conj() if is_complex else self._A2.T
        self._dom_is_flat = tuple(self.dom.shape) == (self._dom_size,)
        self._cod_is_flat = tuple(self.cod.shape) == (self._cod_size,)
        self._dom_vector_fast_path = type(self.dom) is VectorSpace
        self._cod_vector_fast_path = type(self.cod) is VectorSpace
        if not self._enable_checks:
            self.apply = self._apply_unchecked
            self.rapply = self._rapply_unchecked
            self.vapply = self._vapply_unchecked
            self.rvapply = self._rvapply_unchecked

    @property
    def A(self) -> DenseArray:
        """
        Stored dense tensor representation of this operator.

        The returned array has shape ``self.codomain.shape + self.domain.shape``
        and is the same object supplied at construction.
        """
        return self._A

    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action: y = A ⋅ x with y in cod.shape.
        """
        if self._enable_checks:
            self.dom._check_member(x)
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: DenseArray) -> DenseArray:
        x1 = x if self._dom_is_flat else x.reshape((self._dom_size,))
        y1 = self._A2 @ x1
        if self._cod_vector_fast_path:
            return y1 if self._cod_is_flat else y1.reshape(self.cod.shape)
        return self.cod.unflatten(y1)

    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Adjoint action: x = A^* ⋅ y with x in dom.shape.

        For complex A, uses conjugate-transpose of the 2D reshaped matrix.
        """
        if self._enable_checks:
            self.cod._check_member(y)
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: DenseArray) -> DenseArray:
        y1 = y if self._cod_is_flat else y.reshape((self._cod_size,))
        x1 = self._A2H @ y1
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
        ys2 = xs2 @ self._A2T
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
        xs2 = ys2 @ self._A2H.T
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
        Return the stored dense tensor representation of this operator.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        return self.A

    def __eq__(self, x: Any) -> bool:
        if type(x) is type(self):
            return (self.dom == x.dom
                and self.cod == x.cod
                and self.ops.allclose(self.A, x.A)
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

    def _convert(self, new_ctx: Context) -> DenseLinOp:
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_A = new_ctx.asarray(self.A)
        return DenseLinOp(new_A, new_dom, new_cod, new_ctx)
