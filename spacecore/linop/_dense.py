from __future__ import annotations

from math import prod
from typing import Any

from ._base import LinOp, Domain, Codomain
from ..space import VectorSpace
from ..types import DenseArray
from ..backend import jax_pytree_class, Context


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
        ctx = self._resolve_ctx_priority(ctx, dom, cod)
        ctx.assert_dense(A)  # Check if A is ndarray of ctx

        if cod is None:
            cod_shape_len = len(A.shape) - len(dom.shape)
            cod = VectorSpace(A.shape[:cod_shape_len], ctx)

        super(DenseLinOp, self).__init__(dom, cod, ctx)

        expected = tuple(self.cod.shape) + tuple(self.dom.shape)
        if tuple(A.shape) != expected:
            raise TypeError(f"Expected A.shape == cod.shape + dom.shape == {expected}, got {A.shape}")

        self.A = A  # No dtype conversion

    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action: y = A ⋅ x with y in cod.shape.
        """
        self.assert_domain(x)

        m = prod(self.cod.shape)
        n = prod(self.dom.shape)

        A = self.A.reshape((m, n))
        x1 = x.reshape((n,))
        y1 = A @ x1
        y  = self.cod.unflatten(y1)
        return y

    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Adjoint action: x = A^* ⋅ y with x in dom.shape.

        For complex A, uses conjugate-transpose of the 2D reshaped matrix.
        """
        self.assert_codomain(y)

        m = prod(self.cod.shape)
        n = prod(self.dom.shape)

        A2 = self.A.reshape((m, n))
        y1 = y.reshape((m,))
        x1 = A2.T.conj() @ y1
        x  = self.dom.unflatten(x1)
        return x

    def __eq__(self, x: Any) -> bool:
        if type(x) is type(self):
            return (self.dom == x.dom
                and self.cod == x.cod
                and self.ops.allclose(self.A, x.A)
            )
        return False

    def tree_flatten(self):
        aux = (self.dom, self.cod)
        children = (self.A,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod = aux
        A = children[0]
        return cls(A, dom, cod)

    def _convert(self, new_ctx: Context | str | None = None) -> DenseLinOp:
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_A = new_ctx.asarray(self.A)
        return DenseLinOp(new_A, new_dom, new_cod, new_ctx)
