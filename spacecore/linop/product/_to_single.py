from __future__ import annotations

from typing import Any, Tuple

from ._base import ProductLinOp
from .._base import LinOp, Codomain
from ...space import ProductSpace
from ...backend import jax_pytree_class


@jax_pytree_class
class SumToSingleLinOp(ProductLinOp[ProductSpace, Codomain]):
    """
    Sum of component operators from a product domain into a single codomain.

    dom = X1 × ... × Xk
    cod = Y

    ops[i] : Xi -> Y
    apply(x)  = sum_i ops[i](x_i)
    rapply(y) = (ops[i]^*(y))_i
    """

    def _check_layout(self) -> None:
        if not isinstance(self.dom, ProductSpace):
            raise TypeError("SumToSingleLinOp expects dom to be ProductSpace.")

        if len(self.parts) != len(self.dom.spaces):
            raise ValueError("Number of ops must match product arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom.spaces[i] and A.cod == self.cod:
                continue
            else:
                raise TypeError(f"Component op {i} must map dom.spaces[{i}] -> cod.")

    def apply(self, x: Any) -> Any:
        self.assert_domain(x)
        acc = None
        for A, xi in zip(self.parts, x):
            yi = A.apply(xi)
            acc = yi if acc is None else self.cod.add(yi, acc)
        return acc

    def rapply(self, y: Any) -> Any:
        self.assert_codomain(y)
        return tuple(A.rapply(y) for A in self.parts)

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> SumToSingleLinOp:
        if not parts:
            raise ValueError("Parts must be non-empty.")

        dom = ProductSpace(tuple(op.dom for op in parts))
        cod = parts[0].cod

        return cls(dom, cod, parts)
