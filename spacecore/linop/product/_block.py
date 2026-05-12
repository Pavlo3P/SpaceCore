from __future__ import annotations

from typing import Any, Tuple

from ._base import ProductLinOp
from .._base import LinOp
from ... import Context
from ...space import ProductSpace
from ...backend import jax_pytree_class


@jax_pytree_class
class BlockDiagonalLinOp(ProductLinOp[ProductSpace, ProductSpace]):
    """
    Block-diagonal operator between product spaces.

    dom = X1 × ... × Xk
    cod = Y1 × ... × Yk

    ops[i] : Xi -> Yi
    """

    def _check_layout(self) -> None:
        if not isinstance(self.dom, ProductSpace) or not isinstance(self.cod, ProductSpace):
            raise TypeError("BlockDiagonalLinOp expects dom and cod to be ProductSpace.")

        if len(self.parts) != len(self.dom.spaces) or len(self.parts) != len(self.cod.spaces):
            raise ValueError("Number of component ops must match product arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom.spaces[i] and A.cod == self.cod.spaces[i]:
                continue
            else:
                raise TypeError(f"Component op {i} has incompatible dom/cod spaces.")

    def apply(self, x: Any) -> Any:
        if self._enable_checks:
            self.dom.check_member(x)
        if self._num_parts == 2:
            return self._apply_parts[0](x[0]), self._apply_parts[1](x[1])
        return tuple(apply(xi) for apply, xi in zip(self._apply_parts, x))

    def rapply(self, y: Any) -> Any:
        if self._enable_checks:
            self.cod.check_member(y)
        if self._num_parts == 2:
            return self._rapply_parts[0](y[0]), self._rapply_parts[1](y[1])
        return tuple(rapply(yi) for rapply, yi in zip(self._rapply_parts, y))

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> BlockDiagonalLinOp:
        if not parts:
            raise ValueError("Parts must be non-empty.")

        dom = ProductSpace(tuple(op.dom for op in parts))
        cod = ProductSpace(tuple(op.cod for op in parts))
        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> BlockDiagonalLinOp:
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return BlockDiagonalLinOp(new_dom, new_cod, new_parts)
