from __future__ import annotations

from typing import Any, Tuple

from ._base import ProductLinOp
from .._base import LinOp, Codomain
from ..._checks import checked_method
from ...space import ProductSpace, VectorSpace
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class SumToSingleLinOp(ProductLinOp[ProductSpace, Codomain]):
    r"""
    Sum of component operators from a product domain into a single codomain.

    If ``dom = X1 x ... x Xk`` and ``cod = Y``, component ``parts[i]`` maps
    ``Xi`` to ``Y``. Forward application sums component outputs in ``Y``;
    adjoint application returns the tuple of component adjoints.

    Parameters
    ----------
    dom : ProductSpace
        Product domain.
    cod : Space
        Shared codomain.
    parts : sequence of LinOp
        Operators from each product component to ``cod``.
    ctx : Context, str, or None, optional
        Backend context specification.
    """

    def _check_layout(self) -> None:
        """Check that every component maps one product part to the shared codomain."""
        if not isinstance(self.dom, ProductSpace):
            raise TypeError("SumToSingleLinOp expects dom to be ProductSpace.")

        if len(self.parts) != len(self.dom.spaces):
            raise ValueError("Number of ops must match product arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom.spaces[i] and A.cod == self.cod:
                continue
            else:
                raise TypeError(f"Component op {i} must map dom.spaces[{i}] -> cod.")

    @checked_method(in_space="dom", out_space="cod")
    def apply(self, x: Any) -> Any:
        """Apply component operators and sum in the codomain."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        """Apply component operators without membership checks."""
        if self._num_parts == 2:
            y0 = self._apply_parts[0](x[0])
            y1 = self._apply_parts[1](x[1])
            return y0 + y1 if type(self.cod) is VectorSpace else self.cod.add(y0, y1)
        acc = None
        use_direct_add = type(self.cod) is VectorSpace
        for apply, xi in zip(self._apply_parts, x):
            yi = apply(xi)
            acc = yi if acc is None else (acc + yi if use_direct_add else self.cod.add(yi, acc))
        return acc

    @checked_method(in_space="cod", out_space="dom")
    def rapply(self, y: Any) -> Any:
        """Apply each component adjoint to the shared codomain element."""
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        """Apply component adjoints without membership checks."""
        if self._num_parts == 2:
            return self._rapply_parts[0](y), self._rapply_parts[1](y)
        return tuple(rapply(y) for rapply in self._rapply_parts)

    def vapply(self, x: Any) -> Any:
        """Apply this sum-to-single operator over a product batch."""
        acc = None
        for op, xi in zip(self.parts, x):
            yi = op.vapply(xi)
            acc = yi if acc is None else acc + yi
        return acc

    def rvapply(self, y: Any) -> Any:
        """Apply the adjoint over a codomain batch."""
        return tuple(op.rvapply(y) for op in self.parts)

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> SumToSingleLinOp:
        """Build a sum-to-single operator from component operators."""
        if not parts:
            raise ValueError("Parts must be non-empty.")

        dom = ProductSpace(tuple(op.dom for op in parts))
        cod = parts[0].cod

        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> SumToSingleLinOp:
        """Convert spaces and component operators to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return SumToSingleLinOp(new_dom, new_cod, new_parts)
