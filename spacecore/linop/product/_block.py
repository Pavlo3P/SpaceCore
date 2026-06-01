from __future__ import annotations

from typing import Any, Tuple

from ._base import ProductLinOp
from .._base import LinOp
from ..._batching import _check_batched
from ... import Context
from ...space import ProductSpace
from ...backend import jax_pytree_class


@jax_pytree_class
class BlockDiagonalLinOp(ProductLinOp[ProductSpace, ProductSpace]):
    r"""
    Block-diagonal operator between product spaces.

    If ``dom = X1 x ... x Xk`` and ``cod = Y1 x ... x Yk``, component
    ``parts[i]`` maps ``Xi`` to ``Yi``.

    Parameters
    ----------
    dom : ProductSpace
        Product domain.
    cod : ProductSpace
        Product codomain.
    parts : sequence of LinOp
        Component operators with matching product incidence.
    ctx : Context, str, or None, optional
        Backend context specification.
    """

    def __init__(
        self,
        dom: ProductSpace,
        cod: ProductSpace,
        parts: Tuple[LinOp, ...],
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, cod, parts, ctx)
        if not self._enable_checks:
            self.apply = self._apply_unchecked
            self.rapply = self._rapply_unchecked
            self.vapply = self._vapply_unchecked
            self.rvapply = self._rvapply_unchecked

    def _check_layout(self) -> None:
        """Check that each component maps the matching product component."""
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
        """Apply each block to the matching product component."""
        if self._enable_checks:
            self.dom._check_member(x)
        y = self._apply_unchecked(x)
        if self._enable_checks:
            self.cod._check_member(y)
        return y

    def _apply_unchecked(self, x: Any) -> Any:
        """Apply each block without membership checks."""
        if self._num_parts == 2:
            return self._apply_parts[0](x[0]), self._apply_parts[1](x[1])
        return tuple(apply(xi) for apply, xi in zip(self._apply_parts, x))

    def rapply(self, y: Any) -> Any:
        """Apply each adjoint block to the matching product component."""
        if self._enable_checks:
            self.cod._check_member(y)
        x = self._rapply_unchecked(y)
        if self._enable_checks:
            self.dom._check_member(x)
        return x

    def _rapply_unchecked(self, y: Any) -> Any:
        """Apply each adjoint block without membership checks."""
        if self._num_parts == 2:
            return self._rapply_parts[0](y[0]), self._rapply_parts[1](y[1])
        return tuple(rapply(yi) for rapply, yi in zip(self._rapply_parts, y))

    def vapply(self, x: Any) -> Any:
        """Apply this block-diagonal operator over a product batch."""
        if self._enable_checks:
            _check_batched(self.domain, x)
        return self._vapply_unchecked(x)

    def _vapply_unchecked(self, x: Any) -> Any:
        """Apply over a product batch without membership checks."""
        y = tuple(op.vapply(xi) for op, xi in zip(self.parts, x))
        return y

    def rvapply(self, y: Any) -> Any:
        """Apply the adjoint over a product batch."""
        if self._enable_checks:
            _check_batched(self.codomain, y)
        x = self._rvapply_unchecked(y)
        if self._enable_checks:
            _check_batched(self.domain, x)
        return x

    def _rvapply_unchecked(self, y: Any) -> Any:
        """Apply the adjoint over a product batch without membership checks."""
        x = tuple(op.rvapply(yi) for op, yi in zip(self.parts, y))
        return x

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> BlockDiagonalLinOp:
        """Build a block-diagonal operator from component operators."""
        if not parts:
            raise ValueError("Parts must be non-empty.")

        dom = ProductSpace(tuple(op.dom for op in parts))
        cod = ProductSpace(tuple(op.cod for op in parts))
        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> BlockDiagonalLinOp:
        """Convert spaces and component operators to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return BlockDiagonalLinOp(new_dom, new_cod, new_parts, new_ctx)
