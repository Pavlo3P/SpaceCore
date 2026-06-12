from __future__ import annotations

from typing import Any, Tuple

from ._base import TreeLinOp
from .._base import LinOp
from ..._checks import checked_method
from ... import Context
from ...space import TreeSpace
from ...backend import jax_pytree_class


@jax_pytree_class
class BlockDiagonalLinOp(TreeLinOp[TreeSpace, TreeSpace]):
    r"""
    Represent a block-diagonal map between tree spaces.

    If ``dom = X1 x ... x Xk`` and ``cod = Y1 x ... x Yk``, component
    ``parts[i]`` maps ``Xi`` to ``Yi``. Input and output representations follow
    the corresponding :class:`TreeSpace` structures. Each block acts on one
    leaf in deterministic path order and the result is rebuilt with the
    codomain tree definition.

    Parameters
    ----------
    dom : TreeSpace
        Tree-structured product domain.
    cod : TreeSpace
        Tree-structured product codomain.
    parts : sequence of LinOp
        Component operators with matching product incidence.
    ctx : Context, str, or None, optional
        Backend context specification.
    """

    def __init__(
        self,
        dom: TreeSpace,
        cod: TreeSpace,
        parts: Tuple[LinOp, ...],
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, cod, parts, ctx)

    def _check_layout(self) -> None:
        """Check that each component maps the matching tree leaf."""
        if not isinstance(self.dom, TreeSpace) or not isinstance(self.cod, TreeSpace):
            raise TypeError("BlockDiagonalLinOp expects dom and cod to be TreeSpace.")

        if len(self.parts) != self.dom.arity or len(self.parts) != self.cod.arity:
            raise ValueError("Number of component ops must match tree arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom.leaf_spaces[i] and A.cod == self.cod.leaf_spaces[i]:
                continue
            else:
                raise TypeError(f"Component op {i} has incompatible dom/cod spaces.")

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Apply each block to the matching component of a product element."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        """Apply each block without membership checks and rebuild codomain representation."""
        x_parts = self.dom._components(x)
        if self._num_parts == 2:
            y_parts = (self._apply_parts[0](x_parts[0]), self._apply_parts[1](x_parts[1]))
        else:
            y_parts = tuple(apply(xi) for apply, xi in zip(self._apply_parts, x_parts))
        return self.cod._from_components(y_parts)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Apply each adjoint block to the matching component of a product element."""
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        """Apply each adjoint block without checks and rebuild domain representation."""
        y_parts = self.cod._components(y)
        if self._num_parts == 2:
            x_parts = (self._rapply_parts[0](y_parts[0]), self._rapply_parts[1](y_parts[1]))
        else:
            x_parts = tuple(rapply(yi) for rapply, yi in zip(self._rapply_parts, y_parts))
        return self.dom._from_components(x_parts)

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, x: Any) -> Any:
        """Apply this block-diagonal operator over a structured product batch."""
        return self._vapply_unchecked(x)

    def _vapply_unchecked(self, x: Any) -> Any:
        """Apply over a product batch without checks and rebuild codomain representation."""
        x_parts = self.dom._components(x)
        y_parts = tuple(op.vapply(xi) for op, xi in zip(self.parts, x_parts))
        return self.cod._from_components(y_parts)

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, y: Any) -> Any:
        """Apply the adjoint over a structured product batch."""
        return self._rvapply_unchecked(y)

    def _rvapply_unchecked(self, y: Any) -> Any:
        """Apply the adjoint over a product batch without checks and rebuild domain representation."""
        y_parts = self.cod._components(y)
        x_parts = tuple(op.rvapply(yi) for op, yi in zip(self.parts, y_parts))
        return self.dom._from_components(x_parts)

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> BlockDiagonalLinOp:
        """Build a block-diagonal operator from component operators."""
        if not parts:
            raise ValueError("Parts must be non-empty.")

        structure = tuple(range(len(parts)))
        dom = TreeSpace(structure, tuple(op.dom for op in parts))
        cod = TreeSpace(structure, tuple(op.cod for op in parts))
        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> BlockDiagonalLinOp:
        """Convert spaces and component operators to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return BlockDiagonalLinOp(new_dom, new_cod, new_parts, new_ctx)
