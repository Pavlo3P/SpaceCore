from __future__ import annotations

from typing import Any, Tuple

from ._base import ProductLinOp
from .._base import LinOp, Domain
from ..._batching import _check_batched
from ..._checks import checked_method
from ...space import ProductSpace
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class StackedLinOp(ProductLinOp[Domain, ProductSpace]):
    r"""
    Stack of operators from a single domain into a product codomain.

    If ``dom = X`` and ``cod = Y1 x ... x Yk``, component ``parts[i]`` maps
    ``X`` to ``Yi``. Forward application returns a tuple of component outputs;
    adjoint application sums component adjoints in ``X``.

    Parameters
    ----------
    dom : Space
        Shared component domain.
    cod : ProductSpace
        Product codomain.
    parts : sequence of LinOp
        Operators from ``dom`` to each component of ``cod``.
    ctx : Context, str, or None, optional
        Backend context specification.
    """

    def _check_layout(self) -> None:
        """Check that every component maps the shared domain to one codomain part."""
        if not isinstance(self.cod, ProductSpace):
            raise TypeError("StackedLinOp expects cod to be ProductSpace.")

        if len(self.parts) != len(self.cod.spaces):
            raise ValueError("Number of ops must match codomain product arity.")

        for i, A in enumerate(self.parts):
            if A.dom == self.dom and A.cod == self.cod.spaces[i]:
                continue
            else:
                raise TypeError(f"Component op {i} must map dom -> cod.spaces[{i}].")

    @checked_method(in_space="dom", out_space="cod")
    def apply(self, x: Any) -> Any:
        """Apply each component operator to the same input."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        """Apply component operators without membership checks."""
        if self._num_parts == 2:
            return self._apply_parts[0](x), self._apply_parts[1](x)
        return tuple(apply(x) for apply in self._apply_parts)

    @checked_method(in_space="cod", out_space="dom")
    def rapply(self, y: Any) -> Any:
        """Apply component adjoints and sum in the shared domain."""
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        """Apply component adjoints without membership checks."""
        if self._num_parts == 2:
            x0 = self._rapply_parts[0](y[0])
            x1 = self._rapply_parts[1](y[1])
            return self.dom.add(x0, x1)
        acc = None
        for rapply, yi in zip(self._rapply_parts, y):
            xi = rapply(yi)
            acc = xi if acc is None else self.dom.add(acc, xi)
        return acc

    def vapply(self, x: Any) -> Any:
        """Apply this stacked operator over a batch."""
        if self._enable_checks:
            _check_batched(self.domain, x)
        y = tuple(op.vapply(x) for op in self.parts)
        if self._enable_checks:
            _check_batched(self.codomain, y)
        return y

    def rvapply(self, y: Any) -> Any:
        """Apply the adjoint stacked operator over a product batch."""
        if self._enable_checks:
            _check_batched(self.codomain, y)
        acc = None
        for op, yi in zip(self.parts, y):
            xi = op.rvapply(yi)
            acc = xi if acc is None else self.domain.add_batch(acc, xi)
        if self._enable_checks:
            _check_batched(self.domain, acc)
        return acc

    @classmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> StackedLinOp:
        """Build a stacked operator from component operators."""
        if not parts:
            raise ValueError("Parts must be non-empty.")

        cod = ProductSpace(tuple(op.cod for op in parts))
        dom = parts[0].dom

        return cls(dom, cod, parts)

    def _convert(self, new_ctx: Context) -> StackedLinOp:
        """Convert spaces and component operators to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_parts = [op.convert(new_ctx) for op in self.parts]
        return StackedLinOp(new_dom, new_cod, new_parts, new_ctx)
