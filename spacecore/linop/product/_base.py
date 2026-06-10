from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, Sequence, Any

from .._base import LinOp, Domain, Codomain
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class ProductLinOp(LinOp[Domain, Codomain]):
    """
    Define a base class for operators assembled from component operators.

    Parameters
    ----------
    dom : Space
        Domain space of the assembled operator.
    cod : Space
        Codomain space of the assembled operator.
    parts : sequence of LinOp
        Nonempty sequence of component operators.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom`` and
        ``cod``.
    """

    parts: Tuple[LinOp, ...]

    def __init__(
        self, dom: Domain, cod: Codomain, parts: Sequence[LinOp], ctx: Context | str | None = None
    ) -> None:
        if not parts:
            raise ValueError("Parts must be non-empty.")

        super().__init__(dom, cod, ctx)

        self.parts = tuple(op.convert(self.ctx) for op in parts)
        self._num_parts = len(self.parts)
        self._apply_parts = tuple(
            getattr(op, "_apply_core", getattr(op, "_apply_unchecked", op.apply))
            for op in self.parts
        )
        self._rapply_parts = tuple(
            getattr(op, "_rapply_core", getattr(op, "_rapply_unchecked", op.rapply))
            for op in self.parts
        )
        self._check_layout()

    @abstractmethod
    def _check_layout(self) -> None:
        """Check incidence compatibility between parts and endpoint spaces."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> ProductLinOp:
        """Build a product operator from component operators."""
        ...

    def __eq__(self, x: Any) -> bool:
        """Return whether another product operator has the same layout."""
        if type(x) is type(self):
            return (
                self.dom == x.dom
                and self.cod == x.cod
                and len(self.parts) == len(x.parts)
                and all([op1 == op2 for op1, op2 in zip(self.parts, x.parts)])
            )
        return False

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        children = self.parts
        aux = (self.dom, self.cod, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        dom, cod, ctx = aux
        return cls(dom, cod, tuple(children), ctx)
