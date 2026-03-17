from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, Sequence, Any

from .._base import LinOp, Domain, Codomain
from ...backend import jax_pytree_class, Context


@jax_pytree_class
class ProductLinOp(LinOp[Domain, Codomain]):
    """
    Base class for linear operators assembled from component operators.
    """

    parts: Tuple[LinOp, ...]

    def __init__(self,
                 dom: Domain,
                 cod: Codomain,
                 parts: Sequence[LinOp],
                 ctx: Context | str | None = None
                 ) -> None:
        if not parts:
            raise ValueError("Parts must be non-empty.")

        super().__init__(dom, cod, ctx)

        hom_parts = [op.convert(self.ctx) for op in parts]
        self.parts = tuple(hom_parts)

        self._check_layout()

    @abstractmethod
    def _check_layout(self) -> None:
        """
        Check incidence compatibility between self.parts and self.dom/self.cod.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> ProductLinOp:
        ...

    def __eq__(self, x: Any) -> bool:
        if type(x) is type(self):
            return (self.dom == x.dom
                and self.cod == x.cod
                and len(self.parts) == len(x.parts)
                and all([op1 == op2 for op1, op2 in zip(self.parts, x.parts)])
            )
        return False

    def tree_flatten(self):
        children = self.parts
        aux = (self.dom, self.cod)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod = aux
        return cls(dom, cod, tuple(children))
