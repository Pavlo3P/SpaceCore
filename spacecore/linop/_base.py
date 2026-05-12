from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from ..space import Space
from ..backend import Context
from .._contextual import ContextBound
from .._contextual.manager import ctx_manager

Domain = TypeVar('Domain', bound=Space)
Codomain = TypeVar('Codomain', bound=Space)

class LinOp(ContextBound, Generic[Domain, Codomain]):
    """
    Minimal linear operator (morphism) between two spaces.

    This class is intentionally small:
      - no matrix semantics
      - no arithmetic
      - no storage assumptions

    Its sole purpose is to represent a linear map
        A : dom -> cod
    with access to both forward and adjoint actions.
    """

    def __init__(self, dom: Domain, cod: Codomain, ctx: Context | str | None = None):
        ctx = ctx_manager.resolve_context_priority(ctx, dom, cod)
        super(LinOp, self).__init__(ctx)

        self.dom = dom.convert(self.ctx)
        self.cod = cod.convert(self.ctx)
        self._enable_checks = self.ctx.enable_checks

    @abstractmethod
    def apply(self, x: Any) -> Any:
        """
        Forward application: y = A x

        Contract:
          - x is an element of self.dom
          - return value is an element of self.cod
        """

    @abstractmethod
    def rapply(self, y: Any) -> Any:
        """
        Adjoint application: x = A^* y

        Contract:
          - y is an element of self.cod
          - return value is an element of self.dom
        """

    def __call__(self, x: Any) -> Any:
        return self.apply(x)

    def assert_domain(self, x: Any) -> None:
        self.dom.check_member(x)

    def assert_codomain(self, y: Any) -> None:
        self.cod.check_member(y)

    def __eq__(self, x: Any) -> bool:
        raise NotImplementedError()

    def tree_flatten(self):
        raise NotImplementedError()

    @classmethod
    def tree_unflatten(cls, aux, children):
        raise NotImplementedError()
