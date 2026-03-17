from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Tuple

from ..space import Space
from ..backend import Context
from .._contextual import ContextBound

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
        ctx = self._resolve_ctx_priority(ctx, dom, cod)
        super(LinOp, self).__init__(ctx)

        dom, cod = self._homogenize(dom, cod)

        self.dom = dom
        self.cod = cod

    def _resolve_ctx_priority(self,
                               explicit_ctx: Context | str | None = None,
                               dom: Domain | None = None,
                               cod: Domain | None = None,
                               ) -> Context | str | None:
        """
        If explicitly passed ctx is None, prioritize domain ctx.
        If codomain_ctx is None, return None, so ContextBound initializes currently default ctx.
        """
        if explicit_ctx is None:
            if dom is None:
                if cod is None:
                    return None
                return cod.ctx
            return dom.ctx
        return explicit_ctx

    def _homogenize(self, s1: Space, s2: Space) -> Tuple[Domain, Codomain]:
        return s1.convert(self.ctx), s2.convert(self.ctx)

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
