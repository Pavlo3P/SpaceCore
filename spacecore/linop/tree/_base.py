from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, Sequence

from .._base import LinOp, Domain, Codomain
from ...contextual import Context
from ..._checks import CheckLevel


class TreeLinOp(LinOp[Domain, Codomain]):
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
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("dom", "cod", "parts", "ctx")
    _dynamic = ("parts",)

    parts: Tuple[LinOp, ...]

    def __init__(
        self,
        dom: Domain,
        cod: Codomain,
        parts: Sequence[LinOp],
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        if not parts:
            raise ValueError("Parts must be non-empty.")

        super().__init__(dom, cod, ctx, check_level=check_level)

        # Bind the parts, then cache each part's check-free apply and adjoint.
        self.parts = tuple(op.convert(self.ctx) for op in parts)
        self._apply_parts = tuple(op._apply_core for op in self.parts)
        self._rapply_parts = tuple(op._rapply_core for op in self.parts)
        self._check_layout()

    @abstractmethod
    def _check_layout(self) -> None:
        """Check incidence compatibility between parts and endpoint spaces."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_operators(cls, parts: Tuple[LinOp, ...]) -> "TreeLinOp":
        """Build a tree-structured operator from component operators."""
        ...
