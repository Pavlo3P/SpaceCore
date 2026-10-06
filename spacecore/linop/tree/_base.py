from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, Sequence, Any

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
        self._num_parts = len(self.parts)
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

    def _endpoint_treedefs(self) -> tuple[Any, Any]:
        """Return the tree definitions of the domain/codomain endpoints.

        A single (non-tree) endpoint contributes ``None``. Used as an explicit
        tree-structure guard in :meth:`__eq__`.
        """
        return (getattr(self.dom, "treedef", None), getattr(self.cod, "treedef", None))

    def __eq__(self, other: Any) -> bool:
        """Return whether another tree operator has the same layout."""
        if not self.same_math(other):                  # Tier 1: backend
            return NotImplemented
        if self.dom != other.dom or self.cod != other.cod:          # Tier 2a: endpoint spaces
            return False
        if self._endpoint_treedefs() != other._endpoint_treedefs(): # Tier 2b: tree structure
            return False
        if len(self.parts) != len(other.parts):                     # Tier 2c: count before zip
            return False
        return all(op1 == op2 for op1, op2 in zip(self.parts, other.parts))  # Tier 3: per-part
