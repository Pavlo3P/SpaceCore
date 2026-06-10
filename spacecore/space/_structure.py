from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .._tree import tree_flatten, tree_unflatten


class ProductStructure(ABC):
    """Maps a product element to and from an ordered component sequence."""

    @abstractmethod
    def to_components(self, x: Any, *, arity: int) -> tuple[Any, ...]:
        """Return ordered components; validate structure and arity."""
        ...

    @abstractmethod
    def from_components(self, parts: tuple[Any, ...], *, arity: int) -> Any:
        """Rebuild an element from ordered components; validate arity."""
        ...


class TupleStructure(ProductStructure):
    """Default product structure: elements are plain tuples."""

    def to_components(self, x: Any, *, arity: int) -> tuple[Any, ...]:
        if not isinstance(x, tuple):
            raise TypeError(f"ProductSpace element must be a tuple, got {type(x).__name__}")
        if len(x) != arity:
            raise ValueError(f"Expected tuple of length {arity}, got {len(x)}")
        return x

    def from_components(self, parts: tuple[Any, ...], *, arity: int) -> tuple[Any, ...]:
        parts = tuple(parts)
        if len(parts) != arity:
            raise ValueError(f"Expected {arity} product components, got {len(parts)}")
        return parts

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, TupleStructure)

    def __hash__(self) -> int:
        return hash(TupleStructure)


class PytreeStructure(ProductStructure):
    """
    Registered pytree product structure with fixed treedef and leaf order.

    Bare Python dataclasses are opaque leaves to JAX unless registered with
    ``jax.tree_util.register_dataclass`` or
    ``jax.tree_util.register_pytree_node_class``.

    Parameters
    ----------
    template_element : Any
        Example product element whose pytree definition and leaf order define
        the product structure.
    """

    def __init__(self, template_element: Any) -> None:
        leaves, treedef = tree_flatten(template_element)
        self._treedef = treedef
        self._leaf_count = len(leaves)

    @classmethod
    def from_treedef(cls, treedef: Any) -> "PytreeStructure":
        obj = cls.__new__(cls)
        obj._treedef = treedef
        try:
            obj._leaf_count = int(treedef.num_leaves)
        except Exception:
            obj._leaf_count = None
        return obj

    @property
    def treedef(self) -> Any:
        """Structure-only pytree definition."""
        return self._treedef

    @property
    def leaf_count(self) -> int | None:
        """Number of component leaves when known."""
        return self._leaf_count

    def to_components(self, x: Any, *, arity: int) -> tuple[Any, ...]:
        leaves, treedef = tree_flatten(x)
        if treedef != self._treedef:
            raise TypeError(
                f"ProductSpace pytree structure mismatch: expected {self._treedef}, got {treedef}."
            )
        if len(leaves) != arity:
            raise ValueError(
                "ProductSpace pytree arity mismatch: "
                f"expected {arity} leaves/components, got {len(leaves)}. "
                "If this is a dataclass, register it as a JAX pytree/dataclass."
            )
        return tuple(leaves)

    def from_components(self, parts: tuple[Any, ...], *, arity: int) -> Any:
        parts = tuple(parts)
        if len(parts) != arity:
            raise ValueError(f"Expected {arity} product components, got {len(parts)}")
        return tree_unflatten(self._treedef, parts)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PytreeStructure)
            and self._treedef == other._treedef
            and self._leaf_count == other._leaf_count
        )

    def __hash__(self) -> int:
        try:
            return hash((PytreeStructure, self._treedef, self._leaf_count))
        except TypeError:
            return hash((PytreeStructure, repr(self._treedef), self._leaf_count))
