from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class GeneratedCase(Generic[T]):
    """One generated test object together with its deterministic reference data."""

    obj: T
    reference: Mapping[str, Any] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    marks: tuple[Any, ...] = ()
    id: str | None = None
