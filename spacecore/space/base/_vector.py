from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ._space import Space


class VectorSpace(Space):
    """Abstract vector-space capability: linear operations only."""

    @abstractmethod
    def zeros(self) -> Any:
        """Return the additive identity."""

    @abstractmethod
    def add(self, x: Any, y: Any) -> Any:
        """Return x + y."""

    @abstractmethod
    def scale(self, a: Any, x: Any) -> Any:
        """Return a * x."""

    def axpy(self, a: Any, x: Any, y: Any) -> Any:
        """Return a*x + y."""
        return self.add(self.scale(a, x), y)
