from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ._space import Space


class StarSpace(Space):
    """Space capability with a canonical involution/star operation."""

    @abstractmethod
    def star(self, x: Any) -> Any:
        """Return the canonical star/involution of ``x``."""
