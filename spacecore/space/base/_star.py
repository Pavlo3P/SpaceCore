from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ._space import Space


class StarSpace(Space):
    """
    Space capability with a canonical involution/star operation.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    check_level : {{"none", "cheap", "standard", "strict"}}, optional
        Runtime validation policy for this object. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used.
    """

    @abstractmethod
    def star(self, x: Any) -> Any:
        """Return the canonical star/involution of ``x``."""
