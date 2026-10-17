from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..._checks import checked_method
from ._space import Space


class StarSpace(Space):
    """
    Space capability with a canonical involution/star operation.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    @checked_method(in_space="self", out_space="self")
    def star(self, x: Any) -> Any:
        """Return the canonical star/involution of ``x``."""
        return self._star_core(x)

    @abstractmethod
    def _star_core(self, x: Any) -> Any:
        """Return the star of ``x`` without membership checks."""
