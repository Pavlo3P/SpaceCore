from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..._checks import checked_method
from ._space import Space


class VectorSpace(Space):
    """
    Abstract vector-space capability: linear operations only.

    A family implements the check-free cores ``_add_core`` and ``_scale_core``;
    the public ``add`` and ``scale`` wrap them with membership checks.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    @abstractmethod
    def zeros(self) -> Any:
        """Return the additive identity."""

    @abstractmethod
    def _add_core(self, x: Any, y: Any) -> Any:
        """Return x + y without membership checks."""

    @abstractmethod
    def _scale_core(self, a: Any, x: Any) -> Any:
        """Return a * x without membership checks."""

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: Any, y: Any) -> Any:
        """Return x + y."""
        return self._add_core(x, y)

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: Any) -> Any:
        """Return a * x; a multiplier that provably leaves the space is rejected."""
        self.check_scalar(a)
        return self._scale_core(a, x)

    @checked_method(in_space="self", arg_positions=(1, 2))
    def axpy(self, a: Any, x: Any, y: Any) -> Any:
        """Return a*x + y."""
        self.check_scalar(a)
        return self._axpy_core(a, x, y)

    def _axpy_core(self, a: Any, x: Any, y: Any) -> Any:
        """Return a*x + y without membership checks."""
        return self._add_core(self._scale_core(a, x), y)
