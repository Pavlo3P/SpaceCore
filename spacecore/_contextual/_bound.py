from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Self

from ..types import DType
from ._state import enforce_convert_policy, normalize_context

if TYPE_CHECKING:
    from ..backend import BackendFamily, BackendOps, Context


def _same_math_context(left: Context, right: Context) -> bool:
    """Return whether contexts match for algebra, ignoring validation checks."""
    return left.ops == right.ops and left.dtype == right.dtype


class ContextBound(ABC):
    """Base class for objects bound to a SpaceCore execution context."""

    def __init__(self, ctx: Context | str | None = None):
        ctx = normalize_context(ctx)
        self._ctx = ctx

    @property
    def ops(self) -> BackendOps:
        """Return backend operations associated with this object's context."""
        return self.ctx.ops

    @property
    def dtype(self) -> DType:
        """Return the default dtype associated with this object's context."""
        return self.ctx.dtype

    @property
    def ctx(self) -> Context:
        """Return the execution context bound to this object."""
        return self._ctx

    def _convert(self, new_ctx: Context) -> Self:
        """Rebuild this object in ``new_ctx``."""
        raise NotImplementedError()

    def convert(self, new_ctx: Context | BackendFamily | str | None = None) -> Self:
        """Return this object represented in ``new_ctx``."""
        _, new_ctx = enforce_convert_policy(self, new_ctx)
        if self.ctx == new_ctx:
            return self
        return self._convert(new_ctx)
