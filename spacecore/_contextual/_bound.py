from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Self

from ..types import DType
from ._manager import enforce_convert_policy, normalize_context

if TYPE_CHECKING:
    from ..backend import BackendFamily, BackendOps, Context


def _same_context_for_conversion(left: Context, right: Context) -> bool:
    """
    Compare contexts for conversion equivalence.

    Parameters
    ----------
    left:
        First context to compare.
    right:
        Second context to compare.

    Returns
    -------
    bool
        ``True`` when both contexts use the same backend operations, dtype, and
        check policy; otherwise ``False``.

    Notes
    -----
    This predicate is used by ``convert()`` and intentionally includes
    ``enable_checks`` because a converted object with different runtime checks
    is operationally different.
    """
    return (
        left.ops == right.ops
        and left.dtype == right.dtype
        and left.enable_checks == right.enable_checks
    )


def _same_context_for_algebra(left: Context, right: Context) -> bool:
    """
    Compare contexts for algebraic compatibility.

    Parameters
    ----------
    left:
        First context to compare.
    right:
        Second context to compare.

    Returns
    -------
    bool
        ``True`` when both contexts use the same backend operations and dtype.

    Notes
    -----
    Algebraic combinators ignore ``enable_checks`` because validation policy is
    operational, not mathematical.
    """
    return left.ops == right.ops and left.dtype == right.dtype


class ContextBound(ABC):
    """
    Base class for objects bound to a SpaceCore execution context.

    ``ContextBound`` normalizes and stores a :class:`~spacecore.backend.Context`
    for subclasses such as spaces, linear operators, and functionals. It also
    provides convenience access to the context's backend operations and dtype,
    plus a common ``convert`` workflow that respects the global context
    conversion policy.

    Subclasses that own backend arrays or nested context-bound objects must
    implement :meth:`_convert` to rebuild themselves in a target context.

    Parameters
    ----------
    ctx:
        Context specification passed to :meth:`__init__`. This may be a
        concrete :class:`~spacecore.backend.Context`, a backend-family string,
        or ``None`` to use the current default context.

    Returns
    -------
    ContextBound
        A context-aware object whose concrete type is provided by a subclass.
    """

    def __init__(self, ctx: Context | str | None = None):
        """
        Initialize this object with a normalized context.

        Parameters
        ----------
        ctx:
            Context specification for the object. This may be a concrete
            :class:`~spacecore.backend.Context`, a backend-family string, or
            ``None`` to use the current default context.

        Returns
        -------
        None
            The initializer stores the normalized context on ``self``.
        """
        ctx = normalize_context(ctx)
        self._ctx = ctx

    @property
    def ops(self) -> BackendOps:
        """
        Return backend operations associated with this object's context.

        Parameters
        ----------
        None

        Returns
        -------
        BackendOps
            Backend operation object used by this instance.
        """
        return self.ctx.ops

    @property
    def dtype(self) -> DType:
        """
        Return the default dtype associated with this object's context.

        Parameters
        ----------
        None

        Returns
        -------
        DType
            Backend-normalized dtype stored in the bound context.
        """
        return self.ctx.dtype

    @property
    def ctx(self) -> Context:
        """
        Return the execution context bound to this object.

        Parameters
        ----------
        None

        Returns
        -------
        Context
            Context that controls backend operations, dtype, and validation
            policy for this instance.
        """
        return self._ctx

    def _convert(self, new_ctx: Context) -> Self:
        """
        Rebuild this object in ``new_ctx``.

        Subclasses implement this hook with their concrete conversion logic.
        The public :meth:`convert` method handles policy enforcement and skips
        conversion when the target context is effectively identical.

        Parameters
        ----------
        new_ctx:
            Concrete target context in which the subclass should rebuild its
            owned arrays, spaces, operators, or nested context-bound objects.

        Returns
        -------
        Self
            New object of the subclass type represented in ``new_ctx``.
        """
        raise NotImplementedError()

    def convert(self, new_ctx: Context | BackendFamily | str | None = None) -> Self:
        """
        Return this object represented in ``new_ctx``.

        Parameters
        ----------
        new_ctx:
            Target context specification. ``None`` resolves according to the
            current conversion policy and default context.

        Returns
        -------
        Self
            ``self`` when no effective context change is needed; otherwise a
            converted object produced by :meth:`_convert`.
        """
        _, new_ctx = enforce_convert_policy(self, new_ctx)
        if _same_context_for_conversion(self.ctx, new_ctx):
            return self
        return self._convert(new_ctx)
