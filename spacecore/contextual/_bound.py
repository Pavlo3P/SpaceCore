from __future__ import annotations

from abc import ABC
from typing import Any, Self

from ..backend import BackendFamily, BackendOps
from .._check_policy import (
    CheckLevel,
    check_level_at_least,
    level_to_enabled,
    minimum_check_level,
    normalize_check_level,
)
from .._repr import format_dtype
from ..types import DType
from ._state import (
    enforce_convert_policy,
    get_check_level,
    normalize_context,
    resolve_context_priority,
)
from ._context import Context


class ContextBound(ABC):
    """
    Base class for objects bound to a SpaceCore execution context.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used to resolve backend operations and dtype.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Runtime validation policy for this object. When omitted, the ambient
        default (see :func:`spacecore.get_check_level`) is used. Unlike the
        backend/dtype context, the validation policy is a property of the bound
        object, not of the :class:`Context`.
    """

    def __init__(
        self,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
    ):
        ctx = normalize_context(ctx)
        self._ctx = ctx
        self._check_level = (
            normalize_check_level(check_level) if check_level is not None else get_check_level()
        )

    def same_math(self, other: Any) -> bool:
        """Tier-1 equality gate: same concrete type and same math context.

        Combines a runtime-type identity check with :meth:`Context.same_math`
        (equal backend ops and dtype, ignoring ``check_level`` — a validation
        policy, not a mathematical or backend property). This is the mandatory
        first check in every ``__eq__``: the type gate guarantees ``other``
        exposes the same attributes this object's ``__eq__`` will read, and
        ``same_math`` guarantees any later ``ops.allclose`` runs only on
        same-backend, same-dtype arrays, making cross-backend objects compare
        unequal.

        Callers that fail this gate should return ``NotImplemented`` so Python
        can try the reflected comparison and fall back to identity symmetrically.
        """
        return type(self) is type(other) and self._ctx.same_math(other.ctx)

    def _bind_context(
        self,
        ctx: Any,
        *children: Any,
        sources: tuple[Any, ...] | None = None,
        check_level: CheckLevel | bool | None = None,
    ):
        """Resolve the priority context, store it, and re-bind children onto it.

        This factors the context-binding prologue shared by every contextual
        container: it resolves ``ctx`` (taking an explicit context first, then
        inferring from ``sources``), assigns :attr:`_ctx`, and returns each
        child converted onto the resolved context. Containers assign the result
        to their named attributes, e.g.::

            self.dom, self.cod = self._bind_context(ctx, dom, cod)

        Parameters
        ----------
        ctx : Context, BackendFamily, str, or None
            Context specification. An already-resolved :class:`Context` is used
            as-is, which avoids re-resolving when a subclass has normalized the
            context early (e.g. to validate stored arrays before ``__init__``).
        *children : Any
            Context-bound objects (spaces, operators, functionals) to convert
            onto the resolved context and return, in order.
        sources : tuple, optional
            Objects used to infer the context when ``ctx`` is not explicit.
            Defaults to ``children``. Use this when the inference sources differ
            from the attributes being converted.

        Returns
        -------
        tuple
            ``children`` converted onto the resolved context, in order.
        """
        if isinstance(ctx, Context):
            resolved = ctx
        else:
            resolved = resolve_context_priority(
                ctx, *(children if sources is None else sources)
            )
        self._ctx = resolved
        if check_level is not None:
            self._check_level = normalize_check_level(check_level)
        else:
            levels = [c._check_level for c in children if isinstance(c, ContextBound)]
            self._check_level = minimum_check_level(tuple(levels)) if levels else get_check_level()
        return tuple(child.convert(resolved) for child in children)

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

    @property
    def check_level(self) -> CheckLevel:
        """Return this object's runtime validation level.

        The level is stored on the bound object (``_check_level``), seeded at
        construction. The fallback covers any construction path that assigns
        ``_ctx`` without going through ``__init__``/``_bind_context``.
        """
        if hasattr(self, "_check_level"):
            return self._check_level
        return get_check_level()

    @property
    def _enable_checks(self) -> bool:
        """Deprecated internal Boolean view retained for compatibility."""
        return level_to_enabled(self.check_level)

    def _checks_at_least(self, level: CheckLevel) -> bool:
        """Return whether this object runs checks assigned to ``level``."""
        return check_level_at_least(self.check_level, level)

    def _coerce_dense(self, x: Any) -> Any:
        """Return ``x`` asserted as a dense array when cheap checks are enabled.

        Hot-path helper for dense spaces: when ``check_level`` is at least
        ``"cheap"`` it validates ``x`` via ``self.ctx.assert_dense`` (returning
        the validated value); otherwise it returns ``x`` unchanged.
        """
        return self.ctx.assert_dense(x) if self._checks_at_least("cheap") else x

    def _backend_tag(self) -> str:
        """Return the terse backend half of a repr (``backend='numpy', dtype=float64``)."""
        return f"backend={self.ops.family!r}, dtype={format_dtype(self.dtype)}"

    def _repr_class_name(self) -> str:
        """Return the class label shown in reprs.

        Defaults to the runtime class name. Containers whose public constructor
        dispatches to private capability subclasses (e.g. ``StackedSpace`` ->
        ``_StackedInnerProductStarSpace``) override this to present the importable
        public name instead of the internal one.
        """
        return type(self).__name__

    def _repr_body(self) -> str:
        """Return the algebraic half of a repr. Override per class.

        The base returns an empty string, so a plain context-bound object shows
        only its backend tag. Subclasses return a compact math descriptor (a
        space shape, a domain/codomain arrow, an operand expression).
        """
        return ""

    def _short_repr(self) -> str:
        """Return a compact, backend-free repr for nesting inside other reprs."""
        body = self._repr_body()
        name = self._repr_class_name()
        return f"{name}({body})" if body else name

    def __repr__(self) -> str:
        body = self._repr_body()
        tag = self._backend_tag()
        inner = f"{body}, {tag}" if body else tag
        return f"{self._repr_class_name()}({inner})"

    def _convert(self, new_ctx: Context) -> Self:
        """Rebuild this object in ``new_ctx``."""
        raise NotImplementedError()

    def convert(self, new_ctx: Context | BackendFamily | str | None = None) -> Self:
        """Return this object represented in ``new_ctx``.

        The object's ``check_level`` is a property of the object, not of the
        backend/dtype context, so it is preserved across conversion rather than
        re-seeded from the ambient default.
        """
        _, new_ctx = enforce_convert_policy(self, new_ctx)
        if self.ctx == new_ctx:
            return self
        result = self._convert(new_ctx)
        result._check_level = self._check_level
        return result
