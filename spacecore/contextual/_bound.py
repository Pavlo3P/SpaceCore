from __future__ import annotations

from abc import ABC
from typing import Any, Self

from ..backend import BackendFamily, BackendOps
from ..backend._container import field_values, rebuild_class
from .._checks import (
    install_public_checks,
    CheckLevel,
    check_level_at_least,
    effective_check_level,
    minimum_check_level,
    normalize_check_level,
)
from .._repr import format_dtype
from ..types import DType
from ._state import (
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
        Validation level; the ambient default when omitted.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Families implement ``_*_core``; a public operation they override keeps the base checks.
        install_public_checks(cls)
        # Defining __eq__ sets __hash__ to None; a value class keeps its family's hash, which
        # hashes only structure every override of __eq__ also compares.
        if "__eq__" in vars(cls) and vars(cls).get("__hash__", ...) is None:
            inherited = next((vars(base)["__hash__"] for base in cls.__mro__[1:]
                              if vars(base).get("__hash__") not in (None, object.__hash__)), None)
            if inherited is not None:
                cls.__hash__ = inherited

    def _repr_html_(self) -> str:
        """Render the text repr for notebooks."""
        import html

        return f"<code>{html.escape(repr(self))}</code>"


    def __init__(
        self,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
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
        check_level: CheckLevel | None = None,
    ):
        """
        Resolve the context, store it with the check level, and return ``children`` converted onto it.

        An explicit ``ctx`` wins; otherwise it is inferred from ``sources`` (``children`` by
        default). The check level is the explicit one or the lowest among ``children``.
        Typical use: ``self.dom, self.cod = self._bind_context(ctx, dom, cod)``.
        """
        # An explicit context wins; otherwise infer one from the sources.
        if isinstance(ctx, Context):
            resolved = ctx
        else:
            resolved = resolve_context_priority(
                ctx, *(children if sources is None else sources)
            )
        self._ctx = resolved
        # Explicit level, else the least strict of the children, else ambient.
        if check_level is not None:
            self._check_level = normalize_check_level(check_level)
        else:
            levels = [c._check_level for c in children if isinstance(c, ContextBound)]
            self._check_level = minimum_check_level(tuple(levels)) if levels else get_check_level()
        # Move every child onto the resolved context.
        return tuple(child.convert(resolved) for child in children)

    def _with_check_level(self, level: CheckLevel | bool) -> Self:
        """Return a copy of this object validating at ``level``.

        Routes through the subclass's own :meth:`_convert` rather than calling
        ``type(self)(...)``, so a subclass whose constructor takes more than a
        context keeps working — ``_convert`` is the extension point that already
        has to know how to rebuild it. Mirrors :meth:`convert`, which likewise
        assigns ``_check_level`` onto a freshly built object.
        """
        result = self._convert(self.ctx)
        result._check_level = normalize_check_level(level)
        return result

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

    def _checks_at_least(self, level: CheckLevel) -> bool:
        """Return whether this object runs checks assigned to ``level``."""
        return check_level_at_least(effective_check_level(self.check_level, self.ops), level)

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
        """Rebuild this object in ``new_ctx`` from its declared ``_fields``."""
        if not getattr(type(self), "_fields", ()):
            raise NotImplementedError(f"{type(self).__name__} declares no _fields to convert.")
        ops = self.ops

        def move(name: str, value: Any) -> Any:
            # The context is replaced, convertible objects and dense arrays move, the rest stays.
            if name == "ctx":
                return new_ctx
            if hasattr(value, "convert"):
                return value.convert(new_ctx)
            if isinstance(value, (tuple, list)) and value and all(hasattr(v, "convert") for v in value):
                return tuple(v.convert(new_ctx) for v in value)
            if ops.is_dense(value):
                return new_ctx.asarray(value)
            return value

        return rebuild_class(type(self))(**{name: move(name, value) for name, value in field_values(self).items()})

    def convert(self, new_ctx: Context | BackendFamily | str | None = None) -> Self:
        """Return this object represented in ``new_ctx``.

        The object's ``check_level`` is a property of the object, not of the
        backend/dtype context, so it is preserved across conversion rather than
        re-seeded from the ambient default.
        """
        # Same context: nothing to do. Otherwise rebuild, keeping the check level.
        new_ctx = normalize_context(new_ctx)
        if self.ctx == new_ctx:
            return self
        result = self._convert(new_ctx)
        result._check_level = self._check_level
        return result
