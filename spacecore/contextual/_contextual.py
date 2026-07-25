from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable, Iterator, Tuple
from warnings import warn

from .._check_policy import CheckLevel, normalize_check_level
from ..types import DType
from ..backend import BackendFamily, BackendOps, NumpyOps, OpsRegistry
from ..backend import ops_registry as _default_ops_registry
from ._context import Context
from .._errors import ContextInferenceError


# Scoped ambient overrides installed by ``Contextual.scoped_ctx`` /
# ``scoped_check_level`` (public entry points: ``use_context`` /
# ``use_check_level``).
#
# Module level by design. ``ContextVar`` objects are never garbage collected, so
# the contextvars documentation requires creating them at module scope rather
# than per call or per instance. Sharing one pair across ``Contextual`` instances
# is also the intended semantics: an override is ambient to the current thread /
# async task, not a property of whichever resolver observes it.
#
# ``None`` is the sentinel for "no override installed here" — readers fall back to
# the process-wide baseline held on the instance.
_ctx_override: ContextVar[Context | None] = ContextVar("spacecore_ctx", default=None)
_check_override: ContextVar[CheckLevel | None] = ContextVar(
    "spacecore_check_level", default=None
)


class Contextual:
    """Resolve which context and validation level are currently in effect.

    Holds **ambient policy only** — the process-wide baseline context and check
    level, each read through a scoped :class:`~contextvars.ContextVar` override
    before falling back to the baseline stored here.

    *Which backends exist* is a different kind of state and lives in
    :class:`~spacecore.backend.OpsRegistry`: a registry is shared by every thread
    and must never be scoped (a backend registered inside a ``with`` block that
    vanished on exit would be a bug), whereas ambient policy is swappable by
    design. This object consults the registry; it does not own it.

    Parameters
    ----------
    ops_registry : OpsRegistry, optional
        Backend registry to resolve names against. Defaults to the process-wide
        instance; injectable so tests can resolve against their own.
    """

    _baseline_ctx: Context
    _default_dtype: DType | None = None
    # Baseline ambient validation level seeded onto newly created bound objects.
    # check_level is a property of the bound object, not of the Context.
    _baseline_check_level: CheckLevel = "standard"

    def __init__(self, ops_registry: OpsRegistry | None = None) -> None:
        ops = NumpyOps()
        self._baseline_ctx = Context(ops=ops, dtype=ops.sanitize_dtype(self._default_dtype))
        self._baseline_check_level = type(self)._baseline_check_level
        self._ops_registry = ops_registry if ops_registry is not None else _default_ops_registry

    @property
    def ops_registry(self) -> OpsRegistry:
        """Backend registry this resolver consults."""
        return self._ops_registry

    def get_check_level(self) -> CheckLevel:
        """Return the active ambient validation level for new bound objects.

        Resolution order: the scoped override installed by
        :meth:`scoped_check_level`, then the process-wide baseline.
        """
        level = _check_override.get()
        return level if level is not None else self._baseline_check_level

    def set_check_level(self, level: CheckLevel | bool | None) -> None:
        """Set the process-wide *baseline* validation level for new bound objects.

        Deliberately does not disturb scoped overrides: a ``scoped_check_level``
        block already in progress keeps winning until it exits.
        """
        self._baseline_check_level = normalize_check_level(level)

    @contextmanager
    def scoped_check_level(self, level: CheckLevel | bool | None) -> Iterator[CheckLevel]:
        """Override the ambient validation level for the duration of the block.

        The override is visible only to the current thread / async task. The
        :class:`~contextvars.Token` returned by ``set`` unwinds exactly this
        override, so nesting and concurrent scopes cannot clobber one another.
        """
        resolved = normalize_check_level(level)
        token = _check_override.set(resolved)
        try:
            yield resolved
        finally:
            _check_override.reset(token)

    def normalize_context(
        self,
        ctx: Context | BackendFamily | str | None = None,
        dtype: Any = None,
    ) -> Context:
        if ctx is None:
            if dtype is not None:
                warn("Provided context is None; dtype is ignored.", UserWarning)
            return self.default_ctx
        if isinstance(ctx, Context):
            if dtype is not None:
                warn("Provided concrete context; dtype is ignored.", UserWarning)
            return Context(ops=ctx.ops, dtype=ctx.ops.sanitize_dtype(ctx.dtype))
        if isinstance(ctx, (str, BackendFamily)):
            ops = self._ops_registry.get(ctx)
            return self.ctx_from_ops(ops, dtype=dtype)
        else:
            raise TypeError(f"Expected Context, BackendFamily, str, or None, got {type(ctx)}.")

    def ctx_from_ops(self, ops: BackendOps, dtype: DType | None = None) -> Context:
        return Context(ops=ops, dtype=ops.sanitize_dtype(dtype))

    @property
    def default_ctx(self) -> Context:
        """Active default context: the scoped override if one is installed here,
        otherwise the process-wide baseline.

        Every internal resolution path (:meth:`normalize_context` with ``None``,
        :meth:`resolve_context_priority`, :meth:`infer_context`) reads through this
        property, so a scoped override is honoured everywhere rather than only at
        the public ``get_context`` boundary.
        """
        ctx = _ctx_override.get()
        return ctx if ctx is not None else self._baseline_ctx

    @default_ctx.setter
    def default_ctx(self, ctx: Context | BackendFamily | str | None = None) -> None:
        """Set the process-wide *baseline* context; scoped overrides still win."""
        self._baseline_ctx = self.normalize_context(ctx)

    @contextmanager
    def scoped_ctx(
        self,
        ctx: Context | BackendFamily | str | None = None,
        dtype: Any = None,
    ) -> Iterator[Context]:
        """Override the default context for the duration of the block.

        The override is visible only to the current thread / async task; see
        :meth:`scoped_check_level` for the Token-unwind rationale.
        """
        resolved = self.normalize_context(ctx, dtype=dtype)
        token = _ctx_override.set(resolved)
        try:
            yield resolved
        finally:
            _ctx_override.reset(token)

    def infer_context(self, x: Any) -> Context | None:
        """Infer context from `.ctx` first, then registered backend arrays.

        The reverse lookup — which registered backend claims this array — is the
        registry's job (:meth:`~spacecore.backend.OpsRegistry.match`); turning the
        match into a :class:`Context` is this object's.
        """
        if isinstance(x, Context):
            return x

        ctx = getattr(x, "ctx", None)
        if isinstance(ctx, Context):
            return ctx

        matched = self._ops_registry.match(x)

        if not matched:
            return None
        if len(matched) > 1:
            raise ContextInferenceError(
                f"Ambiguous backend inference for object of type {type(x)!r}: {matched!r}."
            )

        ops = matched[0]
        try:
            dtype = ops.get_dtype(x)
        except Exception:
            dtype = getattr(x, "dtype", self.default_ctx.dtype)

        return self.ctx_from_ops(ops, dtype)

    def infer_contexts(self, values: Iterable[Any]) -> Tuple[Context, ...]:
        out: list[Context] = []
        for x in values:
            ctx = self.infer_context(x)
            if ctx is not None:
                out.append(ctx)
        return tuple(out)

    def are_compatible_contexts(self, *ctxs: Context) -> bool:
        if len(ctxs) < 2:
            return True
        first = ctxs[0]
        return all(ctx.same_backend(first) for ctx in ctxs[1:])

    def are_compatible_values(self, *values: Any) -> bool:
        return self.are_compatible_contexts(*self.infer_contexts(values))

    def are_compatible_ops(self, *ops: BackendOps) -> bool:
        if not ops:
            return True
        first = ops[0]
        return all(op == first for op in ops)

    def enforce_convert_policy(
        self, x: Any, to: Context | BackendFamily | str | None = None
    ) -> Tuple[Any, Context]:
        """Resolve the target context for ``x``."""
        self.infer_context(x)
        ctx = self.normalize_context(to)
        return x, ctx

    def resolve_context_priority(
        self,
        priority_ctx: Context | BackendFamily | str | None = None,
        *other_ctx: object,
    ) -> Context:
        """Resolve explicit context first, then compatible inferred contexts."""
        if priority_ctx is not None:
            return self.normalize_context(priority_ctx)

        inferred = self.infer_contexts(other_ctx)
        if not inferred:
            return self.default_ctx

        if not self.are_compatible_contexts(*inferred):
            fams = tuple(ctx.ops.family for ctx in inferred)
            raise ValueError(f"Incompatible inferred contexts: {fams!r}")

        first = inferred[0]
        ops = type(first.ops)()
        dtype = self._join_dtypes(ops, *(ctx.dtype for ctx in inferred))

        return self.ctx_from_ops(ops=ops, dtype=dtype)

    def _join_dtypes(self, ops: BackendOps, *dtypes: DType | None) -> DType | None:
        clean = [ops.sanitize_dtype(dt) for dt in dtypes if dt is not None]
        if not clean:
            return ops.sanitize_dtype(None)

        # Promote through the operands' OWN backend namespace. NumPy's
        # ``result_type`` cannot interpret a torch/jax dtype, so joining the
        # inferred contexts of a non-NumPy operator (for example a
        # ``BlockDiagonalLinOp`` built with ``from_operators``, which infers a
        # ``TreeSpace`` and joins its leaf dtypes) would otherwise raise
        # ``TypeError: Cannot interpret 'torch.float64' as a data type``.
        joined = ops.xp.result_type(*clean)
        return ops.sanitize_dtype(joined)
