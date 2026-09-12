from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from ._context import Context
from ._contextual import Contextual
from .._check_policy import CheckLevel
from ..backend import BackendFamily, BackendOps
from ..backend import ops_registry as _ops_registry


_contextual: Contextual | None = None


def _state() -> Contextual:
    """Return the process-wide contextual singleton."""
    global _contextual
    if _contextual is None:
        _contextual = Contextual()
    return _contextual


def set_context(
    ctx: Context | BackendFamily | str | None = None,
    dtype: Any = None,
) -> None:
    """
    Set the process-wide *baseline* SpaceCore context.

    Visible to every thread and async task. A :func:`use_context` override already
    in progress keeps winning until its block exits.

    Parameters
    ----------
    ctx : Context, BackendFamily, str, or None, optional
        Context or backend specification.
    dtype : Any, optional
        Default dtype override.
    """
    state = _state()
    state.default_ctx = state.normalize_context(ctx, dtype=dtype)


def get_check_level() -> CheckLevel:
    """
    Return the ambient default validation level applied to new bound objects.

    Returns
    -------
    CheckLevel
        The level that seeds a context-bound object constructed without an
        explicit ``check_level``: the scoped override installed for this thread
        or task if there is one, otherwise the process-wide baseline.
    """
    return _state().get_check_level()


def set_check_level(level: CheckLevel | bool | None) -> None:
    """
    Set the ambient default validation level applied to new bound objects.

    ``check_level`` is a property of the context-bound object (space, operator,
    functional), not of the :class:`Context`. This sets the process-wide default
    that seeds a new object when it is constructed without an explicit level.

    Parameters
    ----------
    level : {"none", "cheap", "standard", "strict"} or None
        New process-wide baseline. ``None`` restores the built-in default.
    """
    _state().set_check_level(level)


@contextmanager
def use_check_level(level: CheckLevel | bool | None):
    """
    Temporarily override the ambient default validation level within a ``with`` block.

    Scoped to the current thread / async task; see :func:`use_context` for the
    scoping rules and the thread-pool caveat.

    Parameters
    ----------
    level : {"none", "cheap", "standard", "strict"} or None
        Level to install for the duration of the block. ``None`` leaves the
        ambient default in place.

    Yields
    ------
    CheckLevel
        The resolved level in effect inside the block.
    """
    with _state().scoped_check_level(level) as resolved:
        yield resolved


@contextmanager
def use_context(
    ctx: Context | BackendFamily | str | None = None,
    dtype: Any = None,
):
    """
    Temporarily override the default context within a ``with`` block.

    Yields the resolved :class:`Context` and unwinds on exit, so the override is
    exception-safe and never leaks — the dependency-injection-friendly counterpart
    to :func:`set_context`.

    Unlike :func:`set_context`, which moves the process-wide baseline seen by every
    thread, this override is stored in a :class:`~contextvars.ContextVar` and is
    visible only to the **current thread and async task**. Each ``asyncio`` task
    receives its own copy of the ambient context at creation, so interleaved
    coroutines cannot clobber one another's override.

    Parameters
    ----------
    ctx : Context, BackendFamily, str, or None, optional
        Context to install for the duration of the block, or a backend family
        name (for example ``"numpy"`` or ``"jax"``) to resolve into one. ``None``
        keeps the active context and only applies ``dtype``.
    dtype : dtype-like, optional
        Representation dtype for the scoped context. ``None`` keeps the dtype of
        the context being overridden.

    Yields
    ------
    Context
        The resolved context in effect inside the block.

    Notes
    -----
    ``ContextVar`` values are deliberately *not* inherited by threads started with
    :class:`threading.Thread`, so a scoped override does not silently leak into
    workers you spawn::

        with use_context("jax"):
            list(executor.map(solve, problems))   # workers see the BASELINE

    To propagate it on purpose, snapshot the context and run the worker inside it::

        from contextvars import copy_context

        with use_context("jax"):
            snapshot = copy_context()
            list(executor.map(lambda p: snapshot.run(solve, p), problems))

    ``asyncio.create_task`` needs no such handling.
    """
    with _state().scoped_ctx(ctx, dtype=dtype) as resolved:
        yield resolved


def get_context() -> Context:
    """
    Return the currently active default SpaceCore context.

    Resolves the :func:`use_context` override for this thread / task if one is
    installed, otherwise the process-wide baseline set by :func:`set_context`.

    Returns
    -------
    Context
        Active default context.
    """
    return _state().default_ctx


def resolve_context_priority(
    priority_ctx: Context | BackendFamily | str | None = None,
    *other_ctx: object,
) -> Context:
    """
    Resolve the context assigned to a newly created object.

    Parameters
    ----------
    priority_ctx : Context, BackendFamily, str, or None, optional
        Explicit context that takes precedence when provided.
    *other_ctx : object
        Objects or contexts used as fallback context sources.

    Returns
    -------
    Context
        Resolved context.
    """
    return _state().resolve_context_priority(priority_ctx, *other_ctx)


def register_ops(ops: type[BackendOps]) -> type[BackendOps]:
    """
    Register a backend operations implementation.

    Writes to the process-wide backend registry
    (:data:`spacecore.backend.ops_registry`), not to the ambient context state:
    a registration is shared by every thread and is never scoped.

    Parameters
    ----------
    ops : type of BackendOps
        Backend operations class to register.

    Returns
    -------
    type of BackendOps
        Registered backend operations class.

    Raises
    ------
    ContextConflictError
        If the backend family is already registered.
    """
    return _ops_registry.register(ops)


def normalize_context(
    ctx: Context | BackendFamily | str | None = None,
    dtype: Any = None,
) -> Context:
    """
    Normalize a context specification through the process-wide state.

    Parameters
    ----------
    ctx : Context, BackendFamily, str, or None, optional
        Context or backend specification.
    dtype : Any, optional
        Default dtype override.

    Returns
    -------
    Context
        Normalized context.
    """
    return _state().normalize_context(ctx, dtype=dtype)


def normalize_ops(ops: str | BackendFamily | BackendOps | type[BackendOps] | Context) -> BackendOps:
    """
    Normalize backend operations through the process-wide state.

    Parameters
    ----------
    ops : str, BackendFamily, BackendOps, type of BackendOps, or Context
        Backend operations specification.

    Returns
    -------
    BackendOps
        Normalized backend operations singleton.
    """
    if isinstance(ops, BackendOps):
        return ops
    return _ops_registry.get(ops)


def enforce_convert_policy(
    x: Any,
    to: Context | BackendFamily | str | None = None,
) -> tuple[Any, Context]:
    """Resolve a conversion target context."""
    return _state().enforce_convert_policy(x, to)
