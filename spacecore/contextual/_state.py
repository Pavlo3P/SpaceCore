from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from ._context import Context
from ._contextual import Contextual
from .._checks import CheckLevel
from ..backend import BackendFamily, BackendOps
from ..backend import ops_registry as _ops_registry


_contextual: Contextual | None = None


def _state() -> Contextual:
    """Return the process-wide contextual singleton."""
    # Create the process-wide state lazily, on first use.
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


def set_check_level(level: CheckLevel | None) -> None:
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
def use_check_level(level: CheckLevel | None):
    """
    Override the ambient validation level inside a ``with`` block (thread/task-scoped, like :func:`use_context`).

    Parameters
    ----------
    level : {"none", "cheap", "standard", "strict"} or None
        Level for the block; ``None`` keeps the ambient one.

    Yields
    ------
    CheckLevel
        The level in effect inside the block.
    """
    with _state().scoped_check_level(level) as resolved:
        yield resolved


@contextmanager
def use_context(
    ctx: Context | BackendFamily | str | None = None,
    dtype: Any = None,
):
    """
    Override the default context inside a ``with`` block.

    The override lives in a :class:`~contextvars.ContextVar`: it is seen by the current
    thread and asyncio task only, and is undone on exit. Threads started inside the block
    see the baseline; run them under ``contextvars.copy_context()`` to pass it on.

    Parameters
    ----------
    ctx : Context, BackendFamily, str, or None, optional
        Context, or a backend family name such as ``"jax"``. ``None`` keeps the active
        context and applies only ``dtype``.
    dtype : dtype-like, optional
        Dtype for the scoped context; ``None`` keeps the current one.

    Yields
    ------
    Context
        The context in effect inside the block.
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
    Register a backend in the process-wide registry; registrations are never scoped.

    Parameters
    ----------
    ops : type of BackendOps
        Backend class to register.

    Returns
    -------
    type of BackendOps
        The registered class.

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
