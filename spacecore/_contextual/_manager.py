from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..backend._family import BackendFamily
from ..backend._ops import BackendOps
from ._policies import ContextPolicy, DtypePreservePolicy

if TYPE_CHECKING:
    from ..backend._context import Context


_cached_state = None


def _state():
    global _cached_state
    if _cached_state is not None:
        return _cached_state
    from ._state import _contextual

    _cached_state = _contextual
    return _cached_state


def set_context(
        ctx: Context | BackendFamily | str | None = None,
        dtype: Any = None,
        enable_checks: bool | None = None
) -> None:
    """
    Set the process-wide default SpaceCore context.

    Parameters
    ----------
    ctx:
        Context specification to make default. This may be a concrete
        :class:`spacecore.backend.Context`, a backend family enum, a backend
        family string such as ``"numpy"`` or ``"jax"``, or ``None``.
    dtype:
        Optional dtype used when ``ctx`` is a backend family string or enum.
        Ignored when ``ctx`` is ``None`` or already a concrete ``Context``.
    enable_checks:
        Optional validation flag used when constructing a context from a backend
        family. Ignored when ``ctx`` is ``None`` or already a concrete
        ``Context``.

    Notes
    -----
    Objects created without an explicit context use this default context.
    Existing spaces, operators, and contexts are not modified.
    """
    ctx = _state().normalize_context(ctx, dtype=dtype, enable_checks=enable_checks)
    _state().default_ctx = ctx


def get_context() -> Context:
    """
    Return the current process-wide default SpaceCore context.

    Returns
    -------
    Context
        The default context used by constructors when no explicit context can
        be inferred or provided.
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
    priority_ctx:
        Explicit context supplied by the caller. If this is not ``None``, it
        wins over every inferred context.
    *other_ctx:
        Objects that may carry a ``ctx`` attribute or be backend-native arrays.
        These are used for context inference when no explicit context is
        supplied.

    Returns
    -------
    Context
        The resolved context.

    Notes
    -----
    This is the public entry point for SpaceCore's context-priority resolution.
    User code should call this function instead of accessing the internal
    context manager singleton.
    """
    return _state().resolve_context_priority(priority_ctx, *other_ctx)


def register_ops(ops: type[BackendOps]) -> type[BackendOps]:
    """
    Register a backend operations implementation.

    Parameters
    ----------
    ops:
        Backend operations class to register. It must be a subclass of
        :class:`spacecore.backend.BackendOps` and define a unique backend
        family key.

    Returns
    -------
    type[BackendOps]
        The registered class. Returning the class allows this function to be
        used as a decorator.

    Raises
    ------
    TypeError
        If ``ops`` is not a ``BackendOps`` subclass.
    ContextConflictError
        If another backend with the same family key is already registered.

    Examples
    --------
    ``register_ops`` can be used as a decorator::

        @register_ops
        class MyOps(BackendOps):
            ...
    """
    return _state().register_ops(ops)


def normalize_context(
    ctx: Context | BackendFamily | str | None = None,
    dtype: Any = None,
    enable_checks: bool | None = None,
) -> Context:
    """Normalize a context specification through the process-wide state."""
    return _state().normalize_context(ctx, dtype=dtype, enable_checks=enable_checks)


def normalize_ops(
    ops: str | BackendFamily | BackendOps | type[BackendOps] | Context
) -> BackendOps:
    """Normalize backend operations through the process-wide state."""
    if isinstance(ops, BackendOps):
        return ops
    return _state().get_ops(ops)


def enforce_convert_policy(
    x: Any,
    to: Context | BackendFamily | str | None = None,
) -> tuple[Any, Context]:
    """Resolve a conversion target and enforce the configured policy."""
    return _state().enforce_convert_policy(x, to)


def set_resolution_policy(policy: ContextPolicy | str | None = None) -> None:
    """
    Set the policy for cross-backend context conversion.

    Parameters
    ----------
    policy:
        Conversion policy to use. Accepted values are ``"warning"``,
        ``"error"``, ``"silent"``, matching :class:`ContextPolicy`, or
        ``None`` to restore the default policy.

    Notes
    -----
    The resolution policy is consulted when SpaceCore detects conversion from
    one backend family to another. It does not prevent explicit construction of
    contexts.

    Policy values are:

    * ``"warning"``: allow backend conversion and issue a warning.
    * ``"error"``: reject backend conversion.
    * ``"silent"``: allow backend conversion without warning.
    """
    _state().resolution_policy = policy


def get_resolution_policy() -> str:
    """
    Return the active cross-backend conversion policy.

    Returns
    -------
    str
        Policy name, one of ``"warning"``, ``"error"``, or ``"silent"``.
    """
    return _state().resolution_policy.value


def set_dtype_resolution_policy(
    policy: DtypePreservePolicy | str | None = None,
) -> None:
    """
    Set the policy for dtype handling during context conversion.

    Parameters
    ----------
    policy:
        Dtype policy to use. Accepted values are ``"keep_native"`` and
        ``"convert"``, matching :class:`DtypePreservePolicy`, or ``None`` to
        restore the default policy.

    Notes
    -----
    ``"keep_native"`` preserves the source dtype where possible when converting
    context-bound objects. ``"convert"`` follows normal target-context dtype
    resolution.

    Policy values are:

    * ``"keep_native"``: preserve the object's dtype by mapping it to an
      equivalent dtype in the target backend.
    * ``"convert"``: use the dtype provided by the resolved target context.
    """
    _state().dtype_resolution_policy = policy


def get_dtype_resolution_policy() -> str:
    """
    Return the active dtype conversion policy.

    Returns
    -------
    str
        Policy name, one of ``"keep_native"`` or ``"convert"``.
    """
    return _state().dtype_resolution_policy.value
