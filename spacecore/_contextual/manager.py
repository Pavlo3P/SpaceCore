from typing import Any

from ..backend import Context, BackendOps
from .contextual import Contextual, ContextPolicy, DtypePreservePolicy
from ..backend import BackendFamily


ctx_manager = Contextual()


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
    ctx = ctx_manager.normalize_context(ctx, dtype=dtype, enable_checks=enable_checks)
    ctx_manager.default_ctx = ctx


def get_context() -> Context:
    """
    Return the current process-wide default SpaceCore context.

    Returns
    -------
    Context
        The default context used by constructors when no explicit context can
        be inferred or provided.
    """
    return ctx_manager.default_ctx


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
    return ctx_manager.register_ops(ops)


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
    ctx_manager.resolution_policy = policy


def get_resolution_policy() -> str:
    """
    Return the active cross-backend conversion policy.

    Returns
    -------
    str
        Policy name, one of ``"warning"``, ``"error"``, or ``"silent"``.
    """
    return ctx_manager.resolution_policy.value


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
    ctx_manager.dtype_resolution_policy = policy


def get_dtype_resolution_policy() -> str:
    """
    Return the active dtype conversion policy.

    Returns
    -------
    str
        Policy name, one of ``"keep_native"`` or ``"convert"``.
    """
    return ctx_manager.dtype_resolution_policy.value
