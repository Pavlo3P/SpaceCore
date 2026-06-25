"""Structural dispatch for the benchmarked-spec layer (ADR-016).

The dispatcher is the single place that selects an optimized
:class:`~spacecore.kernels.KernelSpec` for an instrumented call site. A call
site delegates *selection* — it names the operation *family* it is performing
(a ``dispatch_key``) and supplies the inline ``generic`` path as the fallback;
it does **not** name a spec. Operator and functional bodies hold no
structural-selection logic; it all lives here.

Selection walks the eligible specs registered under the key in descending
``priority`` order and returns the first whose ``applicable(*args)`` is ``True``
and whose memory cost (if any) fits the context budget. If none applies, the
``generic`` fallback runs. With dispatch ``off`` — the default — ``generic``
runs immediately, so a wired call site is result-identical to its pre-dispatch
inline path.

Three modes, resolved per call from the process-global default, an optional
context-scoped override, and the operand context's check level:

``off``
    Always run ``generic``. The regression baseline and the default until a
    key's specs are proven and benchmarked.
``on``
    Route to the applicable optimized spec; fall back to ``generic``.
``verify``
    Run both, assert agreement within the spec's ``rtol``/``atol`` (which is
    ``0``/``0`` for every eligible spec — exact), raise on mismatch, return the
    optimized result. ADR-014 ``check_level="strict"`` implies ``verify``.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Literal

from ._policy import KernelCost

DispatchMode = Literal["off", "on", "verify"]
"""Dispatcher state: ``off`` (default), ``on``, or ``verify``."""

_MODES: tuple[DispatchMode, ...] = ("off", "on", "verify")

# Process-global default. The context-scoped override below takes precedence
# when set; ``check_level="strict"`` forces ``verify`` over either.
_global_mode: DispatchMode = "off"
_mode_override: ContextVar["DispatchMode | None"] = ContextVar(
    "spacecore_dispatch_mode", default=None
)

# Fraction of backend-reported free memory a materializing fast path may use.
_budget_fraction: float = 0.9


class DispatchVerificationError(AssertionError):
    """Raised in ``verify`` mode when optimized and generic results disagree."""


def _check_mode(mode: str) -> DispatchMode:
    if mode not in _MODES:
        allowed = ", ".join(repr(m) for m in _MODES)
        raise ValueError(f"Unknown dispatch mode {mode!r}. Expected one of: {allowed}.")
    return mode  # type: ignore[return-value]


def get_dispatch_mode() -> DispatchMode:
    """Return the active dispatch mode (context override over global default)."""
    override = _mode_override.get()
    return override if override is not None else _global_mode


def set_dispatch_mode(mode: DispatchMode) -> None:
    """Set the process-global default dispatch mode.

    Parameters
    ----------
    mode : {"off", "on", "verify"}
        New global default. A context-scoped :func:`dispatch_mode` override,
        where active, still takes precedence.
    """
    global _global_mode
    _global_mode = _check_mode(mode)


@contextmanager
def dispatch_mode(mode: DispatchMode) -> Iterator[None]:
    """Temporarily override the dispatch mode for the enclosing context.

    Restores the previous override on exit. Nestable and async/thread-safe
    (the override is a :class:`~contextvars.ContextVar`).

    Examples
    --------
    >>> from spacecore.kernels import dispatch_mode, get_dispatch_mode
    >>> with dispatch_mode("on"):
    ...     mode = get_dispatch_mode()  # wired sites route to optimized specs
    >>> mode
    'on'
    """
    token = _mode_override.set(_check_mode(mode))
    try:
        yield
    finally:
        _mode_override.reset(token)


def get_memory_budget_fraction() -> float:
    """Return the fraction of free memory a materializing fast path may use."""
    return _budget_fraction


def set_memory_budget_fraction(fraction: float) -> None:
    """Set the memory-budget fraction (in ``(0, 1]``) for the memory gate."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError("memory budget fraction must be in (0, 1].")
    global _budget_fraction
    _budget_fraction = fraction


def _is_strict(ctx: Any) -> bool:
    return ctx is not None and getattr(ctx, "check_level", None) == "strict"


def effective_mode(ctx: Any = None) -> DispatchMode:
    """Resolve the dispatch mode for an operand context.

    ``check_level="strict"`` (ADR-014) implies ``verify``, overriding both the
    context override and the global default — the strictest policy always runs
    the optimized/generic agreement check.
    """
    if _is_strict(ctx):
        return "verify"
    return get_dispatch_mode()


def should_consult_dispatch(ctx: Any = None) -> bool:
    """Return whether a wired call site must consult the dispatcher.

    ``False`` in the default (``off``, non-strict) state, so a guarded call
    site keeps its plain inline path with only this one cheap check between it
    and today's behavior.
    """
    return effective_mode(ctx) != "off"


def _memory_budget(ctx: Any) -> "int | None":
    """Return the byte budget for a materializing fast path, or ``None``.

    ``None`` means the budget is unknown (no context, or the backend does not
    report free memory). The dispatcher treats an unknown budget as a hard stop
    for any spec carrying a :class:`KernelCost`: a materializing fast path is
    selected only against a known budget.
    """
    if ctx is None:
        return None
    ops = getattr(ctx, "ops", None)
    if ops is None:
        return None
    free = ops.free_memory_bytes()
    if free is None:
        return None
    return int(free * _budget_fraction)


def _affordable(estimate: KernelCost | None, ctx: Any) -> bool:
    """Decide a materializing fast path's affordability from its cost estimate.

    Called only for specs that *carry* a cost estimator (the dispatcher selects
    a spec with no estimator unconditionally — it is non-materializing).
    ``estimate`` is the estimator's result: a materializing spec is affordable
    only when it produced an estimate *and* that estimate fits a known budget —
    **no estimate, no fuse**, and no budget, no fuse.
    """
    if estimate is None:
        return False  # no estimate, no fuse
    budget = _memory_budget(ctx)
    if budget is None:
        return False  # no budget, no fuse for a materializing path
    return estimate.peak_bytes <= budget


def _looks_like_array(x: Any) -> bool:
    """Duck-typed array test that does not need backend ops."""
    return hasattr(x, "shape")


def _results_close(
    optimized: Any, generic: Any, rtol: float, atol: float, ctx: Any
) -> bool:
    """Structural agreement check for ``verify`` mode.

    Recurses through tuples/lists (structured operator outputs) and compares
    backend arrays elementwise; falls back to ``==`` only for genuine scalars.
    With ``rtol == atol == 0`` this is exact equality.

    Array comparison prefers the context's ``ops.allclose``. When no context is
    supplied (``dispatch`` is called without ``ctx``), it falls back to a
    shape-aware NumPy compare so that array results — whose ``==`` is itself an
    array — never reach the scalar ``bool(a == b)`` path, which would raise on a
    multi-element array and silently degrade to an identity check.
    """
    if isinstance(optimized, (tuple, list)) and isinstance(generic, (tuple, list)):
        return len(optimized) == len(generic) and all(
            _results_close(o, g, rtol, atol, ctx)
            for o, g in zip(optimized, generic)
        )
    ops = getattr(ctx, "ops", None) if ctx is not None else None
    if ops is not None and ops.is_array(optimized) and ops.is_array(generic):
        return ops.allclose(optimized, generic, rtol=rtol, atol=atol)
    if _looks_like_array(optimized) or _looks_like_array(generic):
        import numpy as np

        try:
            return bool(
                np.allclose(np.asarray(optimized), np.asarray(generic), rtol=rtol, atol=atol)
            )
        except Exception:
            return optimized is generic
    try:
        return bool(optimized == generic)
    except Exception:
        return optimized is generic


def dispatch(
    key: str,
    *args: Any,
    generic: Callable[..., Any],
    ctx: Any = None,
) -> Any:
    """Select and run an optimized spec for ``key``, else run ``generic``.

    Parameters
    ----------
    key : str
        The ``dispatch_key`` the call site requests. Only dispatch-eligible
        specs registered under this key are considered.
    *args
        The operands, passed verbatim to each candidate's ``applicable`` /
        ``cost`` / ``optimized`` and to ``generic``.
    generic : callable
        The inline fallback — the call site's own pre-dispatch path. Runs when
        dispatch is ``off``, when no eligible spec applies/fits, and (in
        ``verify``) as the value the optimized result is checked against.
    ctx : optional
        The operand execution context. Supplies the check level (for the
        ``strict`` → ``verify`` rule) and the backend (for the memory gate).

    Returns
    -------
    The optimized result when a spec is selected, otherwise ``generic(*args)``.
    """
    mode = effective_mode(ctx)
    if mode == "off":
        return generic(*args)

    from ._registry import registry

    for spec in registry.dispatch_candidates(key):
        try:
            is_applicable = spec.applicable(*args)
        except Exception as exc:  # noqa: BLE001 — a buggy predicate must not crash apply
            # Per the Applicability contract, a raised predicate is treated as
            # "not applicable, plus a bug to fix": skip the spec and surface it.
            warnings.warn(
                f"dispatch: applicable() of kernel {spec.name!r} raised {exc!r}; "
                f"treating it as not applicable. This indicates a bug in the spec.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not is_applicable:
            continue
        if spec.cost is not None and not _affordable(spec.cost(*args), ctx):
            continue
        if mode == "on":
            return spec.optimized(*args)
        # verify: run both, assert exact agreement, raise on mismatch.
        optimized = spec.optimized(*args)
        reference = generic(*args)
        if not _results_close(optimized, reference, spec.rtol, spec.atol, ctx):
            raise DispatchVerificationError(
                f"dispatch verify mismatch for key {key!r}: optimized kernel "
                f"{spec.name!r} disagrees with the generic fallback within "
                f"rtol={spec.rtol}, atol={spec.atol}."
            )
        return optimized

    return generic(*args)
