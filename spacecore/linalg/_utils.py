from __future__ import annotations

from math import prod
from typing import Any

from ..linop import LinOp

DEFAULT_CONVERGENCE_CHECK_INTERVAL = 64


def require_linop(A: Any) -> LinOp:
    """Return ``A`` as a ``LinOp`` or raise a clear type error."""
    if not isinstance(A, LinOp):
        raise TypeError(f"A must be a LinOp, got {type(A).__name__}.")
    return A


def require_square(A: LinOp, name: str) -> None:
    """Raise if ``A`` is not a square operator."""
    if A.domain != A.codomain:
        raise ValueError(f"{name} requires a square LinOp; got {A.domain!r} -> {A.codomain!r}.")


def require_strict_cg_preconditions(A: LinOp) -> None:
    """Run expensive development-time Hermitian and positive-curvature probes."""
    if not A._checks_at_least("strict"):
        return
    if A.is_hermitian() is False:
        raise ValueError("cg strict checks require A to be Hermitian/self-adjoint.")

    x = default_initial_vector(A)
    curvature = real_inner(A.domain, x, A.apply(x))
    try:
        positive = bool(curvature > 0)
    except Exception as exc:
        raise ValueError("cg strict positive-definiteness probe could not be evaluated.") from exc
    if not positive:
        raise ValueError("cg strict checks require positive curvature on the probe vector.")


def default_maxiter(A: LinOp) -> int:
    """Return the default Krylov iteration count for ``A``."""
    return max(1, prod(A.domain.shape))


def check_maxiter(maxiter: int | None, A: LinOp) -> int:
    """Validate an optional iteration count."""
    if maxiter is None:
        return default_maxiter(A)
    maxiter = int(maxiter)
    if maxiter < 0:
        raise ValueError("maxiter must be nonnegative.")
    return maxiter


def check_interval(interval: int) -> int:
    """Validate a convergence-check interval."""
    interval = int(interval)
    if interval < 1:
        raise ValueError("check_every must be positive.")
    return interval


def should_check_iteration(k: Any, maxiter: int, interval: int) -> Any:
    """Return whether iteration ``k`` should refresh convergence diagnostics."""
    return (k >= maxiter) | ((k % interval) == 0)


def threshold(norm_b: Any, tol: float, atol: float) -> Any:
    """Return the absolute-plus-relative convergence threshold."""
    return max(float(atol), 0.0) + max(float(tol), 0.0) * norm_b


# ---------------------------------------------------------------------------
# Check-free primitives for solver hot loops.
#
# Iterative solvers validate their operator and right-hand side once, at entry.
# Their iterates (search directions, residuals, Krylov vectors) are then
# generated internally and are valid by construction, so the per-iteration
# membership checks the public ``apply`` / space ops run are pure overhead
# (measured at well over half of CG's NumPy runtime). These helpers route the
# hot loop through the operators' and spaces' check-free ``_*_core`` kernels.
#
# Safety: a core is used only when it is *consistent* with the public method —
# i.e. the public method is not overridden by a subclass below the class that
# defines the core. That guards against a user space that overrides ``inner``
# (custom geometry) while inheriting a built-in ``_inner_core``; there the
# override wins and the checked public method is used. Resolution is done once,
# at solver entry, so the hot loop pays neither the check nor the lookup.
# ---------------------------------------------------------------------------
def _defining_class(cls: type, name: str) -> type | None:
    """Return the first class in ``cls``'s MRO whose ``__dict__`` defines ``name``."""
    for base in cls.__mro__:
        if name in base.__dict__:
            return base
    return None


def resolve_core(obj: Any, public_name: str, core_name: str) -> Any:
    """Return ``obj``'s bound check-free core when it is safe to use.

    The core is safe when ``public_name`` is not overridden in a subclass below
    the class that defines ``core_name`` (so the core cannot silently bypass a
    subclass's custom semantics). Otherwise the bound public method is returned.
    """
    cls = type(obj)
    core_owner = _defining_class(cls, core_name)
    if core_owner is None:
        return getattr(obj, public_name)
    pub_owner = _defining_class(cls, public_name)
    if pub_owner is None or issubclass(core_owner, pub_owner):
        return getattr(obj, core_name)
    return getattr(obj, public_name)


class SpaceCoreOps:
    """Resolved check-free linear-space ops for a solver's hot loop.

    Each op is resolved once (via :func:`resolve_core`) to the space's check-free
    core when consistent, or the checked public method otherwise — so the hot
    loop skips per-iteration validation while still honoring any custom geometry
    a user space defines. ``axpy`` and ``norm`` are derived from the resolved
    ``add``/``scale``/``inner`` to match the space's definitions.
    """

    __slots__ = ("ops", "_inner", "_add", "_scale")

    def __init__(self, space: Any) -> None:
        self.ops = space.ops
        self._inner = resolve_core(space, "inner", "_inner_core")
        self._add = resolve_core(space, "add", "_add_core")
        self._scale = resolve_core(space, "scale", "_scale_core")

    def inner(self, x: Any, y: Any) -> Any:
        return self._inner(x, y)

    def real_inner(self, x: Any, y: Any) -> Any:
        return self.ops.real(self._inner(x, y))

    def add(self, x: Any, y: Any) -> Any:
        return self._add(x, y)

    def scale(self, a: Any, x: Any) -> Any:
        return self._scale(a, x)

    def axpy(self, a: Any, x: Any, y: Any) -> Any:
        return self._add(self._scale(a, x), y)

    def norm(self, x: Any) -> Any:
        return self.ops.sqrt(self.ops.real(self._inner(x, x)))


def resolve_apply(A: Any) -> Any:
    """Return ``A``'s bound check-free forward core (or checked ``apply``)."""
    return resolve_core(A, "apply", "_apply_core")


def resolve_rapply(A: Any) -> Any:
    """Return ``A``'s bound check-free adjoint core (or checked ``rapply``)."""
    return resolve_core(A, "rapply", "_rapply_core")


def real_inner(space: Any, x: Any, y: Any) -> Any:
    """Return the real part of ``space.inner(x, y)`` (checked; for one-shot use)."""
    return space.ops.real(space.inner(x, y))


def is_converged(residual_norm: Any, threshold_value: Any) -> Any:
    """Return backend-compatible convergence predicate."""
    return residual_norm <= threshold_value


def safe_inverse_nonneg(ops: Any, value: Any) -> Any:
    """
    Return ``1 / value`` where ``value > 0`` and zero otherwise.

    This helper is intended for norms and nonnegative residual magnitudes. It
    is not a general scalar inverse: for example, ``-2`` maps to ``0``, not
    ``-0.5``.
    """
    positive = value > 0
    safe_value = ops.where(positive, value, ops.ones_like(value))
    return ops.where(positive, 1.0 / safe_value, ops.zeros_like(value))


def normalize(space: Any, x: Any) -> tuple[Any, Any]:
    """Normalize a space member and return ``(unit, norm)``."""
    norm = space.norm(x)
    return space.scale(safe_inverse_nonneg(space.ops, norm), x), norm


def core_normalize(sops: SpaceCoreOps, x: Any) -> tuple[Any, Any]:
    """Normalize ``x`` through resolved check-free space ops; return ``(unit, norm)``."""
    norm = sops.norm(x)
    return sops.scale(safe_inverse_nonneg(sops.ops, norm), x), norm


def default_initial_vector(A: LinOp) -> Any:
    """Return a deterministic unit vector in ``A.domain`` using its geometry."""
    size = prod(A.domain.shape)
    flat = A.ops.ones((size,), dtype=A.dtype)
    v = A.domain.unflatten(flat)
    norm = A.domain.norm(v)
    return A.domain.scale(safe_inverse_nonneg(A.ops, norm), v)


def summarize_value(value: Any) -> str:
    """Return a compact representation for arrays, scalars, and pytrees."""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        shape_text = tuple(shape)
        if shape_text == ():
            dtype_text = str(dtype)
            if dtype_text in {"bool", "bool_", "torch.bool"}:
                try:
                    return repr(bool(value))
                except Exception:
                    return repr(value)
            try:
                return f"{float(value):.6g}"
            except Exception:
                return repr(value)
        dtype_text = "" if dtype is None else f", dtype={dtype}"
        return f"<array shape={shape_text}{dtype_text}>"
    if isinstance(value, tuple):
        return "(" + ", ".join(summarize_value(part) for part in value) + ")"
    return repr(value)


def result_repr(name: str, fields: dict[str, Any]) -> str:
    """Return a compact result-object representation."""
    body = ", ".join(f"{key}={summarize_value(value)}" for key, value in fields.items())
    return f"{name}({body})"
