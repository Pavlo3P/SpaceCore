"""Matrix-free CG on a 2D Poisson + identity system.

Workload
--------

Solve

.. math::
    (-\\Delta + \\lambda I) u = f

on an ``n x n`` grid with zero Dirichlet boundary conditions, where
``-\\Delta`` is the 5-point stencil

.. math::
    (-\\Delta u)_{ij} = 4 u_{ij} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1},

with ``u`` zero outside the grid. ``lambda_ = 0.01`` keeps the system
strictly SPD so plain CG converges. The right-hand side is the discrete
sine bump ``f_{ij} = sin(pi i / n) sin(pi j / n)``.

The matrix is never materialized. CG only needs the action ``u -> A u``,
which the bare path implements directly in backend-native ops and the
SpaceCore path wraps in :class:`spacecore.MatrixFreeLinOp`.

Modes
-----

``bare``
    Backend-native CG loop with a backend-native stencil. 100 iterations
    fixed, no convergence check.
``spacecore_public_none``
    :func:`spacecore.cg` on a :class:`spacecore.MatrixFreeLinOp` over a
    :class:`spacecore.DenseCoordinateSpace`, with the context's
    ``check_level="none"``.
``spacecore_public_cheap``
    Same as ``public_none`` with ``check_level="cheap"``.
``spacecore_lowered``
    On JAX, the stencil ``apply`` is wrapped in :func:`jax.jit` so the
    full matvec runs as one compiled kernel; CG itself stays in eager
    SpaceCore code but every matvec is fused. On NumPy/Torch there is no
    distinct lowered path, so the lowered callable is the same as
    ``public_none``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

import spacecore as sc

from bench._operations import _backend_ctx, _np_dtype
from ._registry import MacroBenchmark, MacroPayload, registry
from ._schema import ModeName


# Throughput unit: one CG iteration performs one matvec.
_MATVECS_PER_ITER = 1.0


def _make_rhs_np(n: int, dtype: np.dtype) -> np.ndarray:
    """Build ``f_{ij} = sin(pi i / n) sin(pi j / n)`` as a NumPy array."""
    i = np.arange(n, dtype=np.float64)
    j = np.arange(n, dtype=np.float64)
    sx = np.sin(np.pi * i / n)
    sy = np.sin(np.pi * j / n)
    return np.asarray(np.outer(sx, sy), dtype=dtype)


# ---------------------------------------------------------------------------
# NumPy bare path


def _numpy_apply(u: np.ndarray, lambda_: float) -> np.ndarray:
    """5-point stencil ``4 u - up - down - left - right`` plus ``lambda u``."""
    u.shape[0]
    out = (4.0 + lambda_) * u
    # subtract neighbors with zero-padding outside the grid
    out[1:, :] -= u[:-1, :]
    out[:-1, :] -= u[1:, :]
    out[:, 1:] -= u[:, :-1]
    out[:, :-1] -= u[:, 1:]
    return out


def _numpy_cg(
    apply: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    maxiter: int,
) -> tuple[np.ndarray, float, int]:
    """Plain CG without convergence checks; ``maxiter`` iterations fixed."""
    x = np.zeros_like(b)
    r = b - apply(x)
    p = r.copy()
    rs_old = float(np.vdot(r.ravel(), r.ravel()).real)
    iters = 0
    for _ in range(maxiter):
        Ap = apply(p)
        pAp = float(np.vdot(p.ravel(), Ap.ravel()).real)
        if pAp == 0.0:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(np.vdot(r.ravel(), r.ravel()).real)
        if rs_old == 0.0:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
        iters += 1
    residual_norm = float(np.sqrt(rs_old))
    return x, residual_norm, iters


# ---------------------------------------------------------------------------
# Torch bare path


def _torch_apply_factory(lambda_: float):
    def apply(u):
        # u: 2-D tensor (n, n)
        out = (4.0 + lambda_) * u
        out[1:, :] = out[1:, :] - u[:-1, :]
        out[:-1, :] = out[:-1, :] - u[1:, :]
        out[:, 1:] = out[:, 1:] - u[:, :-1]
        out[:, :-1] = out[:, :-1] - u[:, 1:]
        return out
    return apply


def _torch_cg(apply, b, maxiter: int):
    import torch  # local import: only when the path is actually used

    x = torch.zeros_like(b)
    r = b - apply(x)
    p = r.clone()
    rs_old = torch.vdot(r.reshape(-1), r.reshape(-1)).real
    iters = 0
    for _ in range(maxiter):
        Ap = apply(p)
        pAp = torch.vdot(p.reshape(-1), Ap.reshape(-1)).real
        if float(pAp) == 0.0:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real
        if float(rs_old) == 0.0:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
        iters += 1
    residual_norm = float(torch.sqrt(rs_old))
    return x, residual_norm, iters


# ---------------------------------------------------------------------------
# JAX bare path


def _jax_apply_factory(lambda_: float):
    import jax.numpy as jnp

    def apply(u):
        # u: (n, n) jax array; pad with zeros to apply the stencil.
        # Each shifted slice corresponds to a neighbor; missing neighbors
        # at the boundary become zero via padding.
        n = u.shape[0]
        # Build padded array of zeros and place u in the interior.
        padded = jnp.zeros((n + 2, n + 2), dtype=u.dtype)
        padded = padded.at[1:-1, 1:-1].set(u)
        up = padded[0:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, 0:-2]
        right = padded[1:-1, 2:]
        return (4.0 + lambda_) * u - up - down - left - right

    return apply


def _jax_cg(apply, b, maxiter: int):
    import jax.numpy as jnp

    x = jnp.zeros_like(b)
    r = b - apply(x)
    p = r
    rs_old = jnp.vdot(r.reshape(-1), r.reshape(-1)).real
    iters = 0
    for _ in range(maxiter):
        Ap = apply(p)
        pAp = jnp.vdot(p.reshape(-1), Ap.reshape(-1)).real
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.vdot(r.reshape(-1), r.reshape(-1)).real
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
        iters += 1
    residual_norm = jnp.sqrt(rs_old)
    return x, residual_norm, iters


# ---------------------------------------------------------------------------
# Factory


def _factory(backend: str, device: str, seed: int, size_params: dict[str, Any]) -> MacroPayload:
    n = int(size_params["n"])
    maxiter = int(size_params["maxiter"])
    lambda_ = float(size_params["lambda"])

    # Two contexts, two callables (the runner does NOT switch context).
    ctx_none = _backend_ctx(backend, check_level="none")
    ctx_cheap = _backend_ctx(backend, check_level="cheap")
    np_dtype = _np_dtype(ctx_none)

    # Pre-compute the right-hand side once in NumPy, ship to each backend.
    b_np = _make_rhs_np(n, np_dtype)

    # SpaceCore space and operator for each check level.
    space_none = sc.DenseCoordinateSpace((n, n), ctx_none)
    space_cheap = sc.DenseCoordinateSpace((n, n), ctx_cheap)

    mode_callables: dict[ModeName, Callable[[], Any]] = {}

    if backend == "numpy":
        b_arr_none = ctx_none.asarray(b_np)
        b_arr_cheap = ctx_cheap.asarray(b_np)

        def bare_call(b=b_np, lam=lambda_, mi=maxiter):
            return _numpy_cg(lambda u: _numpy_apply(u, lam), b, mi)

        def sc_apply_none_factory(lam=lambda_):
            def apply(u):
                return _numpy_apply(u, lam)
            return apply

        apply_none = sc_apply_none_factory()
        apply_cheap = sc_apply_none_factory()

        op_none = sc.MatrixFreeLinOp(apply_none, apply_none, space_none, space_none, ctx_none)
        op_cheap = sc.MatrixFreeLinOp(apply_cheap, apply_cheap, space_cheap, space_cheap, ctx_cheap)

        def sc_none_call(op=op_none, b=b_arr_none, mi=maxiter):
            return sc.cg(op, b, maxiter=mi, tol=0.0, atol=0.0)

        def sc_cheap_call(op=op_cheap, b=b_arr_cheap, mi=maxiter):
            return sc.cg(op, b, maxiter=mi, tol=0.0, atol=0.0)

        mode_callables["bare"] = bare_call
        mode_callables["spacecore_public_none"] = sc_none_call
        mode_callables["spacecore_public_cheap"] = sc_cheap_call
        # No distinct lowered path on NumPy.
        mode_callables["spacecore_lowered"] = sc_none_call

    elif backend == "jax":
        import jax
        import jax.numpy as jnp

        jax_dtype = jnp.asarray(b_np).dtype
        jnp.asarray(b_np, dtype=ctx_none.dtype if hasattr(ctx_none, "dtype") else jax_dtype)
        # Re-cast in the active jax dtype for both contexts.
        b_arr_none = ctx_none.asarray(b_np)
        b_arr_cheap = ctx_cheap.asarray(b_np)

        apply_jax = _jax_apply_factory(lambda_)

        def bare_call(b=b_arr_none, mi=maxiter):
            return _jax_cg(apply_jax, b, mi)

        # SpaceCore public path uses the same eager apply.
        op_none = sc.MatrixFreeLinOp(apply_jax, apply_jax, space_none, space_none, ctx_none)
        op_cheap_apply = _jax_apply_factory(lambda_)
        op_cheap = sc.MatrixFreeLinOp(
            op_cheap_apply, op_cheap_apply, space_cheap, space_cheap, ctx_cheap
        )

        def sc_none_call(op=op_none, b=b_arr_none, mi=maxiter):
            return sc.cg(op, b, maxiter=mi, tol=0.0, atol=0.0)

        def sc_cheap_call(op=op_cheap, b=b_arr_cheap, mi=maxiter):
            return sc.cg(op, b, maxiter=mi, tol=0.0, atol=0.0)

        # Lowered: jit the apply so each matvec is a fused kernel.
        jit_apply = jax.jit(_jax_apply_factory(lambda_))
        # Warm the jit cache so the runner's compile-vs-steady split is on a
        # stable, pre-traced callable. We still want jax_full_loop / runner
        # to time the first call; the runner already separates that. Here we
        # let it count compile cost as compile_time_ns on the lowered mode.
        op_lowered = sc.MatrixFreeLinOp(
            jit_apply, jit_apply, space_none, space_none, ctx_none
        )

        def sc_lowered_call(op=op_lowered, b=b_arr_none, mi=maxiter):
            return sc.cg(op, b, maxiter=mi, tol=0.0, atol=0.0)

        mode_callables["bare"] = bare_call
        mode_callables["spacecore_public_none"] = sc_none_call
        mode_callables["spacecore_public_cheap"] = sc_cheap_call
        mode_callables["spacecore_lowered"] = sc_lowered_call

    elif backend == "torch":

        b_arr_none = ctx_none.asarray(b_np)
        b_arr_cheap = ctx_cheap.asarray(b_np)

        bare_apply = _torch_apply_factory(lambda_)

        def bare_call(b=b_arr_none, mi=maxiter):
            return _torch_cg(bare_apply, b, mi)

        apply_none = _torch_apply_factory(lambda_)
        apply_cheap = _torch_apply_factory(lambda_)
        op_none = sc.MatrixFreeLinOp(apply_none, apply_none, space_none, space_none, ctx_none)
        op_cheap = sc.MatrixFreeLinOp(apply_cheap, apply_cheap, space_cheap, space_cheap, ctx_cheap)

        def sc_none_call(op=op_none, b=b_arr_none, mi=maxiter):
            return sc.cg(op, b, maxiter=mi, tol=0.0, atol=0.0)

        def sc_cheap_call(op=op_cheap, b=b_arr_cheap, mi=maxiter):
            return sc.cg(op, b, maxiter=mi, tol=0.0, atol=0.0)

        mode_callables["bare"] = bare_call
        mode_callables["spacecore_public_none"] = sc_none_call
        mode_callables["spacecore_public_cheap"] = sc_cheap_call
        # No distinct lowered path on Torch.
        mode_callables["spacecore_lowered"] = sc_none_call

    else:
        raise ValueError(f"unsupported backend {backend!r}")

    def reference_metric_extractor(result: Any) -> dict[str, float]:
        # SpaceCore CGResult has a ``residual_norm`` attribute. NamedTuples
        # are also tuples, so attribute access is the disambiguating test.
        if hasattr(result, "residual_norm"):
            residual = result.residual_norm
            iters = getattr(result, "num_iters", float("nan"))
        else:
            # Bare path returns ``(x, residual_norm, iters)``.
            _, residual, iters = result
        try:
            residual_f = float(residual)
        except Exception:
            try:
                residual_f = float(np.asarray(residual))
            except Exception:
                residual_f = float("nan")
        try:
            iters_f = float(iters)
        except Exception:
            iters_f = float("nan")
        return {
            "final_residual_norm": residual_f,
            "iterations_completed": iters_f,
        }

    return MacroPayload(
        iterations=maxiter,
        size_params=dict(size_params),
        mode_callables=mode_callables,
        reference_metric_extractor=reference_metric_extractor,
        throughput_per_iteration=_MATVECS_PER_ITER,
    )


registry.register(
    MacroBenchmark(
        name="cg_poisson",
        workload="matrix-free CG on 2D (-Laplacian + lambda I) with Dirichlet BCs",
        sizes={
            "n=64": {"n": 64, "maxiter": 100, "lambda": 0.01},
            "n=128": {"n": 128, "maxiter": 100, "lambda": 0.01},
            "n=256": {"n": 256, "maxiter": 100, "lambda": 0.01},
        },
        backends=("numpy", "jax", "torch"),
        factory=_factory,
        quick_sizes=("n=64",),
        notes="1 matvec / iteration; SPD ensured by lambda > 0.",
    )
)
