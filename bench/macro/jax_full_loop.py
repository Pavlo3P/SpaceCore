"""JAX full-loop macrobenchmarks: ``jax.lax.scan``-driven CG and PDHG.

This module exists to validate one specific claim about SpaceCore on JAX:
a workload that is most naturally expressed against the SpaceCore public
surface (``LinOp.apply``, ``space.add``, etc.) can be *lowered* to a
single ``jax.jit`` compilation whose steady-state runtime matches a
hand-written ``jax.lax.scan`` body that uses raw JAX primitives.

Two benchmarks are registered:

* ``jax_full_loop.cg_poisson`` — conjugate gradients on a 1D Poisson
  (tridiagonal ``-1, 2, -1``) operator. The bare path runs the scan body
  on a JAX array; the SpaceCore-lowered path runs the same scan body
  but the matvec goes through ``DenseLinOp.apply``.
* ``jax_full_loop.pdhg`` — primal-dual hybrid gradient on a dense
  least-squares problem. Same shape: bare uses raw JAX matvecs;
  ``spacecore_lowered`` wraps the operator in ``DenseLinOp`` and runs
  the scan body through SpaceCore.

JAX-only. The module imports JAX lazily so it can be loaded on Python
installations that do not have JAX, in which case nothing is registered.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import spacecore as sc

from .._operations import _backend_ctx, _rng
from ._registry import MacroBenchmark, MacroPayload, registry


# ---------------------------------------------------------------------------
# JAX availability — module-scope guard. If JAX is missing we register no
# benchmarks and the runner walks past us.


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared helpers


def _np_real_dtype() -> np.dtype:
    """Return the host-side NumPy dtype matching JAX's real default."""
    from tests._helpers import jax_real_dtype

    return np.dtype(jax_real_dtype())


def _poisson_matrix(n: int, dtype: np.dtype) -> np.ndarray:
    """Return the dense 1D Poisson tridiagonal matrix ``tridiag(-1, 2, -1)``."""
    a = np.zeros((n, n), dtype=dtype)
    idx = np.arange(n)
    a[idx, idx] = 2.0
    a[idx[:-1], idx[:-1] + 1] = -1.0
    a[idx[1:], idx[1:] - 1] = -1.0
    return a


# ---------------------------------------------------------------------------
# Benchmark 1 — CG with jax.lax.scan body
#
# The Poisson operator is symmetric positive-definite, so a fixed-iteration
# CG converges monotonically. Every mode runs the same number of
# iterations (``maxiter``) so the residuals are directly comparable as
# numerical references via ``error_vs_bare``.


def _cg_factory(
    *,
    backend: str,
    device: str,
    seed: int,
    size_params: dict[str, Any],
) -> MacroPayload:
    if backend != "jax":
        raise ValueError(f"jax_full_loop.cg_poisson only supports backend='jax', got {backend!r}")

    import jax
    import jax.numpy as jnp

    n = int(size_params["n"])
    maxiter = int(size_params["maxiter"])
    np_dtype = _np_real_dtype()

    rng = _rng(seed)
    a_np = _poisson_matrix(n, np_dtype)
    b_np = np.asarray(rng.standard_normal(n), dtype=np_dtype)

    a_jax = jnp.asarray(a_np)
    b_jax = jnp.asarray(b_np)

    # --- bare path: jax.lax.scan over raw JAX matvecs.
    def _bare_cg_loop(a, b):
        x0 = jnp.zeros_like(b)
        r0 = b - a @ x0
        p0 = r0
        rs0 = jnp.vdot(r0, r0).real

        def body(carry, _):
            x, r, p, rs = carry
            ap = a @ p
            pap = jnp.vdot(p, ap).real
            alpha = rs / pap
            x_next = x + alpha * p
            r_next = r - alpha * ap
            rs_next = jnp.vdot(r_next, r_next).real
            beta = rs_next / rs
            p_next = r_next + beta * p
            return (x_next, r_next, p_next, rs_next), None

        (x, r, _p, rs), _ = jax.lax.scan(body, (x0, r0, p0, rs0), None, length=maxiter)
        residual = jnp.sqrt(rs)
        return x, residual

    _bare_cg_jit = jax.jit(_bare_cg_loop)

    def bare_callable():
        x, residual = _bare_cg_jit(a_jax, b_jax)
        return {"x": x, "residual": residual}

    # --- SpaceCore eager paths (public_none / public_cheap). Same callable
    # shape, separate Context per check_level. We build the full LinOp once
    # in setup; the timed callable just rebuilds the carries and steps.
    ctx_none = _backend_ctx("jax", check_level="none")
    ctx_cheap = _backend_ctx("jax", check_level="cheap")

    space_none = sc.DenseCoordinateSpace((n,), ctx_none)
    space_cheap = sc.DenseCoordinateSpace((n,), ctx_cheap)

    a_arr_none = ctx_none.asarray(a_np)
    a_arr_cheap = ctx_cheap.asarray(a_np)
    b_arr_none = ctx_none.asarray(b_np)
    b_arr_cheap = ctx_cheap.asarray(b_np)

    op_none = sc.DenseLinOp(a_arr_none, space_none, space_none, ctx_none)
    op_cheap = sc.DenseLinOp(a_arr_cheap, space_cheap, space_cheap, ctx_cheap)

    def _public_cg_step(op, space, b_arr):
        """Eager CG: one Python-level loop with SpaceCore public ops."""
        x = space.zeros()
        r = space.add(b_arr, space.scale(-1.0, op.apply(x)))
        p = r
        rs = space.inner(r, r).real
        for _ in range(maxiter):
            ap = op.apply(p)
            pap = space.inner(p, ap).real
            alpha = rs / pap
            x = space.axpy(alpha, p, x)
            r = space.axpy(-alpha, ap, r)
            rs_next = space.inner(r, r).real
            beta = rs_next / rs
            p = space.axpy(beta, p, r)
            rs = rs_next
        residual = jnp.sqrt(rs)
        return {"x": x, "residual": residual}

    def spacecore_public_none_callable():
        return _public_cg_step(op_none, space_none, b_arr_none)

    def spacecore_public_cheap_callable():
        return _public_cg_step(op_cheap, space_cheap, b_arr_cheap)

    # --- SpaceCore lowered path: jax.jit a function that uses the SC LinOp's
    # apply inside a jax.lax.scan. The trace lowers ``op.apply`` to the same
    # underlying ``a @ x`` JAX op without going through any Python-level
    # SpaceCore validation per iteration.
    a_arr_lowered = ctx_none.asarray(a_np)
    b_arr_lowered = ctx_none.asarray(b_np)
    op_lowered = sc.DenseLinOp(a_arr_lowered, space_none, space_none, ctx_none)

    def _lowered_cg_loop(b):
        x0 = space_none.zeros()
        r0 = space_none.add(b, space_none.scale(-1.0, op_lowered.apply(x0)))
        p0 = r0
        rs0 = space_none.inner(r0, r0).real

        def body(carry, _):
            x, r, p, rs = carry
            ap = op_lowered.apply(p)
            pap = space_none.inner(p, ap).real
            alpha = rs / pap
            x_next = space_none.axpy(alpha, p, x)
            r_next = space_none.axpy(-alpha, ap, r)
            rs_next = space_none.inner(r_next, r_next).real
            beta = rs_next / rs
            p_next = space_none.axpy(beta, p, r_next)
            return (x_next, r_next, p_next, rs_next), None

        (x, _r, _p, rs), _ = jax.lax.scan(body, (x0, r0, p0, rs0), None, length=maxiter)
        residual = jnp.sqrt(rs)
        return x, residual

    _lowered_cg_jit = jax.jit(_lowered_cg_loop)

    def spacecore_lowered_callable():
        x, residual = _lowered_cg_jit(b_arr_lowered)
        return {"x": x, "residual": residual}

    def reference_metric_extractor(result: Any) -> dict[str, float]:
        residual = result["residual"]
        return {
            "residual": float(np.asarray(residual)),
            "iterations": float(maxiter),
        }

    return MacroPayload(
        iterations=maxiter,
        size_params=dict(size_params),
        mode_callables={
            "bare": bare_callable,
            "spacecore_public_none": spacecore_public_none_callable,
            "spacecore_public_cheap": spacecore_public_cheap_callable,
            "spacecore_lowered": spacecore_lowered_callable,
        },
        reference_metric_extractor=reference_metric_extractor,
        throughput_per_iteration=1.0,  # one matvec per CG step
    )


# ---------------------------------------------------------------------------
# Benchmark 2 — PDHG with jax.lax.scan body
#
# Problem: ``min_x 0.5 * ||A x - b||^2`` for a tall dense ``A``. PDHG with
# the indicator-free formulation reduces to:
#
#   y_{k+1} = (y_k + sigma * (A x_bar - b)) / (1 + sigma)
#   x_{k+1} = x_k - tau * A^T y_{k+1}
#   x_bar   = 2 * x_{k+1} - x_k
#
# The step sizes ``sigma * tau * ||A||^2 < 1`` are conservative; we use
# ``sigma = tau = 0.9 / ||A||_2`` based on a power-iteration estimate so the
# scheme converges. Objective at the final iterate is the metric we extract.


def _pdhg_factory(
    *,
    backend: str,
    device: str,
    seed: int,
    size_params: dict[str, Any],
) -> MacroPayload:
    if backend != "jax":
        raise ValueError(f"jax_full_loop.pdhg only supports backend='jax', got {backend!r}")

    import jax
    import jax.numpy as jnp

    m = int(size_params["m"])
    n = int(size_params["n"])
    iterations = int(size_params["iterations"])
    np_dtype = _np_real_dtype()

    rng = _rng(seed)
    # A is m x n, b is m. Scale A so step sizes are well-conditioned.
    a_np = np.asarray(rng.standard_normal((m, n)), dtype=np_dtype) / np.sqrt(m)
    b_np = np.asarray(rng.standard_normal(m), dtype=np_dtype)

    # Conservative step sizes from the spectral norm of A. We estimate via
    # a small SVD on the host so all modes share identical hyperparameters.
    sigma_max = float(np.linalg.norm(a_np, ord=2))
    step = 0.9 / max(sigma_max, 1e-12)
    sigma = step
    tau = step

    a_jax = jnp.asarray(a_np)
    b_jax = jnp.asarray(b_np)

    # --- bare path: jax.lax.scan over raw matvecs.
    def _bare_pdhg_loop(a, b):
        x0 = jnp.zeros(n, dtype=a.dtype)
        y0 = jnp.zeros(m, dtype=a.dtype)
        xbar0 = x0

        def body(carry, _):
            x, y, xbar = carry
            y_next = (y + sigma * (a @ xbar - b)) / (1.0 + sigma)
            x_next = x - tau * (a.T @ y_next)
            xbar_next = 2.0 * x_next - x
            return (x_next, y_next, xbar_next), None

        (x, _y, _xbar), _ = jax.lax.scan(body, (x0, y0, xbar0), None, length=iterations)
        residual_vec = a @ x - b
        objective = 0.5 * jnp.vdot(residual_vec, residual_vec).real
        return x, objective

    _bare_pdhg_jit = jax.jit(_bare_pdhg_loop)

    def bare_callable():
        x, objective = _bare_pdhg_jit(a_jax, b_jax)
        return {"x": x, "objective": objective}

    # --- SpaceCore eager public paths.
    ctx_none = _backend_ctx("jax", check_level="none")
    ctx_cheap = _backend_ctx("jax", check_level="cheap")

    domain_none = sc.DenseCoordinateSpace((n,), ctx_none)
    codomain_none = sc.DenseCoordinateSpace((m,), ctx_none)
    domain_cheap = sc.DenseCoordinateSpace((n,), ctx_cheap)
    codomain_cheap = sc.DenseCoordinateSpace((m,), ctx_cheap)

    a_arr_none = ctx_none.asarray(a_np)
    a_arr_cheap = ctx_cheap.asarray(a_np)
    b_arr_none = ctx_none.asarray(b_np)
    b_arr_cheap = ctx_cheap.asarray(b_np)

    op_none = sc.DenseLinOp(a_arr_none, domain_none, codomain_none, ctx_none)
    op_cheap = sc.DenseLinOp(a_arr_cheap, domain_cheap, codomain_cheap, ctx_cheap)

    def _public_pdhg_step(op, domain, codomain, b_arr):
        x = domain.zeros()
        y = codomain.zeros()
        xbar = x
        for _ in range(iterations):
            ax_bar = op.apply(xbar)
            ax_bar_minus_b = codomain.add(ax_bar, codomain.scale(-1.0, b_arr))
            y_next = codomain.scale(
                1.0 / (1.0 + sigma),
                codomain.axpy(sigma, ax_bar_minus_b, y),
            )
            x_next = domain.axpy(-tau, op.rapply(y_next), x)
            xbar = domain.axpy(-1.0, x, domain.scale(2.0, x_next))
            x = x_next
            y = y_next
        residual_vec = codomain.add(op.apply(x), codomain.scale(-1.0, b_arr))
        objective = 0.5 * codomain.inner(residual_vec, residual_vec).real
        return {"x": x, "objective": objective}

    def spacecore_public_none_callable():
        return _public_pdhg_step(op_none, domain_none, codomain_none, b_arr_none)

    def spacecore_public_cheap_callable():
        return _public_pdhg_step(op_cheap, domain_cheap, codomain_cheap, b_arr_cheap)

    # --- SpaceCore lowered path: jit a function that uses op.apply/op.rapply
    # inside a jax.lax.scan body.
    a_arr_lowered = ctx_none.asarray(a_np)
    b_arr_lowered = ctx_none.asarray(b_np)
    op_lowered = sc.DenseLinOp(a_arr_lowered, domain_none, codomain_none, ctx_none)

    def _lowered_pdhg_loop(b):
        x0 = domain_none.zeros()
        y0 = codomain_none.zeros()
        xbar0 = x0

        def body(carry, _):
            x, y, xbar = carry
            ax_bar = op_lowered.apply(xbar)
            ax_bar_minus_b = codomain_none.add(ax_bar, codomain_none.scale(-1.0, b))
            y_next = codomain_none.scale(
                1.0 / (1.0 + sigma),
                codomain_none.axpy(sigma, ax_bar_minus_b, y),
            )
            x_next = domain_none.axpy(-tau, op_lowered.rapply(y_next), x)
            xbar_next = domain_none.axpy(-1.0, x, domain_none.scale(2.0, x_next))
            return (x_next, y_next, xbar_next), None

        (x, _y, _xbar), _ = jax.lax.scan(body, (x0, y0, xbar0), None, length=iterations)
        residual_vec = codomain_none.add(op_lowered.apply(x), codomain_none.scale(-1.0, b))
        objective = 0.5 * codomain_none.inner(residual_vec, residual_vec).real
        return x, objective

    _lowered_pdhg_jit = jax.jit(_lowered_pdhg_loop)

    def spacecore_lowered_callable():
        x, objective = _lowered_pdhg_jit(b_arr_lowered)
        return {"x": x, "objective": objective}

    def reference_metric_extractor(result: Any) -> dict[str, float]:
        return {"objective": float(np.asarray(result["objective"]))}

    return MacroPayload(
        iterations=iterations,
        size_params=dict(size_params),
        mode_callables={
            "bare": bare_callable,
            "spacecore_public_none": spacecore_public_none_callable,
            "spacecore_public_cheap": spacecore_public_cheap_callable,
            "spacecore_lowered": spacecore_lowered_callable,
        },
        reference_metric_extractor=reference_metric_extractor,
        throughput_per_iteration=2.0,  # one matvec + one rmatvec per PDHG step
    )


# ---------------------------------------------------------------------------
# Registration. Skip cleanly when JAX is not importable.

if _jax_available():
    registry.register(
        MacroBenchmark(
            name="jax_full_loop.cg_poisson",
            workload="CG on the 1D Poisson operator inside jax.lax.scan",
            sizes={
                "n=32": {"n": 32, "maxiter": 100},
                "n=64": {"n": 64, "maxiter": 100},
            },
            backends=("jax",),
            factory=_cg_factory,
            quick_sizes=("n=32",),
            notes=(
                "Validates that a SpaceCore-described CG lowers to a single "
                "jax.jit-compiled jax.lax.scan whose steady-state runtime "
                "matches a hand-written bare scan."
            ),
        )
    )
    registry.register(
        MacroBenchmark(
            name="jax_full_loop.pdhg",
            workload="PDHG least-squares inside jax.lax.scan",
            sizes={
                "(2000,1000)": {"m": 2000, "n": 1000, "iterations": 200},
            },
            backends=("jax",),
            factory=_pdhg_factory,
            quick_sizes=("(2000,1000)",),
            notes=(
                "Validates that a SpaceCore-described primal-dual iteration "
                "lowers to a jax.jit-compiled scan and matches a bare scan "
                "in steady state."
            ),
        )
    )
