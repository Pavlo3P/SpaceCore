"""PDHG for L1-regularized least squares.

Solves ``min_x 0.5 * ||A x - b||^2 + lambda * ||x||_1`` with the standard
Chambolle-Pock primal-dual hybrid gradient iteration:

    y_{k+1} = y_k + sigma * (A x_bar_k - b) / (1 + sigma)        (prox of f^*)
    x_{k+1} = soft(x_k - tau * A^T y_{k+1}, tau * lambda)        (prox of g)
    x_bar_{k+1} = x_{k+1} + theta * (x_{k+1} - x_k)              (over-relaxation)

Here ``f(z) = 0.5 * ||z - b||^2`` and ``g(x) = lambda * ||x||_1``. The
prox of the conjugate ``f^*`` simplifies to ``(y + sigma * (A x_bar - b)) /
(1 + sigma)``. ``theta = 1`` and step sizes are chosen with the
Frobenius-norm bound ``sigma * tau * ||A||_F^2 < 1``.

Modes
-----
* ``bare`` — pure backend matmul + soft-threshold loop.
* ``spacecore_public_none`` / ``spacecore_public_cheap`` — ``DenseLinOp``
  for ``A x`` and ``A^T y`` inside the loop; prox steps stay backend-native.
* ``spacecore_lowered`` — JAX-jitted whole-loop on ``jax``; equal to
  ``spacecore_public_none`` on ``numpy`` / ``torch``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import spacecore as sc

from .._operations import _backend_ctx, _np_dtype, _rng
from ._registry import MacroBenchmark, MacroPayload, registry


_BENCHMARK_NAME = "pdhg.l1_lsq"
_WORKLOAD = "PDHG L1-regularized least squares"


def _soft_threshold_np(z: np.ndarray, thresh: float) -> np.ndarray:
    sign = np.sign(z)
    return sign * np.maximum(np.abs(z) - thresh, 0.0)


def _factory(
    *,
    backend: str,
    device: str,
    seed: int,
    size_params: dict[str, Any],
) -> MacroPayload:
    """Build a :class:`MacroPayload` for one ``(backend, size, seed)`` cell."""
    m = int(size_params["m"])
    n = int(size_params["n"])
    iterations = int(size_params["iterations"])
    lambda_ = float(size_params["lambda_"])

    rng = _rng(seed)

    # NumPy ground truth (single source of truth for problem instance).
    a_np = np.asarray(rng.standard_normal((m, n)) / np.sqrt(m), dtype=np.float64)
    b_np = np.asarray(rng.standard_normal(m), dtype=np.float64)

    # Conservative step-size bound: sigma * tau * ||A||_F^2 < 1.
    # We pick sigma = tau = 0.9 / ||A||_F so that sigma * tau * ||A||_F^2
    # = 0.81 < 1, which is safe without a power iteration.
    a_fro = float(np.linalg.norm(a_np))
    step = 0.9 / max(a_fro, 1e-12)
    sigma = step
    tau = step
    theta = 1.0
    lam_tau = lambda_ * tau

    payload_meta = {
        "m": m,
        "n": n,
        "iterations": iterations,
        "lambda_": lambda_,
        "sigma": sigma,
        "tau": tau,
        "theta": theta,
        "a_frobenius": a_fro,
    }

    if backend == "numpy":
        return _build_numpy_payload(
            a_np=a_np,
            b_np=b_np,
            iterations=iterations,
            lambda_=lambda_,
            sigma=sigma,
            tau=tau,
            theta=theta,
            lam_tau=lam_tau,
            payload_meta=payload_meta,
        )
    if backend == "jax":
        return _build_jax_payload(
            a_np=a_np,
            b_np=b_np,
            iterations=iterations,
            lambda_=lambda_,
            sigma=sigma,
            tau=tau,
            theta=theta,
            lam_tau=lam_tau,
            payload_meta=payload_meta,
        )
    if backend == "torch":
        return _build_torch_payload(
            a_np=a_np,
            b_np=b_np,
            iterations=iterations,
            lambda_=lambda_,
            sigma=sigma,
            tau=tau,
            theta=theta,
            lam_tau=lam_tau,
            payload_meta=payload_meta,
        )
    raise ValueError(f"unsupported backend {backend!r}")


# ---------------------------------------------------------------------------
# NumPy


def _build_numpy_payload(
    *,
    a_np: np.ndarray,
    b_np: np.ndarray,
    iterations: int,
    lambda_: float,
    sigma: float,
    tau: float,
    theta: float,
    lam_tau: float,
    payload_meta: dict[str, Any],
) -> MacroPayload:
    m, n = a_np.shape
    ctx_none = _backend_ctx("numpy", check_level="none")
    ctx_cheap = _backend_ctx("numpy", check_level="cheap")
    dom_none = sc.DenseCoordinateSpace((n,), ctx_none)
    cod_none = sc.DenseCoordinateSpace((m,), ctx_none)
    dom_cheap = sc.DenseCoordinateSpace((n,), ctx_cheap)
    cod_cheap = sc.DenseCoordinateSpace((m,), ctx_cheap)

    a_arr_none = ctx_none.asarray(a_np)
    a_arr_cheap = ctx_cheap.asarray(a_np)
    ctx_none.asarray(b_np)
    op_none = sc.DenseLinOp(a_arr_none, dom_none, cod_none, ctx_none)
    op_cheap = sc.DenseLinOp(a_arr_cheap, dom_cheap, cod_cheap, ctx_cheap)

    a_local = a_np
    b_local = b_np
    one_plus_sigma = 1.0 + sigma

    def bare_run() -> dict[str, Any]:
        x = np.zeros(n, dtype=np.float64)
        x_bar = np.zeros(n, dtype=np.float64)
        y = np.zeros(m, dtype=np.float64)
        for _ in range(iterations):
            ax_bar = a_local @ x_bar
            y = (y + sigma * (ax_bar - b_local)) / one_plus_sigma
            at_y = a_local.T @ y
            x_new = _soft_threshold_np(x - tau * at_y, lam_tau)
            x_bar = x_new + theta * (x_new - x)
            x = x_new
        return {"x": x, "y": y}

    def _sc_run(op: Any) -> dict[str, Any]:
        x = np.zeros(n, dtype=np.float64)
        x_bar = np.zeros(n, dtype=np.float64)
        y = np.zeros(m, dtype=np.float64)
        for _ in range(iterations):
            ax_bar = op.apply(x_bar)
            y = (y + sigma * (ax_bar - b_local)) / one_plus_sigma
            at_y = op.rapply(y)
            x_new = _soft_threshold_np(x - tau * at_y, lam_tau)
            x_bar = x_new + theta * (x_new - x)
            x = x_new
        return {"x": x, "y": y}

    def public_none_run() -> dict[str, Any]:
        return _sc_run(op_none)

    def public_cheap_run() -> dict[str, Any]:
        return _sc_run(op_cheap)

    def extractor(state: dict[str, Any]) -> dict[str, float]:
        x_arr = np.asarray(state["x"], dtype=np.float64)
        y_arr = np.asarray(state["y"], dtype=np.float64)
        residual_vec = a_local @ x_arr - b_local
        objective = float(0.5 * float(np.dot(residual_vec, residual_vec))
                          + lambda_ * float(np.sum(np.abs(x_arr))))
        primal_residual = float(np.linalg.norm(residual_vec))
        dual_residual = float(np.linalg.norm(a_local.T @ y_arr))
        return {
            "objective": objective,
            "primal_residual": primal_residual,
            "dual_residual": dual_residual,
        }

    return MacroPayload(
        iterations=iterations,
        size_params=payload_meta,
        mode_callables={
            "bare": bare_run,
            "spacecore_public_none": public_none_run,
            "spacecore_public_cheap": public_cheap_run,
            "spacecore_lowered": public_none_run,
        },
        reference_metric_extractor=extractor,
        throughput_per_iteration=1.0,
    )


# ---------------------------------------------------------------------------
# JAX


def _build_jax_payload(
    *,
    a_np: np.ndarray,
    b_np: np.ndarray,
    iterations: int,
    lambda_: float,
    sigma: float,
    tau: float,
    theta: float,
    lam_tau: float,
    payload_meta: dict[str, Any],
) -> MacroPayload:
    import jax
    import jax.numpy as jnp

    m, n = a_np.shape
    ctx_none = _backend_ctx("jax", check_level="none")
    ctx_cheap = _backend_ctx("jax", check_level="cheap")
    dom_none = sc.DenseCoordinateSpace((n,), ctx_none)
    cod_none = sc.DenseCoordinateSpace((m,), ctx_none)
    dom_cheap = sc.DenseCoordinateSpace((n,), ctx_cheap)
    cod_cheap = sc.DenseCoordinateSpace((m,), ctx_cheap)

    np_dtype = _np_dtype(ctx_none)
    a_typed = np.asarray(a_np, dtype=np_dtype)
    b_typed = np.asarray(b_np, dtype=np_dtype)

    a_jax_none = ctx_none.asarray(a_typed)
    a_jax_cheap = ctx_cheap.asarray(a_typed)
    b_jax = ctx_none.asarray(b_typed)
    op_none = sc.DenseLinOp(a_jax_none, dom_none, cod_none, ctx_none)
    op_cheap = sc.DenseLinOp(a_jax_cheap, dom_cheap, cod_cheap, ctx_cheap)

    a_local = a_jax_none
    b_local = b_jax
    one_plus_sigma = 1.0 + sigma
    zeros_x = jnp.zeros((n,), dtype=a_local.dtype)
    zeros_y = jnp.zeros((m,), dtype=a_local.dtype)

    def _soft_threshold_jax(z: Any, thresh: float) -> Any:
        return jnp.sign(z) * jnp.maximum(jnp.abs(z) - thresh, 0.0)

    def bare_run() -> dict[str, Any]:
        x = zeros_x
        x_bar = zeros_x
        y = zeros_y
        for _ in range(iterations):
            ax_bar = a_local @ x_bar
            y = (y + sigma * (ax_bar - b_local)) / one_plus_sigma
            at_y = a_local.T @ y
            x_new = _soft_threshold_jax(x - tau * at_y, lam_tau)
            x_bar = x_new + theta * (x_new - x)
            x = x_new
        return {"x": x, "y": y}

    def _sc_run(op: Any) -> dict[str, Any]:
        x = zeros_x
        x_bar = zeros_x
        y = zeros_y
        for _ in range(iterations):
            ax_bar = op.apply(x_bar)
            y = (y + sigma * (ax_bar - b_local)) / one_plus_sigma
            at_y = op.rapply(y)
            x_new = _soft_threshold_jax(x - tau * at_y, lam_tau)
            x_bar = x_new + theta * (x_new - x)
            x = x_new
        return {"x": x, "y": y}

    def public_none_run() -> dict[str, Any]:
        return _sc_run(op_none)

    def public_cheap_run() -> dict[str, Any]:
        return _sc_run(op_cheap)

    def _one_step(carry, _):
        x, x_bar, y = carry
        ax_bar = a_local @ x_bar
        y = (y + sigma * (ax_bar - b_local)) / one_plus_sigma
        at_y = a_local.T @ y
        x_new = _soft_threshold_jax(x - tau * at_y, lam_tau)
        x_bar = x_new + theta * (x_new - x)
        return (x_new, x_bar, y), None

    @jax.jit
    def _lowered_loop(x0, xb0, y0):
        (x, x_bar, y), _ = jax.lax.scan(
            _one_step, (x0, xb0, y0), None, length=iterations
        )
        return x, y

    def lowered_run() -> dict[str, Any]:
        x, y = _lowered_loop(zeros_x, zeros_x, zeros_y)
        return {"x": x, "y": y}

    a_np_ref = np.asarray(a_typed, dtype=np.float64)
    b_np_ref = np.asarray(b_typed, dtype=np.float64)

    def extractor(state: dict[str, Any]) -> dict[str, float]:
        x_arr = np.asarray(state["x"], dtype=np.float64)
        y_arr = np.asarray(state["y"], dtype=np.float64)
        residual_vec = a_np_ref @ x_arr - b_np_ref
        objective = float(0.5 * float(np.dot(residual_vec, residual_vec))
                          + lambda_ * float(np.sum(np.abs(x_arr))))
        primal_residual = float(np.linalg.norm(residual_vec))
        dual_residual = float(np.linalg.norm(a_np_ref.T @ y_arr))
        return {
            "objective": objective,
            "primal_residual": primal_residual,
            "dual_residual": dual_residual,
        }

    return MacroPayload(
        iterations=iterations,
        size_params=payload_meta,
        mode_callables={
            "bare": bare_run,
            "spacecore_public_none": public_none_run,
            "spacecore_public_cheap": public_cheap_run,
            "spacecore_lowered": lowered_run,
        },
        reference_metric_extractor=extractor,
        throughput_per_iteration=1.0,
    )


# ---------------------------------------------------------------------------
# Torch


def _build_torch_payload(
    *,
    a_np: np.ndarray,
    b_np: np.ndarray,
    iterations: int,
    lambda_: float,
    sigma: float,
    tau: float,
    theta: float,
    lam_tau: float,
    payload_meta: dict[str, Any],
) -> MacroPayload:
    import torch

    m, n = a_np.shape
    ctx_none = _backend_ctx("torch", check_level="none")
    ctx_cheap = _backend_ctx("torch", check_level="cheap")
    dom_none = sc.DenseCoordinateSpace((n,), ctx_none)
    cod_none = sc.DenseCoordinateSpace((m,), ctx_none)
    dom_cheap = sc.DenseCoordinateSpace((n,), ctx_cheap)
    cod_cheap = sc.DenseCoordinateSpace((m,), ctx_cheap)

    np_dtype = _np_dtype(ctx_none)
    a_typed = np.asarray(a_np, dtype=np_dtype)
    b_typed = np.asarray(b_np, dtype=np_dtype)

    a_torch_none = ctx_none.asarray(a_typed)
    a_torch_cheap = ctx_cheap.asarray(a_typed)
    b_torch = ctx_none.asarray(b_typed)
    op_none = sc.DenseLinOp(a_torch_none, dom_none, cod_none, ctx_none)
    op_cheap = sc.DenseLinOp(a_torch_cheap, dom_cheap, cod_cheap, ctx_cheap)

    a_local = a_torch_none
    b_local = b_torch
    one_plus_sigma = 1.0 + sigma
    torch_dtype = a_local.dtype
    torch_device = a_local.device

    def _soft_threshold_torch(z: Any, thresh: float) -> Any:
        return torch.sign(z) * torch.clamp(torch.abs(z) - thresh, min=0.0)

    def _zeros_x() -> Any:
        return torch.zeros(n, dtype=torch_dtype, device=torch_device)

    def _zeros_y() -> Any:
        return torch.zeros(m, dtype=torch_dtype, device=torch_device)

    def bare_run() -> dict[str, Any]:
        x = _zeros_x()
        x_bar = _zeros_x()
        y = _zeros_y()
        a_t = a_local
        a_T = a_local.T
        for _ in range(iterations):
            ax_bar = a_t @ x_bar
            y = (y + sigma * (ax_bar - b_local)) / one_plus_sigma
            at_y = a_T @ y
            x_new = _soft_threshold_torch(x - tau * at_y, lam_tau)
            x_bar = x_new + theta * (x_new - x)
            x = x_new
        return {"x": x, "y": y}

    def _sc_run(op: Any) -> dict[str, Any]:
        x = _zeros_x()
        x_bar = _zeros_x()
        y = _zeros_y()
        for _ in range(iterations):
            ax_bar = op.apply(x_bar)
            y = (y + sigma * (ax_bar - b_local)) / one_plus_sigma
            at_y = op.rapply(y)
            x_new = _soft_threshold_torch(x - tau * at_y, lam_tau)
            x_bar = x_new + theta * (x_new - x)
            x = x_new
        return {"x": x, "y": y}

    def public_none_run() -> dict[str, Any]:
        return _sc_run(op_none)

    def public_cheap_run() -> dict[str, Any]:
        return _sc_run(op_cheap)

    a_np_ref = np.asarray(a_typed, dtype=np.float64)
    b_np_ref = np.asarray(b_typed, dtype=np.float64)

    def extractor(state: dict[str, Any]) -> dict[str, float]:
        x_arr = np.asarray(state["x"].detach().cpu().numpy(), dtype=np.float64)
        y_arr = np.asarray(state["y"].detach().cpu().numpy(), dtype=np.float64)
        residual_vec = a_np_ref @ x_arr - b_np_ref
        objective = float(0.5 * float(np.dot(residual_vec, residual_vec))
                          + lambda_ * float(np.sum(np.abs(x_arr))))
        primal_residual = float(np.linalg.norm(residual_vec))
        dual_residual = float(np.linalg.norm(a_np_ref.T @ y_arr))
        return {
            "objective": objective,
            "primal_residual": primal_residual,
            "dual_residual": dual_residual,
        }

    return MacroPayload(
        iterations=iterations,
        size_params=payload_meta,
        mode_callables={
            "bare": bare_run,
            "spacecore_public_none": public_none_run,
            "spacecore_public_cheap": public_cheap_run,
            "spacecore_lowered": public_none_run,
        },
        reference_metric_extractor=extractor,
        throughput_per_iteration=1.0,
    )


# ---------------------------------------------------------------------------
# Registration


_SIZES: dict[str, dict[str, Any]] = {
    "(5000,2000)": {
        "m": 5_000,
        "n": 2_000,
        "iterations": 500,
        "lambda_": 0.01,
    },
    "(50000,20000)": {
        "m": 50_000,
        "n": 20_000,
        "iterations": 500,
        "lambda_": 0.01,
    },
}


registry.register(
    MacroBenchmark(
        name=_BENCHMARK_NAME,
        workload=_WORKLOAD,
        sizes=_SIZES,
        backends=("numpy", "jax", "torch"),
        factory=_factory,
        quick_sizes=("(5000,2000)",),
        notes="Standard PDHG for min 0.5||Ax-b||^2 + lam||x||_1 with theta=1.",
    )
)
