"""Operator algebra stress macrobenchmark.

Builds a nested SpaceCore operator expression

    A = B3 @ (alpha1 * A1 + alpha2 * A2) @ B2 @ B1

over random dense ``(d, d)`` matrices and times three workloads
(``apply``, ``rapply``, ``normal_apply = A.rapply(A.apply(x))``) in the
four canonical run modes.

The benchmark stresses SpaceCore's operator algebra (``@``, ``+``, ``*``)
overhead vs

* ``bare``: a NumPy/JAX/Torch dense matrix ``A_dense @ x`` precomputed
  outside the timed loop.
* ``spacecore_public_none`` / ``spacecore_public_cheap``: rebuild the
  expression via :class:`spacecore.DenseLinOp` and the operator algebra
  on every payload, then reuse the same expression inside the timed
  callable (with the respective check levels).
* ``spacecore_lowered``: precompute the final dense matrix via
  :meth:`spacecore.LinOp.to_dense`, then perform a plain ``A_dense @ x``.

The reference metric is ``apply_norm`` (L2 norm of the apply output), so
the runner can compute ``error_vs_bare`` across modes.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import spacecore as sc
from bench._operations import _backend_ctx, _np_dtype, _rng
from bench.macro._registry import MacroBenchmark, MacroPayload, registry
from bench.macro._schema import ModeName


_BENCHMARK_NAME = "operator_stress"
_ITERATIONS = 50
_SIZES: dict[str, dict[str, Any]] = {
    "d=256": {"d": 256, "chain_depth": 4},
    "d=1024": {"d": 1024, "chain_depth": 4},
    "d=4096_depth8": {"d": 4096, "chain_depth": 8},
}
_QUICK_SIZES: tuple[str, ...] = ("d=256",)
_BACKENDS: tuple[str, ...] = ("numpy", "jax", "torch")


def _to_backend(backend: str, x_np: np.ndarray, ctx: sc.Context) -> Any:
    """Convert a NumPy array to the requested backend in ``ctx.dtype``.

    Uses ``ctx.asarray`` so the resulting tensor lives on the same
    backend and dtype the SpaceCore expression uses. That keeps the
    ``bare`` path numerically identical to the SC paths and avoids
    backend-specific dtype mismatches (e.g. torch defaults to float32
    while NumPy is float64).
    """
    return ctx.asarray(np.asarray(x_np))


def _bare_matmul(backend: str):
    """Return the backend-native matrix-vector multiply (matrix @ vector)."""
    if backend == "numpy":
        return lambda M, v: M @ v
    if backend == "jax":
        import jax.numpy as jnp

        return lambda M, v: jnp.matmul(M, v)
    if backend == "torch":
        import torch

        return lambda M, v: torch.matmul(M, v)
    raise ValueError(f"unknown backend {backend!r}")


def _bare_conj_transpose(backend: str, M: Any) -> Any:
    """Conjugate-transpose ``M`` using backend-native ops."""
    if backend == "numpy":
        return np.conj(M).T
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.conj(M).T
    if backend == "torch":
        import torch

        return torch.conj(M).T
    raise ValueError(f"unknown backend {backend!r}")


def _bare_norm(backend: str, x: Any) -> float:
    """L2 norm as a Python float."""
    if backend == "numpy":
        return float(np.linalg.norm(x))
    if backend == "jax":
        import jax.numpy as jnp

        return float(jnp.linalg.norm(x))
    if backend == "torch":
        import torch

        return float(torch.linalg.norm(x))
    raise ValueError(f"unknown backend {backend!r}")


def _build_sc_expression(
    ctx: sc.Context,
    space: sc.DenseCoordinateSpace,
    A_matrices: list[Any],
    B_matrices: list[Any],
    alphas: tuple[float, float],
) -> sc.LinOp:
    """Assemble ``B_last @ ... @ B2 @ (alpha1*A1 + alpha2*A2) @ B2 @ B1``.

    ``B_matrices`` is the full chain of B operators (``chain_depth`` of
    them, including B1, B2, ..., B_last). The mixture sits between the
    first half and the second half of the chain.
    """
    A1, A2 = A_matrices[0], A_matrices[1]
    op_A1 = sc.DenseLinOp(A1, space, space, ctx)
    op_A2 = sc.DenseLinOp(A2, space, space, ctx)
    mix = alphas[0] * op_A1 + alphas[1] * op_A2

    b_ops = [sc.DenseLinOp(B, space, space, ctx) for B in B_matrices]
    half = len(b_ops) // 2
    if half == 0:
        # Degenerate: at least one B on each side. Fall back to mix only.
        return mix
    left = b_ops[half:]
    right = b_ops[:half]

    expr = mix
    for op in right[::-1]:
        # Right-side: applied to x before the mixture.
        # Compose so that expr.apply(x) = mix.apply(op.apply(x))
        expr = expr @ op
    for op in left:
        # Left-side: applied after the mixture.
        expr = op @ expr
    return expr


def _build_dense_matrix(
    backend: str,
    ctx: sc.Context,
    A_np: list[np.ndarray],
    B_np: list[np.ndarray],
    alphas: tuple[float, float],
) -> Any:
    """Precompute the equivalent dense matrix in NumPy and ship to backend."""
    A1_np, A2_np = A_np[0], A_np[1]
    mix_np = alphas[0] * A1_np + alphas[1] * A2_np
    half = len(B_np) // 2
    right = B_np[:half]
    left = B_np[half:]

    M_np = mix_np
    # Apply right-side Bs to the right of the mixture.
    for B in right[::-1]:
        M_np = M_np @ B
    # Apply left-side Bs to the left of the mixture.
    for B in left:
        M_np = B @ M_np
    return _to_backend(backend, M_np, ctx)


def _factory(
    backend: str,
    device: str,
    seed: int,
    size_params: dict[str, Any],
) -> MacroPayload:
    """Build the operator-stress payload for one ``(backend, size, seed)``."""
    d = int(size_params["d"])
    chain_depth = int(size_params["chain_depth"])
    rng = _rng(seed)

    # SpaceCore contexts and space — built once per payload.
    ctx_none = _backend_ctx(backend, check_level="none")
    ctx_cheap = _backend_ctx(backend, check_level="cheap")
    np_dtype = _np_dtype(ctx_none)

    space_none = sc.DenseCoordinateSpace((d,), ctx_none)
    space_cheap = sc.DenseCoordinateSpace((d,), ctx_cheap)

    # Pre-generate every matrix and vector in NumPy.
    A_np = [
        np.asarray(rng.standard_normal((d, d)), dtype=np_dtype) / np.sqrt(d)
        for _ in range(2)
    ]
    B_np = [
        np.asarray(rng.standard_normal((d, d)), dtype=np_dtype) / np.sqrt(d)
        for _ in range(chain_depth)
    ]
    alphas = (
        float(rng.standard_normal()),
        float(rng.standard_normal()),
    )
    x_np = np.asarray(rng.standard_normal(d), dtype=np_dtype)
    y_np = np.asarray(rng.standard_normal(d), dtype=np_dtype)

    # Convert per-backend operand arrays once.
    A_none = [ctx_none.asarray(M) for M in A_np]
    B_none = [ctx_none.asarray(M) for M in B_np]
    A_cheap = [ctx_cheap.asarray(M) for M in A_np]
    B_cheap = [ctx_cheap.asarray(M) for M in B_np]
    x_none = ctx_none.asarray(x_np)
    y_none = ctx_none.asarray(y_np)
    x_cheap = ctx_cheap.asarray(x_np)
    y_cheap = ctx_cheap.asarray(y_np)

    # Bare-mode operands — backend-native, no SpaceCore objects.
    x_bare = _to_backend(backend, x_np, ctx_none)
    y_bare = _to_backend(backend, y_np, ctx_none)
    A_dense_bare = _build_dense_matrix(backend, ctx_none, A_np, B_np, alphas)
    A_dense_bare_H = _bare_conj_transpose(backend, A_dense_bare)
    matmul = _bare_matmul(backend)

    # SpaceCore expression for the public-API paths.
    expr_none = _build_sc_expression(ctx_none, space_none, A_none, B_none, alphas)
    expr_cheap = _build_sc_expression(ctx_cheap, space_cheap, A_cheap, B_cheap, alphas)

    # Lowered path: collapse the expression to a single dense matrix once.
    A_dense_lowered = expr_none.to_dense()
    A_dense_lowered_H = _bare_conj_transpose(backend, A_dense_lowered)

    iterations = _ITERATIONS

    # Each "iteration" performs one apply, one rapply, one normal_apply
    # on the SAME fixed input vectors. This keeps numerics bounded
    # (no compounding of operator powers) while still measuring per-call
    # SpaceCore overhead. The reference metric is taken from the final
    # apply_out, which is identical across iterations.

    # ----- bare -----
    def bare_call() -> dict[str, Any]:
        apply_out = x_bare
        rapply_out = y_bare
        normal_out = x_bare
        for _ in range(iterations):
            apply_out = matmul(A_dense_bare, x_bare)
            rapply_out = matmul(A_dense_bare_H, y_bare)
            normal_out = matmul(A_dense_bare_H, matmul(A_dense_bare, x_bare))
        return {
            "apply": apply_out,
            "rapply": rapply_out,
            "normal_apply": normal_out,
            "backend": backend,
        }

    # ----- spacecore public (check_level varies via context) -----
    def _sc_public_call(expr: sc.LinOp, x_in: Any, y_in: Any) -> dict[str, Any]:
        apply_out = x_in
        rapply_out = y_in
        normal_out = x_in
        for _ in range(iterations):
            apply_out = expr.apply(x_in)
            rapply_out = expr.rapply(y_in)
            normal_out = expr.rapply(expr.apply(x_in))
        return {
            "apply": apply_out,
            "rapply": rapply_out,
            "normal_apply": normal_out,
            "backend": backend,
        }

    def sc_public_none_call() -> dict[str, Any]:
        return _sc_public_call(expr_none, x_none, y_none)

    def sc_public_cheap_call() -> dict[str, Any]:
        return _sc_public_call(expr_cheap, x_cheap, y_cheap)

    # ----- spacecore lowered: precomputed dense matrix, bare matmuls -----
    def sc_lowered_call() -> dict[str, Any]:
        apply_out = x_bare
        rapply_out = y_bare
        normal_out = x_bare
        for _ in range(iterations):
            apply_out = matmul(A_dense_lowered, x_bare)
            rapply_out = matmul(A_dense_lowered_H, y_bare)
            normal_out = matmul(
                A_dense_lowered_H, matmul(A_dense_lowered, x_bare)
            )
        return {
            "apply": apply_out,
            "rapply": rapply_out,
            "normal_apply": normal_out,
            "backend": backend,
        }

    mode_callables: dict[ModeName, Any] = {
        "bare": bare_call,
        "spacecore_public_none": sc_public_none_call,
        "spacecore_public_cheap": sc_public_cheap_call,
        "spacecore_lowered": sc_lowered_call,
    }

    def reference_metric_extractor(result: dict[str, Any]) -> dict[str, float]:
        be = result.get("backend", backend)
        return {"apply_norm": _bare_norm(be, result["apply"])}

    return MacroPayload(
        iterations=iterations,
        size_params=dict(size_params),
        mode_callables=mode_callables,
        reference_metric_extractor=reference_metric_extractor,
        throughput_per_iteration=3.0,  # apply + rapply + normal_apply per iter
    )


registry.register(
    MacroBenchmark(
        name=_BENCHMARK_NAME,
        workload=(
            "Nested operator algebra: A = B_last @ ... @ "
            "(alpha1*A1 + alpha2*A2) @ ... @ B1; time apply, rapply, normal_apply."
        ),
        sizes=_SIZES,
        backends=_BACKENDS,
        factory=_factory,
        quick_sizes=_QUICK_SIZES,
        notes="Operator algebra stress (@, +, *) vs precomputed dense matmul.",
    )
)
