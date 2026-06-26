"""Power-method and Lanczos eigensolvers on a matrix-free operator.

Two macrobenchmarks share one operator:

* The operator ``A = L + diag(V)`` where ``L`` is the standard 1D
  Laplacian (Dirichlet, second-difference) on ``d`` points and ``V`` is
  a deterministic random potential keyed off ``seed``. ``A`` is real and
  symmetric, so its dominant eigenvalue (sought by power iteration) and
  smallest eigenvalue (sought by Lanczos) are both real.
* Power iteration runs ``K`` matvecs. Lanczos runs ``K`` Krylov steps
  with full reorthogonalization in the bare path; SpaceCore's
  ``lanczos_smallest`` uses its own internal reorthogonalization
  strategy.

Both benchmarks emit four run modes:

* ``bare`` — the operator action is the in-place expression
  ``-x[i-1] + 2 x[i] - x[i+1] + V[i] x[i]`` written in the active
  backend's native array dialect, with no SpaceCore types in the timed
  loop.
* ``spacecore_public_none`` / ``spacecore_public_cheap`` — the SpaceCore
  public API entry points (``sc.power_iteration`` /
  ``sc.lanczos_smallest``) wrapping the operator as a
  ``MatrixFreeLinOp``, timed with ``check_level="none"`` and
  ``check_level="cheap"`` respectively.
* ``spacecore_lowered`` — equivalent to ``spacecore_public_none`` on
  NumPy / Torch; on JAX it is the same callable wrapped in
  ``jax.jit`` so the runner reports a separate ``compile_time_ns``.

The operator is built once per ``(backend, size, seed)`` triple in the
factory body. Timed callables only invoke the solver — they do not
allocate the potential or convert dtypes. For the JAX paths, the public
context is built with the matching ``check_level`` so the runner
honours the cheap-check overhead.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

import spacecore as sc

from .._operations import _backend_ctx, _np_dtype, _rng
from ._registry import MacroBenchmark, MacroPayload, registry


# ---------------------------------------------------------------------------
# Shared operator construction.


def _make_potential(d: int, seed: int, np_dtype: np.dtype) -> np.ndarray:
    """Deterministic small random potential keyed off the seed."""
    rng = _rng(seed)
    # Small magnitude so the Laplacian's structure still dominates; this
    # keeps both the dominant and smallest eigenvalues well-defined.
    return np.asarray(rng.standard_normal(d) * 0.1, dtype=np_dtype)


def _bare_apply_numpy(v_np: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return the NumPy in-place Laplacian + diag(V) matvec."""

    def apply(x: np.ndarray) -> np.ndarray:
        out = 2.0 * x + v_np * x
        out[1:] -= x[:-1]
        out[:-1] -= x[1:]
        return out

    return apply


def _bare_apply_jax(v_jax: Any) -> Callable[[Any], Any]:
    """Return a pure-JAX Laplacian + diag(V) matvec."""
    import jax.numpy as jnp

    def apply(x: Any) -> Any:
        out = 2.0 * x + v_jax * x
        sub_left = jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), x[:-1]])
        sub_right = jnp.concatenate([x[1:], jnp.zeros((1,), dtype=x.dtype)])
        return out - sub_left - sub_right

    return apply


def _bare_apply_torch(v_t: Any) -> Callable[[Any], Any]:
    """Return a pure-Torch Laplacian + diag(V) matvec."""
    import torch

    def apply(x: Any) -> Any:
        out = 2.0 * x + v_t * x
        sub_left = torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), x[:-1]])
        sub_right = torch.cat([x[1:], torch.zeros(1, dtype=x.dtype, device=x.device)])
        return out - sub_left - sub_right

    return apply


def _bare_apply_for_backend(backend: str, v_native: Any) -> Callable[[Any], Any]:
    if backend == "numpy":
        return _bare_apply_numpy(v_native)
    if backend == "jax":
        return _bare_apply_jax(v_native)
    if backend == "torch":
        return _bare_apply_torch(v_native)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Power iteration helpers.


def _bare_power_iteration(
    backend: str,
    matvec: Callable[[Any], Any],
    x0_native: Any,
    K: int,
) -> tuple[float, float]:
    """Run K power-method steps and return ``(eigenvalue, final_norm)``.

    The bare path uses only the active backend's native operations so it
    represents the floor cost of the algorithm, with no SpaceCore types
    or wrapping in the timed loop.
    """
    if backend == "numpy":
        x = x0_native / np.linalg.norm(x0_native)
        for _ in range(K):
            w = matvec(x)
            n = np.linalg.norm(w)
            x = w / n
        Hx = matvec(x)
        eig = float(np.vdot(x, Hx).real)
        final_norm = float(np.linalg.norm(x))
        return eig, final_norm
    if backend == "jax":
        import jax.numpy as jnp

        x = x0_native / jnp.linalg.norm(x0_native)
        for _ in range(K):
            w = matvec(x)
            n = jnp.linalg.norm(w)
            x = w / n
        Hx = matvec(x)
        eig = jnp.real(jnp.vdot(x, Hx))
        final_norm = jnp.linalg.norm(x)
        return float(eig), float(final_norm)
    if backend == "torch":
        import torch

        x = x0_native / torch.linalg.norm(x0_native)
        for _ in range(K):
            w = matvec(x)
            n = torch.linalg.norm(w)
            x = w / n
        Hx = matvec(x)
        eig = torch.vdot(x, Hx).real
        final_norm = torch.linalg.norm(x)
        return float(eig), float(final_norm)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Lanczos helpers (full reorthogonalization for the bare reference).


def _bare_lanczos(
    backend: str,
    matvec: Callable[[Any], Any],
    x0_native: Any,
    K: int,
) -> tuple[float, float]:
    """Run K Lanczos steps with full reorthogonalization (bare path).

    Returns ``(estimated_smallest_eigenvalue, orthogonality_loss)``
    where ``orthogonality_loss = ||V^T V - I||_F`` for the
    reconstructed basis ``V`` of shape ``(K, d)``. Computation is done
    in the active backend's native array dialect.
    """
    if backend == "numpy":
        d = x0_native.shape[0]
        V = np.zeros((K, d), dtype=x0_native.dtype)
        alphas = np.zeros(K, dtype=x0_native.dtype)
        betas = np.zeros(K, dtype=x0_native.dtype)
        v0 = x0_native / np.linalg.norm(x0_native)
        V[0] = v0
        for i in range(K):
            w = matvec(V[i])
            a = float(np.vdot(V[i], w).real)
            alphas[i] = a
            w = w - a * V[i]
            if i > 0:
                w = w - betas[i - 1] * V[i - 1]
            proj = V[: i + 1] @ w
            w = w - proj @ V[: i + 1]
            b = float(np.linalg.norm(w))
            if i + 1 < K:
                betas[i] = b
                if b > 1e-14:
                    V[i + 1] = w / b
        T_alphas = alphas[:K].astype(np.float64)
        T_betas = betas[: K - 1].astype(np.float64) if K > 1 else np.zeros(0)
        T = np.diag(T_alphas) + np.diag(T_betas, 1) + np.diag(T_betas, -1)
        eig = float(np.linalg.eigvalsh(T)[0])
        gram = V @ V.T
        ortho = float(np.linalg.norm(gram - np.eye(K, dtype=gram.dtype)))
        return eig, ortho
    if backend == "jax":
        import jax.numpy as jnp

        v0 = x0_native / jnp.linalg.norm(x0_native)
        V_rows = [v0]
        alphas: list[Any] = []
        betas: list[Any] = []
        for i in range(K):
            w = matvec(V_rows[i])
            a = jnp.real(jnp.vdot(V_rows[i], w))
            alphas.append(a)
            w = w - a * V_rows[i]
            if i > 0:
                w = w - betas[i - 1] * V_rows[i - 1]
            V_stack = jnp.stack(V_rows[: i + 1], axis=0)
            proj = V_stack @ w
            w = w - proj @ V_stack
            b = jnp.linalg.norm(w)
            if i + 1 < K:
                betas.append(b)
                w_next = jnp.where(
                    b > 1e-14, w / jnp.maximum(b, 1e-30), jnp.zeros_like(w)
                )
                V_rows.append(w_next)
        V = jnp.stack(V_rows[:K], axis=0)
        T_alphas = jnp.stack(alphas[:K]).astype(jnp.float64)
        if K > 1:
            T_betas = jnp.stack(betas[: K - 1]).astype(jnp.float64)
        else:
            T_betas = jnp.zeros(0, dtype=jnp.float64)
        T = jnp.diag(T_alphas) + jnp.diag(T_betas, 1) + jnp.diag(T_betas, -1)
        eig = jnp.linalg.eigvalsh(T)[0]
        gram = V @ V.T
        ortho = jnp.linalg.norm(gram - jnp.eye(K, dtype=gram.dtype))
        return float(eig), float(ortho)
    if backend == "torch":
        import torch

        v0 = x0_native / torch.linalg.norm(x0_native)
        V_rows = [v0]
        alphas_t: list[Any] = []
        betas_t: list[Any] = []
        for i in range(K):
            w = matvec(V_rows[i])
            a = torch.vdot(V_rows[i], w).real
            alphas_t.append(a)
            w = w - a * V_rows[i]
            if i > 0:
                w = w - betas_t[i - 1] * V_rows[i - 1]
            V_stack = torch.stack(V_rows[: i + 1], dim=0)
            proj = V_stack @ w
            w = w - proj @ V_stack
            b = torch.linalg.norm(w)
            if i + 1 < K:
                betas_t.append(b)
                if float(b) > 1e-14:
                    V_rows.append(w / b)
                else:
                    V_rows.append(torch.zeros_like(w))
        V = torch.stack(V_rows[:K], dim=0)
        T_alphas = torch.stack(alphas_t[:K]).to(torch.float64)
        if K > 1:
            T_betas = torch.stack(betas_t[: K - 1]).to(torch.float64)
        else:
            T_betas = torch.zeros(0, dtype=torch.float64)
        T = torch.diag(T_alphas) + torch.diag(T_betas, 1) + torch.diag(T_betas, -1)
        eig = torch.linalg.eigvalsh(T)[0]
        gram = V @ V.T
        ortho = torch.linalg.norm(
            gram - torch.eye(K, dtype=gram.dtype, device=gram.device)
        )
        return float(eig), float(ortho)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# SpaceCore callable builders.


def _build_sc_callable_power(
    backend: str,
    ctx_check: str,
    v_np: np.ndarray,
    d: int,
    K: int,
    x0_np: np.ndarray,
    jit: bool,
) -> Callable[[], Any]:
    """Build the SpaceCore power-iteration callable for one check level."""
    ctx = _backend_ctx(backend, check_level=ctx_check)
    space = sc.DenseCoordinateSpace((d,), ctx)
    v_ctx = ctx.asarray(v_np)
    x0_ctx = ctx.asarray(x0_np)

    bare_apply = _bare_apply_for_backend(backend, v_ctx)

    op = sc.MatrixFreeLinOp(bare_apply, bare_apply, space, space, ctx)

    def call() -> Any:
        return sc.power_iteration(op, x0=x0_ctx, tol=0.0, maxiter=K)

    if jit and backend == "jax":
        import jax

        jitted = jax.jit(
            lambda x: sc.power_iteration(op, x0=x, tol=0.0, maxiter=K).eigenvalue
        )

        def call_jit() -> Any:
            return jitted(x0_ctx)

        return call_jit

    return call


def _build_sc_callable_lanczos(
    backend: str,
    ctx_check: str,
    v_np: np.ndarray,
    d: int,
    K: int,
    x0_np: np.ndarray,
    jit: bool,
) -> Callable[[], Any]:
    """Build the SpaceCore lanczos_smallest callable for one check level."""
    ctx = _backend_ctx(backend, check_level=ctx_check)
    space = sc.DenseCoordinateSpace((d,), ctx)
    v_ctx = ctx.asarray(v_np)
    x0_ctx = ctx.asarray(x0_np)

    bare_apply = _bare_apply_for_backend(backend, v_ctx)

    op = sc.MatrixFreeLinOp(bare_apply, bare_apply, space, space, ctx)

    def call() -> Any:
        return sc.lanczos_smallest(op, x0_ctx, max_iter=K, tol=0.0)

    if jit and backend == "jax":
        import jax

        jitted = jax.jit(
            lambda x: sc.lanczos_smallest(op, x, max_iter=K, tol=0.0).eigenvalue
        )

        def call_jit() -> Any:
            return jitted(x0_ctx)

        return call_jit

    return call


# ---------------------------------------------------------------------------
# Factories.


def _native_arrays(
    backend: str, v_np: np.ndarray, x0_np: np.ndarray
) -> tuple[Any, Any]:
    """Move ``(potential, initial vector)`` to the backend's native dtype."""
    if backend == "numpy":
        return v_np, x0_np
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(v_np), jnp.asarray(x0_np)
    if backend == "torch":
        import torch

        return torch.as_tensor(v_np), torch.as_tensor(x0_np)
    raise ValueError(f"unknown backend {backend!r}")


def _factory_power(
    backend: str,
    device: str,
    seed: int,
    size_params: dict[str, Any],
) -> MacroPayload:
    d = int(size_params["d"])
    K = int(size_params["K"])

    # Use the public-none context to derive a canonical dtype for the
    # operator data; the cheap-checked context shares the same dtype.
    base_ctx = _backend_ctx(backend, check_level="none")
    np_dtype = _np_dtype(base_ctx)

    v_np = _make_potential(d, seed, np_dtype)
    x0_np = np.asarray(
        _rng(seed + 1).standard_normal(d), dtype=np_dtype
    )

    v_native, x0_native = _native_arrays(backend, v_np, x0_np)
    bare_matvec = _bare_apply_for_backend(backend, v_native)

    def bare_call() -> tuple[float, float]:
        return _bare_power_iteration(backend, bare_matvec, x0_native, K)

    mode_callables: dict[str, Callable[[], Any]] = {
        "bare": bare_call,
        "spacecore_public_none": _build_sc_callable_power(
            backend, "none", v_np, d, K, x0_np, jit=False
        ),
        "spacecore_public_cheap": _build_sc_callable_power(
            backend, "cheap", v_np, d, K, x0_np, jit=False
        ),
    }
    if backend == "jax":
        mode_callables["spacecore_lowered"] = _build_sc_callable_power(
            backend, "none", v_np, d, K, x0_np, jit=True
        )
    else:
        mode_callables["spacecore_lowered"] = mode_callables["spacecore_public_none"]

    def extractor(result: Any) -> dict[str, float]:
        # Bare path returns a plain ``(eig, final_norm)`` tuple.
        # SC eager path returns a ``PowerIterationResult`` NamedTuple.
        # SC JIT path returns the eigenvalue scalar directly.
        if hasattr(result, "eigenvalue"):
            return {"estimated_eigenvalue": float(result.eigenvalue)}
        if isinstance(result, tuple) and len(result) == 2:
            eig, fin = result
            return {
                "estimated_eigenvalue": float(eig),
                "final_norm": float(fin),
            }
        return {"estimated_eigenvalue": float(result)}

    return MacroPayload(
        iterations=K,
        size_params=dict(size_params),
        mode_callables=mode_callables,  # type: ignore[arg-type]
        reference_metric_extractor=extractor,
        throughput_per_iteration=1.0,
    )


def _factory_lanczos(
    backend: str,
    device: str,
    seed: int,
    size_params: dict[str, Any],
) -> MacroPayload:
    d = int(size_params["d"])
    K = int(size_params["K"])

    base_ctx = _backend_ctx(backend, check_level="none")
    np_dtype = _np_dtype(base_ctx)

    v_np = _make_potential(d, seed, np_dtype)
    x0_np = np.asarray(
        _rng(seed + 2).standard_normal(d), dtype=np_dtype
    )

    v_native, x0_native = _native_arrays(backend, v_np, x0_np)
    bare_matvec = _bare_apply_for_backend(backend, v_native)

    def bare_call() -> tuple[float, float]:
        return _bare_lanczos(backend, bare_matvec, x0_native, K)

    mode_callables: dict[str, Callable[[], Any]] = {
        "bare": bare_call,
        "spacecore_public_none": _build_sc_callable_lanczos(
            backend, "none", v_np, d, K, x0_np, jit=False
        ),
        "spacecore_public_cheap": _build_sc_callable_lanczos(
            backend, "cheap", v_np, d, K, x0_np, jit=False
        ),
    }
    if backend == "jax":
        mode_callables["spacecore_lowered"] = _build_sc_callable_lanczos(
            backend, "none", v_np, d, K, x0_np, jit=True
        )
    else:
        mode_callables["spacecore_lowered"] = mode_callables["spacecore_public_none"]

    def extractor(result: Any) -> dict[str, float]:
        if hasattr(result, "eigenvalue"):
            return {"estimated_eigenvalue": float(result.eigenvalue)}
        if isinstance(result, tuple) and len(result) == 2:
            eig, ortho = result
            return {
                "estimated_eigenvalue": float(eig),
                "orthogonality_loss": float(ortho),
            }
        return {"estimated_eigenvalue": float(result)}

    return MacroPayload(
        iterations=K,
        size_params=dict(size_params),
        mode_callables=mode_callables,  # type: ignore[arg-type]
        reference_metric_extractor=extractor,
        throughput_per_iteration=1.0,
    )


# ---------------------------------------------------------------------------
# Registry entries.


_SIZES: dict[str, dict[str, Any]] = {
    "d=10000_K=50": {"d": 10_000, "K": 50},
    "d=100000_K=200": {"d": 100_000, "K": 200},
}

_QUICK_SIZES = ("d=10000_K=50",)


registry.register(
    MacroBenchmark(
        name="power_iteration",
        workload="power method on -Laplacian + diag(V), matrix-free",
        sizes=_SIZES,
        backends=("numpy", "jax", "torch"),
        factory=_factory_power,
        quick_sizes=_QUICK_SIZES,
        notes="K power-method steps; one matvec per iteration.",
    )
)


registry.register(
    MacroBenchmark(
        name="lanczos",
        workload="K-step Lanczos on -Laplacian + diag(V), matrix-free",
        sizes=_SIZES,
        backends=("numpy", "jax", "torch"),
        factory=_factory_lanczos,
        quick_sizes=_QUICK_SIZES,
        notes="Fixed-K Lanczos; bare path uses full reorthogonalization.",
    )
)
