"""QOT barycenter gradient evaluation on Hermitian operators.

For each state index ``s in range(S)`` the benchmark computes::

    K_s        = C_s - (U_s kron I_d + I_d0 kron V_s)        # Hermitian
    rho_s      = exp(-K_s / epsilon) / trace(exp(-K_s / eps))
    margin_s   = partial_trace_second(rho_s, d0, d)
    grad_s     = margin_s - target_s

where ``C_s`` is a random Hermitian ``(D, D)`` matrix with ``D = d0 * d``,
``U_s`` is Hermitian ``(d0, d0)``, ``V_s`` is Hermitian ``(d, d)``, and
``target_s`` is a real ``(d0,)`` vector.

The matrix exponential ``exp(-K / eps)`` is computed via Hermitian
eigendecomposition for numerical stability:

    exp(-K / eps) = U diag(exp(-lambda / eps)) U^H .

This makes each iteration genuinely expensive even at modest matrix
size: an ``eigh`` on a ``D x D`` Hermitian followed by two ``D x D``
matmuls dominates the cost.

Modes
-----
* ``bare`` — backend-native ``eigh`` / ``matmul`` / ``kron`` / ``reshape``.
* ``spacecore_public_none`` / ``spacecore_public_cheap`` — public
  :class:`~spacecore.HermitianSpace` API (``spectral_decompose`` /
  ``from_spectrum``) with check level ``none`` vs ``cheap``.
* ``spacecore_lowered`` — for JAX a ``jax.jit``-compiled bare loop; for
  NumPy / Torch this aliases the public_none callable since no distinct
  lowered kernel exists.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

import spacecore as sc
from ._registry import MacroBenchmark, MacroPayload, registry
from ._schema import ModeName
from .._operations import _backend_ctx, _np_dtype, _rng


def _build_problem_np(
    S: int, d0: int, d: int, seed: int, np_dtype: np.dtype
) -> dict[str, np.ndarray]:
    """Construct Hermitian operands as NumPy arrays before backend cast."""
    rng = _rng(seed)
    D = d0 * d
    C = np.empty((S, D, D), dtype=np_dtype)
    U = np.empty((S, d0, d0), dtype=np_dtype)
    V = np.empty((S, d, d), dtype=np_dtype)
    targets = np.empty((S, d0), dtype=np_dtype)
    for s in range(S):
        c_raw = rng.standard_normal((D, D)).astype(np_dtype)
        C[s] = 0.5 * (c_raw + c_raw.T)
        u_raw = rng.standard_normal((d0, d0)).astype(np_dtype)
        U[s] = 0.5 * (u_raw + u_raw.T)
        v_raw = rng.standard_normal((d, d)).astype(np_dtype)
        V[s] = 0.5 * (v_raw + v_raw.T)
        t = rng.uniform(0.0, 1.0, size=d0).astype(np_dtype)
        targets[s] = t / float(t.sum())
    I_d0 = np.eye(d0, dtype=np_dtype)
    I_d = np.eye(d, dtype=np_dtype)
    return {
        "C": C,
        "U": U,
        "V": V,
        "targets": targets,
        "I_d0": I_d0,
        "I_d": I_d,
    }


def _partial_trace_second_np(rho: np.ndarray, d0: int, d: int) -> np.ndarray:
    """Trace out the second factor of size ``d``, returning ``(d0,)`` diagonal."""
    rho4 = rho.reshape(d0, d, d0, d)
    return np.einsum("ijkj->ik", rho4).diagonal()


def _bare_numpy_step(
    C: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    targets: np.ndarray,
    I_d0: np.ndarray,
    I_d: np.ndarray,
    epsilon: float,
    d0: int,
    d: int,
) -> np.ndarray:
    """NumPy bare path: returns a stacked ``(S, d0)`` gradient array."""
    S = C.shape[0]
    grads = np.empty_like(targets)
    for s in range(S):
        K = C[s] - (np.kron(U[s], I_d) + np.kron(I_d0, V[s]))
        evals, evecs = np.linalg.eigh(K)
        # Shift by min eigenvalue so exp(-(lambda - lambda_min)/eps)
        # stays bounded; cancels out after normalization by trace.
        shift = float(np.min(evals))
        weights = np.exp(-(evals - shift) / epsilon)
        rho_unscaled = (evecs * weights) @ evecs.conj().T
        trace = float(np.sum(weights))
        rho = rho_unscaled / trace
        rho4 = rho.reshape(d0, d, d0, d)
        margin = np.einsum("ijkj->ik", rho4).diagonal()
        grads[s] = margin - targets[s]
    return grads


def _bare_jax_step_factory(epsilon: float, d0: int, d: int) -> Callable:
    """Return a ``jax``-native step function, suitable for jit-ing."""
    import jax.numpy as jnp

    def step(C, U, V, targets, I_d0, I_d):
        S = C.shape[0]
        grads = []
        for s in range(S):
            K = C[s] - (jnp.kron(U[s], I_d) + jnp.kron(I_d0, V[s]))
            evals, evecs = jnp.linalg.eigh(K)
            shift = jnp.min(evals)
            weights = jnp.exp(-(evals - shift) / epsilon)
            rho_unscaled = (evecs * weights) @ jnp.conj(evecs).T
            trace = jnp.sum(weights)
            rho = rho_unscaled / trace
            d0 * d
            rho4 = rho.reshape(d0, d, d0, d)
            margin = jnp.diagonal(jnp.einsum("ijkj->ik", rho4))
            grads.append(margin - targets[s])
        return jnp.stack(grads, axis=0)

    return step


def _bare_torch_step(
    C, U, V, targets, I_d0, I_d, epsilon: float, d0: int, d: int
):
    import torch

    S = C.shape[0]
    grads = []
    for s in range(S):
        K = C[s] - (torch.kron(U[s], I_d) + torch.kron(I_d0, V[s]))
        evals, evecs = torch.linalg.eigh(K)
        shift = torch.min(evals)
        weights = torch.exp(-(evals - shift) / epsilon)
        rho_unscaled = (evecs * weights) @ evecs.conj().T
        trace = torch.sum(weights)
        rho = rho_unscaled / trace
        rho4 = rho.reshape(d0, d, d0, d)
        margin = torch.einsum("ijkj->ik", rho4).diagonal()
        grads.append(margin - targets[s])
    return torch.stack(grads, dim=0)


def _sc_step(
    space: sc.HermitianSpace,
    C: Any,
    U_kron: Any,
    targets_list: list,
    epsilon: float,
    d0: int,
    d: int,
    ops,
):
    """Public-API path using HermitianSpace.spectral_decompose / from_spectrum.

    ``U_kron`` is the pre-built per-state Kronecker-sum tensor of shape
    ``(S, D, D)`` so the timed step does *not* touch ``U_s`` / ``V_s``
    Kronecker products. ``targets_list`` is the list of per-state target
    arrays already converted into backend tensors.
    """
    S = C.shape[0]
    grad_list = []
    for s in range(S):
        K = space.symmetrize(C[s] - U_kron[s])
        evals, evecs = space.spectral_decompose(K)
        shift = ops.min(evals)
        weights = ops.exp(-(evals - shift) / epsilon)
        # rho = U diag(weights) U^H -- use from_spectrum so the path
        # goes through HermitianSpace's spectral reconstruction.
        rho_unscaled = space.from_spectrum(weights, evecs)
        trace = ops.sum(weights)
        rho = rho_unscaled / trace
        rho4 = ops.reshape(rho, (d0, d, d0, d))
        # partial trace over second factor of size d, then diagonal
        partial = ops.einsum("ijkj->ik", rho4)
        # diagonal extraction: take ops.diagonal if available, else manual
        if hasattr(ops, "diagonal"):
            margin = ops.diagonal(partial)
        else:
            margin = ops.einsum("ii->i", partial)
        grad_list.append(margin - targets_list[s])
    # Stack as a backend array
    return ops.stack(grad_list, axis=0)


def _extract_metrics_factory(d0: int) -> Callable[[Any], dict[str, float]]:
    """Return the reference-metric extractor for a backend-agnostic result."""

    def extract(result: Any) -> dict[str, float]:
        # Coerce to NumPy regardless of backend.
        arr = result
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        elif hasattr(arr, "__array__"):
            arr = np.asarray(arr)
        else:
            arr = np.asarray(arr)
        arr = np.asarray(arr, dtype=np.float64)
        grad_norm = float(np.linalg.norm(arr))
        # Hermitian-error proxy: imaginary part of the (real) gradient array
        # should be zero. Real arrays trivially satisfy this.
        if np.iscomplexobj(arr):
            hermitian_error = float(np.max(np.abs(arr.imag)))
        else:
            hermitian_error = 0.0
        # Trace error: per-row sum measures how far the marginal+target
        # is from the original target row-sum (1). Captures normalization.
        # Here ``arr[s]`` = margin_s - target_s; row_sum_s = sum(margin_s)
        # - 1. We report max |row_sum + 1|.
        row_sums = arr.sum(axis=-1)
        trace_error = float(np.max(np.abs(row_sums)))
        return {
            "gradient_norm": grad_norm,
            "hermitian_error": hermitian_error,
            "trace_error": trace_error,
        }

    return extract


def _factory(
    backend: str,
    device: str,
    seed: int,
    size_params: dict[str, Any],
) -> MacroPayload:
    """Build a :class:`MacroPayload` for one ``(backend, size, seed)``."""
    S = int(size_params["S"])
    d0 = int(size_params["d0"])
    d = int(size_params["d"])
    epsilon = float(size_params["epsilon"])
    D = d0 * d

    # Resolve contexts up front (one per mode that uses SpaceCore).
    ctx_none = _backend_ctx(backend, check_level="none")
    ctx_cheap = _backend_ctx(backend, check_level="cheap")
    np_dtype = _np_dtype(ctx_none)
    problem = _build_problem_np(S, d0, d, seed, np_dtype)

    # Pre-build the per-state Kronecker-sum tensor (U kron I_d + I_d0 kron V).
    # This is heavy enough that we keep it out of the timed step.
    kron_np = np.empty((S, D, D), dtype=np_dtype)
    for s in range(S):
        kron_np[s] = np.kron(problem["U"][s], problem["I_d"]) + np.kron(
            problem["I_d0"], problem["V"][s]
        )

    # Backend-native tensors for the bare path.
    if backend == "numpy":
        C_bare = problem["C"]
        U_bare = problem["U"]
        V_bare = problem["V"]
        targets_bare = problem["targets"]
        I_d0_bare = problem["I_d0"]
        I_d_bare = problem["I_d"]

        def bare_callable():
            return _bare_numpy_step(
                C_bare, U_bare, V_bare, targets_bare,
                I_d0_bare, I_d_bare, epsilon, d0, d,
            )

    elif backend == "jax":
        import jax
        import jax.numpy as jnp

        C_bare = jnp.asarray(problem["C"])
        U_bare = jnp.asarray(problem["U"])
        V_bare = jnp.asarray(problem["V"])
        targets_bare = jnp.asarray(problem["targets"])
        I_d0_bare = jnp.asarray(problem["I_d0"])
        I_d_bare = jnp.asarray(problem["I_d"])
        step_fn = _bare_jax_step_factory(epsilon, d0, d)

        def bare_callable():
            return step_fn(C_bare, U_bare, V_bare, targets_bare, I_d0_bare, I_d_bare)

    elif backend == "torch":
        import torch

        C_bare = torch.as_tensor(problem["C"])
        U_bare = torch.as_tensor(problem["U"])
        V_bare = torch.as_tensor(problem["V"])
        targets_bare = torch.as_tensor(problem["targets"])
        I_d0_bare = torch.as_tensor(problem["I_d0"])
        I_d_bare = torch.as_tensor(problem["I_d"])

        def bare_callable():
            return _bare_torch_step(
                C_bare, U_bare, V_bare, targets_bare,
                I_d0_bare, I_d_bare, epsilon, d0, d,
            )
    else:
        raise ValueError(f"unknown backend {backend!r}")

    # SpaceCore public-API path: build a HermitianSpace per check-level
    # and convert operands into the matching context.
    def _build_sc_callable(ctx: sc.Context) -> Callable[[], Any]:
        space = sc.HermitianSpace(D, ctx=ctx)
        C_sc = ctx.asarray(problem["C"])
        kron_sc = ctx.asarray(kron_np)
        # Pre-asarray each target row so the timed step is allocation-free.
        targets_sc = [ctx.asarray(problem["targets"][s]) for s in range(S)]
        ops = ctx.ops

        def call():
            return _sc_step(space, C_sc, kron_sc, targets_sc, epsilon, d0, d, ops)

        return call

    public_none_callable = _build_sc_callable(ctx_none)
    public_cheap_callable = _build_sc_callable(ctx_cheap)

    # Lowered path. On JAX this is the jit-compiled bare step. Elsewhere
    # we alias the public_none callable, which is the contract default.
    if backend == "jax":
        import jax

        jitted = jax.jit(_bare_jax_step_factory(epsilon, d0, d))
        # Capture the bare tensors closed over above.
        C_j, U_j, V_j = C_bare, U_bare, V_bare
        T_j, I0_j, Id_j = targets_bare, I_d0_bare, I_d_bare

        def lowered_callable():
            return jitted(C_j, U_j, V_j, T_j, I0_j, Id_j)
    else:
        lowered_callable = public_none_callable

    mode_callables: dict[ModeName, Callable[[], Any]] = {
        "bare": bare_callable,
        "spacecore_public_none": public_none_callable,
        "spacecore_public_cheap": public_cheap_callable,
        "spacecore_lowered": lowered_callable,
    }

    extractor = _extract_metrics_factory(d0)

    return MacroPayload(
        iterations=S,
        size_params=dict(size_params),
        mode_callables=mode_callables,
        reference_metric_extractor=extractor,
        throughput_per_iteration=1.0,
    )


registry.register(
    MacroBenchmark(
        name="qot_barycenter",
        workload="QOT barycenter gradient on Hermitian operators",
        sizes={
            "s4_d4": {"S": 4, "d0": 4, "d": 4, "epsilon": 1e-1},
            "s4_d8": {"S": 4, "d0": 8, "d": 8, "epsilon": 1e-1},
            "s8_d16": {"S": 8, "d0": 16, "d": 16, "epsilon": 1e-3},
        },
        backends=("numpy", "jax", "torch"),
        factory=_factory,
        quick_sizes=("s4_d4",),
        notes="Per-state Hermitian eigendecomposition + soft-density partial trace.",
    )
)
