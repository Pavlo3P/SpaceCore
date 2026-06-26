"""Batched density-matrix transform pipeline macrobenchmark.

For a batch of ``B`` density matrices ``rho`` of shape ``(D, D)`` with
``D = d0 * d`` the pipeline applies, per batch element:

1. Hermitian symmetrization: ``rho -> 0.5 * (rho + rho.conj().T)``.
2. Approximate PSD projection via eigen-clip:
   ``eigvals = clip(eigvals(rho), 0, inf)`` and reconstruct.
3. Trace normalization: ``rho /= trace(rho)``.
4. Partial trace down to ``(d0, d0)``: reshape ``rho`` to
   ``(d0, d, d0, d)`` and sum over axes ``(1, 3)``.
5. Spectral logarithm on the ``(d0, d0)`` result (eigen-clip eigvals to
   ``eps`` then take ``log``).
6. Lift back to ``(D, D)`` via Kronecker product with
   ``eye(d) / d``.

All four run modes share the same algebraic recipe; only the layer
through which the spectral/structural operations are issued changes:

* ``bare`` — pure backend ops on raw arrays.
* ``spacecore_public_none`` / ``spacecore_public_cheap`` —
  :class:`spacecore.HermitianSpace` and :class:`spacecore.StackedSpace`,
  with the corresponding :attr:`Context.check_level`.
* ``spacecore_lowered`` — JAX-jitted SpaceCore path; on NumPy / Torch it
  aliases the public-none callable.

Reference metrics:

* ``trace_error`` — mean ``|trace(rho_final) - 1/d|`` (because the
  Kronecker lift of a unit-trace ``log_rho`` against ``eye(d) / d`` has
  trace equal to ``trace(log_rho) / d``; we report the raw absolute
  trace deviation aggregated across the batch).
* ``psd_violation`` — mean over batch of ``max(-min_eigval, 0)`` after
  the spectral log step (the log of clipped eigenvalues may go strongly
  negative, so this is informational rather than an assertion).
* ``hermitian_error`` — mean Frobenius distance between ``rho_final``
  and ``rho_final.conj().T``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

import spacecore as sc

from .._operations import _backend_ctx, _np_dtype, _rng
from ._registry import MacroBenchmark, MacroPayload, registry
from ._schema import ModeName


_EPS = 1e-12


def _make_density_batch(
    rng: np.random.Generator, batch: int, D: int, dtype: np.dtype
) -> np.ndarray:
    """Return a ``(batch, D, D)`` array of well-conditioned PSD-ish matrices."""
    a = rng.standard_normal((batch, D, D)).astype(dtype)
    # Hermitize and add identity to push eigvals away from zero so the
    # log step is finite. This is *setup*, not part of the timed work.
    rho = 0.5 * (a + np.swapaxes(a, -1, -2))
    rho = rho + D * np.eye(D, dtype=dtype)[None, :, :]
    return rho


# ---------------------------------------------------------------------------
# NumPy bare implementation


def _bare_pipeline_numpy(
    rho_batch: np.ndarray, d0: int, d: int
) -> np.ndarray:
    """Pure NumPy version of the pipeline. Returns ``(batch, D, D)``."""
    D = d0 * d
    # 1. Hermitize.
    rho = 0.5 * (rho_batch + np.swapaxes(rho_batch, -1, -2))
    # 2. PSD project via eigen-clip.
    evals, evecs = np.linalg.eigh(rho)
    evals_pos = np.maximum(evals, 0.0)
    rho = np.einsum("...ij,...j,...kj->...ik", evecs, evals_pos, evecs)
    # 3. Trace normalize.
    tr = np.einsum("...ii->...", rho)
    rho = rho / tr[..., None, None]
    # 4. Partial trace to (d0, d0).
    rho_r = rho.reshape(rho.shape[0], d0, d, d0, d)
    rho_pt = np.einsum("...ikjl->...ij", rho_r) * 0.0  # init shape
    rho_pt = rho_r.trace(axis1=2, axis2=4)
    # 5. Spectral log.
    evals_pt, evecs_pt = np.linalg.eigh(rho_pt)
    log_evals = np.log(np.maximum(evals_pt, _EPS))
    log_rho = np.einsum("...ij,...j,...kj->...ik", evecs_pt, log_evals, evecs_pt)
    # 6. Lift via kron with eye(d) / d.
    eye_d = np.eye(d, dtype=rho.dtype) / d
    # Batched kron: log_rho is (B, d0, d0), eye_d is (d, d).
    lifted = np.einsum("bij,kl->bikjl", log_rho, eye_d).reshape(
        rho.shape[0], D, D
    )
    return lifted


# ---------------------------------------------------------------------------
# JAX bare implementation (closures over jnp ops).


def _bare_pipeline_jax_factory(d0: int, d: int) -> Callable[[Any], Any]:
    import jax.numpy as jnp

    D = d0 * d
    eye_d = jnp.eye(d) / d

    def pipeline(rho_batch):
        rho = 0.5 * (rho_batch + jnp.swapaxes(rho_batch, -1, -2))
        evals, evecs = jnp.linalg.eigh(rho)
        evals_pos = jnp.maximum(evals, 0.0)
        rho = jnp.einsum("...ij,...j,...kj->...ik", evecs, evals_pos, evecs)
        tr = jnp.einsum("...ii->...", rho)
        rho = rho / tr[..., None, None]
        B = rho.shape[0]
        rho_r = rho.reshape(B, d0, d, d0, d)
        rho_pt = jnp.trace(rho_r, axis1=2, axis2=4)
        evals_pt, evecs_pt = jnp.linalg.eigh(rho_pt)
        log_evals = jnp.log(jnp.maximum(evals_pt, _EPS))
        log_rho = jnp.einsum(
            "...ij,...j,...kj->...ik", evecs_pt, log_evals, evecs_pt
        )
        lifted = jnp.einsum("bij,kl->bikjl", log_rho, eye_d).reshape(B, D, D)
        return lifted

    return pipeline


# ---------------------------------------------------------------------------
# Torch bare implementation.


def _bare_pipeline_torch_factory(d0: int, d: int, eye_d_t: Any) -> Callable[[Any], Any]:
    import torch

    D = d0 * d

    def pipeline(rho_batch):
        rho = 0.5 * (rho_batch + torch.swapaxes(rho_batch, -1, -2))
        evals, evecs = torch.linalg.eigh(rho)
        evals_pos = torch.clamp(evals, min=0.0)
        rho = torch.einsum("...ij,...j,...kj->...ik", evecs, evals_pos, evecs)
        tr = torch.einsum("...ii->...", rho)
        rho = rho / tr[..., None, None]
        B = rho.shape[0]
        rho_r = rho.reshape(B, d0, d, d0, d)
        rho_pt = torch.diagonal(rho_r, dim1=2, dim2=4).sum(dim=-1)
        evals_pt, evecs_pt = torch.linalg.eigh(rho_pt)
        log_evals = torch.log(torch.clamp(evals_pt, min=_EPS))
        log_rho = torch.einsum(
            "...ij,...j,...kj->...ik", evecs_pt, log_evals, evecs_pt
        )
        lifted = torch.einsum("bij,kl->bikjl", log_rho, eye_d_t).reshape(B, D, D)
        return lifted

    return pipeline


# ---------------------------------------------------------------------------
# SpaceCore public-API pipeline.


def _sc_pipeline_factory(
    ctx: sc.Context,
    stacked_space: sc.StackedSpace,
    hermitian_pt: sc.HermitianSpace,
    stacked_pt: sc.StackedSpace,
    d0: int,
    d: int,
    eye_d_arr: Any,
) -> Callable[[Any], Any]:
    """Build a SpaceCore-public callable that walks one batch through the pipeline."""
    ops = ctx.ops
    D = d0 * d

    def _relu(lam: Any) -> Any:
        zero = ops.zeros_like(lam)
        return ops.maximum(lam, zero)

    def _log_clip(lam: Any) -> Any:
        eps_arr = ops.full_like(lam, _EPS)
        return ops.log(ops.maximum(lam, eps_arr))

    def pipeline(rho_batch):
        # 1. Hermitize through the stacked HermitianSpace.
        rho = stacked_space.base.symmetrize(rho_batch)
        # 2. PSD project via the stacked HermitianSpace public method.
        rho = stacked_space.spectral_apply(rho, _relu)
        # 3. Trace normalize.
        tr = ops.einsum("...ii->...", rho)
        rho = rho / tr[..., None, None]
        # 4. Partial trace via reshape + diagonal sum (backend op).
        B = rho.shape[0]
        rho_r = ops.reshape(rho, (B, d0, d, d0, d))
        # einsum batched partial trace.
        rho_pt = ops.einsum("bikjk->bij", rho_r)
        # 5. Spectral log via the stacked Hermitian partial-trace space.
        log_rho = stacked_pt.spectral_apply(rho_pt, _log_clip)
        # 6. Kron lift.
        lifted = ops.einsum("bij,kl->bikjl", log_rho, eye_d_arr).reshape(B, D, D)
        return lifted

    return pipeline


# ---------------------------------------------------------------------------
# Reference metric extractor.


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        # torch.Tensor
        return x.detach().cpu().numpy()
    # jax / cupy-style arrays use __array__.
    return np.asarray(x)


def _reference_metrics(result: Any) -> dict[str, float]:
    arr = _to_numpy(result)
    # arr is (B, D, D).
    arr.shape[0]
    # Trace error: average |trace| (target is a scalar that depends on
    # log magnitudes; we report raw mean of |trace|).
    traces = np.einsum("...ii->...", arr)
    trace_error = float(np.mean(np.abs(traces)))
    # PSD violation: mean over batch of max(-min_eigval, 0). The log
    # step makes the matrix non-PSD in general, so this is a magnitude
    # report, not a violation gate.
    herm = 0.5 * (arr + np.swapaxes(arr, -1, -2))
    eigs = np.linalg.eigvalsh(herm)
    psd_violation = float(np.mean(np.maximum(-eigs.min(axis=-1), 0.0)))
    # Hermitian error: Frobenius norm of antisymmetric part.
    anti = 0.5 * (arr - np.swapaxes(arr, -1, -2))
    hermitian_error = float(
        np.mean(np.sqrt(np.sum(anti * anti, axis=(-1, -2))))
    )
    return {
        "trace_error": trace_error,
        "psd_violation": psd_violation,
        "hermitian_error": hermitian_error,
    }


# ---------------------------------------------------------------------------
# Factory.


def _factory(
    *, backend: str, device: str, seed: int, size_params: dict[str, Any]
) -> MacroPayload:
    batch = int(size_params["batch"])
    d0 = int(size_params["d0"])
    d = int(size_params["d"])
    D = d0 * d

    # Build a single set of operand arrays on each backend / context.
    rng = _rng(seed)

    # Public-cheap context drives the operand dtype + asarray; reuse the
    # same NumPy operand pool for every mode so cross-mode error
    # comparisons are meaningful.
    ctx_cheap = _backend_ctx(backend, check_level="cheap")
    ctx_none = _backend_ctx(backend, check_level="none")
    np_dtype = _np_dtype(ctx_cheap)

    rho_np = _make_density_batch(rng, batch, D, np_dtype)
    eye_d_np = (np.eye(d) / d).astype(np_dtype)

    # Pre-materialize backend arrays once (operand construction is *not*
    # timed).
    rho_bare = ctx_cheap.asarray(rho_np)
    eye_d_bare = ctx_cheap.asarray(eye_d_np)
    rho_none = ctx_none.asarray(rho_np)
    rho_cheap = ctx_cheap.asarray(rho_np)
    eye_d_none = ctx_none.asarray(eye_d_np)
    eye_d_cheap = ctx_cheap.asarray(eye_d_np)

    # SpaceCore stacked spaces (one per check level).
    herm_full_none = sc.HermitianSpace(D, ctx=ctx_none)
    herm_pt_none = sc.HermitianSpace(d0, ctx=ctx_none)
    stacked_full_none = sc.StackedSpace(herm_full_none, batch, ctx_none)
    stacked_pt_none = sc.StackedSpace(herm_pt_none, batch, ctx_none)

    herm_full_cheap = sc.HermitianSpace(D, ctx=ctx_cheap)
    herm_pt_cheap = sc.HermitianSpace(d0, ctx=ctx_cheap)
    stacked_full_cheap = sc.StackedSpace(herm_full_cheap, batch, ctx_cheap)
    stacked_pt_cheap = sc.StackedSpace(herm_pt_cheap, batch, ctx_cheap)

    # Build bare callables per backend.
    if backend == "numpy":
        def bare_cb():
            return _bare_pipeline_numpy(rho_np, d0, d)
    elif backend == "jax":
        jax_pipeline = _bare_pipeline_jax_factory(d0, d)
        rho_jax = rho_bare  # already a jax array via ctx.asarray
        def bare_cb():
            return jax_pipeline(rho_jax)
    elif backend == "torch":
        torch_pipeline = _bare_pipeline_torch_factory(d0, d, eye_d_bare)
        rho_torch = rho_bare
        def bare_cb():
            return torch_pipeline(rho_torch)
    else:
        raise ValueError(f"unsupported backend {backend!r}")

    # SpaceCore public callables.
    sc_none_pipeline = _sc_pipeline_factory(
        ctx_none, stacked_full_none, herm_pt_none, stacked_pt_none, d0, d, eye_d_none
    )
    sc_cheap_pipeline = _sc_pipeline_factory(
        ctx_cheap, stacked_full_cheap, herm_pt_cheap, stacked_pt_cheap, d0, d, eye_d_cheap
    )
    def sc_none_cb():
        return sc_none_pipeline(rho_none)
    def sc_cheap_cb():
        return sc_cheap_pipeline(rho_cheap)

    # Lowered callable.
    if backend == "jax":
        import jax

        # JIT the public_none path: this exercises the lowered SpaceCore
        # implementation through JAX tracing.
        sc_lowered_jit = jax.jit(sc_none_pipeline)
        def sc_lowered_cb():
            return sc_lowered_jit(rho_none)
    else:
        # For NumPy / Torch the lowered path matches public_none.
        sc_lowered_cb = sc_none_cb

    mode_callables: dict[ModeName, Callable[[], Any]] = {
        "bare": bare_cb,
        "spacecore_public_none": sc_none_cb,
        "spacecore_public_cheap": sc_cheap_cb,
        "spacecore_lowered": sc_lowered_cb,
    }

    return MacroPayload(
        iterations=batch,
        size_params=dict(size_params),
        mode_callables=mode_callables,
        reference_metric_extractor=_reference_metrics,
        throughput_per_iteration=1.0,
    )


registry.register(
    MacroBenchmark(
        name="density_pipeline",
        workload=(
            "Batched density-matrix pipeline: hermitize -> PSD-clip -> "
            "trace-normalize -> partial trace -> spectral log -> kron lift."
        ),
        sizes={
            "b16_d4": {"batch": 16, "d0": 4, "d": 4},
            "b64_d8": {"batch": 64, "d0": 8, "d": 8},
            "b256_d16": {"batch": 256, "d0": 16, "d": 16},
        },
        backends=("numpy", "jax", "torch"),
        factory=_factory,
        quick_sizes=("b16_d4",),
        notes=(
            "iterations = batch size. Spectral-log step intentionally "
            "produces a non-PSD lifted result; psd_violation is a "
            "magnitude report, not an assertion."
        ),
    )
)
