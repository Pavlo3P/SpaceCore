"""Cross-cutting contracts shared by every iterative solver.

Checklist section 8, Solvers (shared rows):

* Batched inputs are explicitly rejected with a clear shape error (0.4.0
  contract; to be replaced by batched support in a later release).
* Convergence is polled on the check interval, not only at the end.
* Vector outputs keep ``ctx.dtype`` and scalar diagnostics use the real dtype
  (ADR-015 Stage 1 workspace policy).
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


def _ctx(dtype=np.float64, check_level="standard"):
    return sc.Context(sc.NumpyOps(), dtype=dtype, check_level=check_level)


# ===========================================================================
# Batched inputs are rejected (0.4.0 contract)
# ===========================================================================
@pytest.mark.parametrize("solver", ["cg", "lsqr", "lanczos", "power"])
def test_iterative_solvers_explicitly_reject_batched_inputs(solver):
    ctx = _ctx(check_level="standard")
    space = sc.DenseCoordinateSpace((2,), ctx)
    operator = sc.DiagonalLinOp(ctx.asarray([2.0, 5.0]), space, ctx)
    batch = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(TypeError, match="Expected shape"):
        if solver == "cg":
            sc.cg(operator, batch)
        elif solver == "lsqr":
            sc.lsqr(operator, batch)
        elif solver == "lanczos":
            sc.lanczos_smallest(operator, batch, max_iter=2)
        else:
            sc.power_iteration(operator, x0=batch)


# ===========================================================================
# Convergence is polled on the check interval
# ===========================================================================
def test_iterative_solvers_poll_convergence_on_check_interval():
    ctx = _ctx(check_level="none")
    space = sc.DenseCoordinateSpace((2,), ctx)
    spd = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
    rectangular = sc.DenseLinOp(
        ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        space,
        sc.DenseCoordinateSpace((3,), ctx),
        ctx,
    )
    diagonal = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    cg_result = sc.cg(spd, ctx.asarray([1.0, 2.0]), maxiter=65)
    lsqr_result = sc.lsqr(rectangular, ctx.asarray([1.0, 2.0, 3.0]), maxiter=65)
    power_result = sc.power_iteration(diagonal, x0=ctx.asarray([1.0, 1.0]), maxiter=65)

    assert cg_result.num_iters < 64
    assert lsqr_result.num_iters == 64
    assert power_result.num_iters < 64
    np.testing.assert_allclose(cg_result.residual_norm, 0.0, atol=1e-12)
    np.testing.assert_allclose(lsqr_result.normal_residual_norm, 0.0, atol=1e-12)
    assert to_numpy(power_result.residual_norm) < 1e-6


# ===========================================================================
# Workspace dtype policy (ADR-015 Stage 1)
# ===========================================================================
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_solver_vector_and_scalar_workspace_dtypes_follow_stage_one_policy(dtype):
    ctx = _ctx(dtype, check_level="standard")
    real_dtype = ctx.ops.real_dtype(ctx.dtype)
    space = sc.DenseCoordinateSpace((2,), ctx)
    diagonal = ctx.asarray(np.asarray([2.0, 5.0], dtype=dtype))
    operator = sc.DiagonalLinOp(diagonal, space, ctx)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        b = ctx.asarray(np.asarray([1.0 + 2.0j, -0.5 + 0.75j], dtype=dtype))
        x0 = ctx.asarray(np.asarray([1.0 + 0.5j, 1.0 - 0.25j], dtype=dtype))
    else:
        b = ctx.asarray(np.asarray([1.0, -0.5], dtype=dtype))
        x0 = ctx.asarray(np.asarray([1.0, 1.0], dtype=dtype))

    cg_result = sc.cg(operator, b, tol=1e-6, maxiter=4, check_every=1)
    lsqr_result = sc.lsqr(operator, b, tol=1e-6, maxiter=4, check_every=1)
    power_result = sc.power_iteration(operator, x0=x0, tol=1e-5, maxiter=40)
    lanczos_result = sc.lanczos_smallest(operator, x0, max_iter=2, tol=1e-6, check_every=1)

    vector_results = (
        cg_result.x,
        lsqr_result.x,
        power_result.eigenvector,
        lanczos_result.eigenvector,
    )
    scalar_results = (
        cg_result.residual_norm,
        lsqr_result.residual_norm,
        lsqr_result.normal_residual_norm,
        power_result.eigenvalue,
        power_result.residual_norm,
        lanczos_result.eigenvalue,
        lanczos_result.residual_norm,
    )
    assert all(ctx.ops.get_dtype(value) == ctx.dtype for value in vector_results)
    assert all(np.asarray(to_numpy(value)).dtype == np.dtype(real_dtype) for value in scalar_results)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        assert np.any(np.abs(np.imag(to_numpy(cg_result.x))) > 0)
        assert np.any(np.abs(np.imag(to_numpy(lsqr_result.x))) > 0)
