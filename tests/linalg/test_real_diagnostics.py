"""Solver scalar diagnostics are real for complex Hermitian operators (ADR-015)."""
import numpy as np
import pytest

import spacecore as sc


def _hermitian_problem(ops, dtype):
    ctx = sc.Context(ops, dtype=dtype)
    M = np.array([[4.0, 1.0 - 1.0j], [1.0 + 1.0j, 3.0]])
    X = sc.DenseCoordinateSpace((2,), ctx)
    return ctx, X, sc.DenseLinOp(ctx.asarray(M), X, X, ctx), M


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_cg_and_power_iteration_on_a_complex_hermitian_operator(backend):
    if backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        ops, dtype = sc.JaxOps(), jnp.complex64
    else:
        ops, dtype = sc.NumpyOps(), np.complex128
    ctx, X, A, M = _hermitian_problem(ops, dtype)
    b = ctx.asarray(np.array([1.0 + 0.5j, -2.0]))
    x = np.asarray(sc.cg(A, b, tol=1e-6).x)
    np.testing.assert_allclose(M @ x, np.asarray(b), atol=1e-4)
    result = sc.power_iteration(A, x0=ctx.asarray(np.array([1.0, 1.0j])), tol=1e-6, maxiter=200)
    assert not np.iscomplexobj(np.asarray(result.eigenvalue))
    np.testing.assert_allclose(float(result.eigenvalue), np.linalg.eigvalsh(M).max(), rtol=1e-4)
