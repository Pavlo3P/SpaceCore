"""Tests for :func:`spacecore.expm_multiply` — Krylov matrix exponential.

Checklist section 8, ``expm_multiply``:

* ``t = 0`` returns the input; results match ``scipy.linalg.expm`` ground truth
  for dense and diagonal Hermitian generators.
* Real ``t`` (heat-equation style) and complex ``t`` (Schrodinger style):
  complex time is norm-preserving for a Hermitian generator and matches the
  dense truth.
* Linearity in ``v`` and the group property ``exp(t1)exp(t2) = exp(t1+t2)``.
* The inactive-tail sentinel is masked after early breakdown; the residual
  estimate decreases with more iterations.
* Rejects structurally non-Hermitian operators; JAX ``jit`` matches eager.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

import spacecore as sc

from tests._helpers import has_jax, jax_complex_dtype, to_numpy
from tests.linalg._helpers import backend_params, make_ctx, numpy_jax_params


def _operator(ctx, matrix):
    space = sc.DenseCoordinateSpace((matrix.shape[0],), ctx)
    return sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)


def _ground_truth(matrix, vector, t):
    return scipy.linalg.expm(t * matrix) @ vector


def _diagonal_expm_truth(diagonal, vector, t):
    return np.exp(t * diagonal) * vector


# ===========================================================================
# Basic behaviour & dense ground truth
# ===========================================================================
class TestGroundTruth:
    def test_t_zero_returns_input(self):
        ctx = make_ctx()
        A = _operator(ctx, np.array([[2.0, 0.5], [0.5, 3.0]]))
        v = ctx.asarray([1.0, -2.0])

        result = sc.expm_multiply(A, v, t=0.0, max_iter=4)

        np.testing.assert_allclose(result.result, v, rtol=1e-12, atol=1e-12)
        assert isinstance(result, sc.ExpmMultiplyResult)

    @pytest.mark.parametrize("backend_name,dtype", backend_params(cupy=False))
    def test_matches_dense_ground_truth(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        matrix = np.array([[1.0, 0.25, 0.0], [0.25, 2.0, -0.5], [0.0, -0.5, 3.0]])
        A = _operator(ctx, matrix)
        v_np = np.array([1.0, -2.0, 0.5])
        v = ctx.asarray(v_np)

        result = sc.expm_multiply(A, v, t=-0.2, max_iter=8, tol=1e-12)

        np.testing.assert_allclose(to_numpy(result.result), _ground_truth(matrix, v_np, -0.2), rtol=1e-5, atol=1e-5)
        assert bool(to_numpy(result.converged))

    @pytest.mark.parametrize("backend_name,dtype", numpy_jax_params())
    def test_masks_inactive_sentinel_for_early_breakdown(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        diagonal = np.array([1.5, 2.0, 3.0], dtype=np.float64)
        space = sc.DenseCoordinateSpace((3,), ctx)
        A = sc.DiagonalLinOp(ctx.asarray(diagonal), space, ctx)
        v_np = np.array([1.0, -2.0, 0.5], dtype=np.float64)

        result = sc.expm_multiply(A, ctx.asarray(v_np), t=1.0, max_iter=20, tol=1e-5)

        assert int(to_numpy(result.krylov_dim)) == 3
        assert np.all(np.isfinite(to_numpy(result.result)))
        np.testing.assert_allclose(
            to_numpy(result.result), _diagonal_expm_truth(diagonal, v_np, 1.0), rtol=1e-5, atol=1e-5
        )


# ===========================================================================
# Algebraic properties
# ===========================================================================
class TestAlgebra:
    def test_linear_in_vector(self):
        ctx = make_ctx()
        A = _operator(ctx, np.array([[1.0, 0.5], [0.5, 2.0]]))
        v1 = ctx.asarray([1.0, -1.0])
        v2 = ctx.asarray([0.5, 2.0])
        alpha, beta = 1.5, -0.25

        combined = sc.expm_multiply(A, alpha * v1 + beta * v2, t=0.3, max_iter=6).result
        expected = (
            alpha * sc.expm_multiply(A, v1, t=0.3, max_iter=6).result
            + beta * sc.expm_multiply(A, v2, t=0.3, max_iter=6).result
        )
        np.testing.assert_allclose(combined, expected, rtol=1e-10, atol=1e-10)

    def test_group_property(self):
        ctx = make_ctx()
        A = _operator(ctx, np.array([[1.0, 0.5], [0.5, 2.0]]))
        v = ctx.asarray([1.0, -1.0])
        t1, t2 = 0.2, -0.35

        first = sc.expm_multiply(A, v, t=t1, max_iter=6).result
        sequential = sc.expm_multiply(A, first, t=t2, max_iter=6).result
        direct = sc.expm_multiply(A, v, t=t1 + t2, max_iter=6).result
        np.testing.assert_allclose(sequential, direct, rtol=1e-10, atol=1e-10)

    def test_residual_estimate_decreases_with_more_iterations(self):
        ctx = make_ctx()
        A = _operator(ctx, np.array([[1.0, 0.25, 0.1], [0.25, 2.0, -0.5], [0.1, -0.5, 4.0]]))
        v = ctx.asarray([1.0, -2.0, 0.5])

        low = sc.expm_multiply(A, v, t=0.4, max_iter=1, tol=1e-12)
        high = sc.expm_multiply(A, v, t=0.4, max_iter=3, tol=1e-12)
        assert to_numpy(high.residual_estimate) <= to_numpy(low.residual_estimate)


# ===========================================================================
# Complex time (Schrodinger-style)
# ===========================================================================
class TestComplexTime:
    def test_complex_time_is_unitary_for_hermitian_generator(self):
        ctx = make_ctx(dtype=np.complex128)
        matrix = np.array([[1.0, 0.5 - 0.25j], [0.5 + 0.25j, 2.0]], dtype=np.complex128)
        A = _operator(ctx, matrix)
        v = ctx.asarray([1.0 + 0.5j, -0.25 + 0.75j])

        result = sc.expm_multiply(A, v, t=-0.5j, max_iter=6, tol=1e-12).result

        np.testing.assert_allclose(
            to_numpy(A.domain.norm(result)), to_numpy(A.domain.norm(v)), rtol=1e-10, atol=1e-10
        )

    def test_complex_time_matches_dense_truth(self):
        ctx = make_ctx(dtype=np.complex128)
        matrix = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.complex128)
        A = _operator(ctx, matrix)
        v_np = np.array([1.0 - 0.5j, 0.25 + 0.75j])

        result = sc.expm_multiply(A, ctx.asarray(v_np), t=-0.5j, max_iter=6, tol=1e-12)

        np.testing.assert_allclose(
            to_numpy(result.result), _ground_truth(matrix, v_np, -0.5j), rtol=1e-10, atol=1e-10
        )

    @pytest.mark.parametrize("backend_name,dtype", numpy_jax_params())
    def test_masks_inactive_sentinel_for_complex_time(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        diagonal = np.array([1.5, 2.0, 3.0], dtype=np.float64)
        space = sc.DenseCoordinateSpace((3,), ctx)
        A = sc.DiagonalLinOp(ctx.asarray(diagonal), space, ctx)
        v_np = np.array([1.0, -2.0, 0.5], dtype=np.float64)
        t = -0.75j

        result = sc.expm_multiply(A, ctx.asarray(v_np), t=t, max_iter=20, tol=1e-5)

        assert int(to_numpy(result.krylov_dim)) == 3
        assert np.all(np.isfinite(to_numpy(result.result)))
        np.testing.assert_allclose(
            to_numpy(result.result), _diagonal_expm_truth(diagonal, v_np, t), rtol=1e-5, atol=1e-5
        )


# ===========================================================================
# Rejections & jit
# ===========================================================================
class TestRejections:
    def test_rejects_structurally_non_hermitian_operator(self):
        ctx = make_ctx()
        A = _operator(ctx, np.array([[1.0, 2.0], [0.0, 3.0]]))
        with pytest.raises(ValueError, match="Hermitian"):
            sc.expm_multiply(A, ctx.asarray([1.0, -2.0]), t=0.1, max_iter=4)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_expm_multiply_jit_matches_eager():
    import jax

    ctx = make_ctx("jax", jax_complex_dtype())
    matrix = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.complex128)
    A = _operator(ctx, matrix)
    v = ctx.asarray([1.0 - 0.5j, 0.25 + 0.75j])

    eager = sc.expm_multiply(A, v, t=-0.5j, max_iter=6).result
    run = jax.jit(lambda op, x: sc.expm_multiply(op, x, t=-0.5j, max_iter=6).result)
    compiled = run(A, v)

    np.testing.assert_allclose(to_numpy(compiled), to_numpy(eager), rtol=1e-6, atol=1e-6)
