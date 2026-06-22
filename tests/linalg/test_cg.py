"""Tests for :func:`spacecore.cg` — conjugate gradients.

Checklist section 8, ``cg``:

* Solves SPD systems and matches a direct ``solve`` (dense / diagonal /
  matrix-free, real and complex Hermitian).
* Convergence diagnostics (``converged`` / ``num_iters`` / ``residual_norm``)
  are correct and observable; tolerance is respected.
* ``x0`` default is the zero vector and an explicit ``x0`` is honored.
* Geometry awareness: weighted inner products give the metric solution and
  metric residual norm.
* JAX ``jit`` round-trip preserves correctness.
* Rectangular operators and (under checks) batched right-hand sides are
  rejected; strict checks reject non-Hermitian operators.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype, to_numpy
from tests.linalg._helpers import backend_params, make_ctx


def _metric_spd_matrix(ctx):
    """SPD-in-metric operator matrix for the weighted-space tests."""
    weights = np.asarray([2.0, 5.0, 11.0])
    symmetric_spd = np.asarray(
        [[6.0, 1.0, 0.5], [1.0, 5.0, -0.25], [0.5, -0.25, 4.0]]
    )
    return ctx.asarray(symmetric_spd / weights[:, None])


def _weighted_space(ctx):
    return sc.DenseCoordinateSpace(
        (3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0, 11.0]))
    )


# ===========================================================================
# SPD solves across backends
# ===========================================================================
class TestSPDSolves:
    @pytest.mark.parametrize("backend_name,dtype", backend_params())
    def test_solves_small_spd_system(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
        b = ctx.asarray([1.0, 2.0])

        result = sc.cg(A, b, tol=1e-7, maxiter=10)

        np.testing.assert_allclose(
            to_numpy(result.x),
            np.linalg.solve([[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0]),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(to_numpy(A.apply(result.x)), to_numpy(b), rtol=1e-5, atol=1e-5)
        assert bool(to_numpy(result.converged))

    def test_solves_complex_hermitian_positive_definite(self):
        ctx = make_ctx(dtype=np.complex128)
        space = sc.DenseCoordinateSpace((2,), ctx)
        matrix = np.array([[4.0, 1.0 + 1.0j], [1.0 - 1.0j, 3.0]], dtype=np.complex128)
        A = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)
        b = ctx.asarray([1.0 + 2.0j, 3.0 - 1.0j])

        result = sc.cg(A, b, tol=1e-10, maxiter=10)

        np.testing.assert_allclose(to_numpy(result.x), np.linalg.solve(matrix, to_numpy(b)), rtol=1e-8)
        assert bool(to_numpy(result.converged))

    @pytest.mark.parametrize("operator_kind", ["dense", "diagonal", "matrix-free"])
    def test_reference_cases_match_direct_solve(self, operator_kind):
        ctx = make_ctx(check_level="standard")
        space = sc.DenseCoordinateSpace((3,), ctx)
        matrix = np.asarray([[6.0, 1.0, 0.5], [1.0, 5.0, -0.25], [0.5, -0.25, 4.0]])
        if operator_kind == "diagonal":
            matrix = np.diag([2.0, 5.0, 11.0])
            operator = sc.DiagonalLinOp(ctx.asarray(np.diag(matrix)), space, ctx)
        elif operator_kind == "matrix-free":
            backend = ctx.asarray(matrix)
            operator = sc.MatrixFreeLinOp(
                lambda x: backend @ x, lambda y: backend.T @ y, space, space, ctx
            )
        else:
            operator = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)
        b = ctx.asarray([1.0, -2.0, 3.0])

        result = sc.cg(operator, b, tol=1e-12, maxiter=8, check_every=1)
        expected = np.linalg.solve(matrix, to_numpy(b))

        np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(to_numpy(operator.apply(result.x) - b), 0.0, atol=1e-10)
        assert bool(to_numpy(result.converged))
        assert 0 < int(to_numpy(result.num_iters)) <= 3


# ===========================================================================
# Convergence diagnostics & tolerance
# ===========================================================================
class TestDiagnostics:
    def test_tolerance_and_iteration_status_observable(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        operator = sc.DiagonalLinOp(ctx.asarray([2.0, 5.0]), space, ctx)
        b = ctx.asarray([1.0, 1.0])

        stopped = sc.cg(operator, b, tol=0.0, atol=0.0, maxiter=0)
        converged = sc.cg(operator, b, tol=1e-12, maxiter=2, check_every=1)

        assert not bool(to_numpy(stopped.converged))
        assert int(to_numpy(stopped.num_iters)) == 0
        assert bool(to_numpy(converged.converged))
        assert int(to_numpy(converged.num_iters)) <= 2

    def test_float64_spd_residual_below_1e_minus_10(self):
        ctx = make_ctx(dtype=np.float64)
        space = sc.DenseCoordinateSpace((4,), ctx)
        matrix = np.array(
            [[6.0, 1.0, 0.5, 0.0], [1.0, 5.0, 0.0, 0.25],
             [0.5, 0.0, 4.0, 0.75], [0.0, 0.25, 0.75, 3.0]],
            dtype=np.float64,
        )
        A = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)
        b = ctx.asarray([1.0, -2.0, 0.5, 3.0])

        result = sc.cg(A, b, tol=1e-13, maxiter=20, check_every=1)

        assert bool(to_numpy(result.converged))
        assert float(to_numpy(space.norm(A.apply(result.x) - b))) < 1e-10

    def test_regression_removes_sqrt_epsilon_residual_floor(self):
        ctx = make_ctx(dtype=np.float64)
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DiagonalLinOp(ctx.asarray([1.0, 1.0e4]), space, ctx)
        b = ctx.asarray([1.0, 1.0e-12])

        result = sc.cg(A, b, tol=1e-13, maxiter=4, check_every=1)

        assert bool(to_numpy(result.converged))
        assert float(to_numpy(space.norm(A.apply(result.x) - b))) < 1e-10

    def test_final_iteration_refreshes_residual_with_sparse_checks(self):
        ctx = make_ctx(dtype=np.float64)
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
        b = ctx.asarray([1.0, 2.0])

        result = sc.cg(A, b, tol=1e-12, maxiter=2, check_every=10)

        actual_residual = space.norm(A.apply(result.x) - b)
        assert bool(to_numpy(result.converged))
        np.testing.assert_allclose(to_numpy(result.residual_norm), to_numpy(actual_residual), atol=1e-14)


# ===========================================================================
# x0 handling
# ===========================================================================
class TestInitialGuess:
    def test_default_x0_is_zero_vector(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
        b = ctx.asarray([1.0, 2.0])
        # No x0 supplied -> defaults to zeros and still converges.
        result = sc.cg(A, b, tol=1e-12, maxiter=10, check_every=1)
        assert bool(to_numpy(result.converged))

    def test_exact_x0_converges_immediately(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DiagonalLinOp(ctx.asarray([2.0, 5.0]), space, ctx)
        b = ctx.asarray([2.0, 5.0])
        x_star = ctx.asarray([1.0, 1.0])

        result = sc.cg(A, b, x0=x_star, tol=1e-12, maxiter=5, check_every=1)

        assert bool(to_numpy(result.converged))
        assert int(to_numpy(result.num_iters)) == 0


# ===========================================================================
# Geometry awareness (weighted inner products)
# ===========================================================================
class TestWeightedGeometry:
    def test_uses_weighted_inner_products_and_residual_norms(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        space = _weighted_space(ctx)
        A = sc.DenseLinOp(_metric_spd_matrix(ctx), space, space, ctx)
        x_true = ctx.asarray([1.0, -2.0, 0.5])
        b = A.apply(x_true)

        result = sc.cg(A, b, tol=1e-12, maxiter=8, check_every=2)
        residual = space.add(A.apply(result.x), space.scale(-1.0, b))

        np.testing.assert_allclose(to_numpy(result.x), to_numpy(x_true), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(to_numpy(result.residual_norm), to_numpy(space.norm(residual)), atol=1e-12)
        assert bool(result.converged)


# ===========================================================================
# Rejections
# ===========================================================================
class TestRejections:
    def test_rejects_rectangular_operator(self):
        ctx = make_ctx()
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)
        with pytest.raises(ValueError, match="square LinOp"):
            sc.cg(A, ctx.asarray([1.0, 2.0, 3.0]))

    def test_strict_checks_reject_non_hermitian(self):
        ctx = make_ctx(check_level="strict")
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, 3.0]]), space, space, ctx)
        with pytest.raises(ValueError, match="Hermitian"):
            sc.cg(A, ctx.asarray([1.0, 2.0]))


# ===========================================================================
# JAX jit
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_cg_jit_compiles_with_operator_argument():
    import jax

    ctx = make_ctx("jax", jax_real_dtype())
    space = sc.DenseCoordinateSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)

    solve = jax.jit(lambda A, b: sc.cg(A, b, maxiter=10).x)
    x = solve(op, ctx.asarray([1.0, 2.0]))

    np.testing.assert_allclose(to_numpy(x), [0.09090909, 0.63636364], rtol=1e-5, atol=1e-5)
