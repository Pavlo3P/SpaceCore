"""Tests for :func:`spacecore.lsqr` — least-squares via LSQR.

Checklist section 8, ``lsqr``:

* Solves over- and under-determined systems and matches ``numpy.linalg.lstsq``
  (real and complex).
* Uses ``A.H.apply`` (the metric adjoint) for adjoint products; matrix-free
  operators route through ``rapply``.
* ``residual_mode`` ``"exact"`` vs ``"recurrence"``: matching solutions /
  diagnostics, fewer operator applications, and rejection of unknown modes.
* Geometry awareness on weighted spaces (metric normal equations).
* JAX ``jit`` round-trip preserves correctness.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype, to_numpy
from tests.linalg._helpers import backend_params, make_ctx


# ===========================================================================
# Least-squares solves across backends
# ===========================================================================
class TestSolves:
    @pytest.mark.parametrize("backend_name,dtype", backend_params())
    def test_solves_rectangular_least_squares(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        A = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
        b = ctx.asarray([1.0, 2.0, 4.0])

        result = sc.lsqr(A, b, tol=1e-7, maxiter=10)

        expected, *_ = np.linalg.lstsq(matrix, np.array([1.0, 2.0, 4.0]), rcond=None)
        np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(to_numpy(A.H.apply(A.apply(result.x) - b)), [0.0, 0.0], atol=1e-5)
        assert bool(to_numpy(result.converged))

    @pytest.mark.parametrize(
        "matrix,b",
        [
            (
                np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]),
                np.asarray([1.0, 2.0, 2.5, -0.5]),
            ),
            (
                np.asarray([[1.0, 0.0, 1.0], [0.0, 2.0, -1.0]]),
                np.asarray([2.0, -1.0]),
            ),
            (
                np.asarray([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]),
                np.asarray([3.0, -4.0, 1.0, 0.25]),
            ),
        ],
        ids=["overdetermined", "underdetermined", "rectangular-diagonal"],
    )
    def test_reference_cases_match_numpy_lstsq(self, matrix, b):
        ctx = make_ctx(check_level="standard")
        domain = sc.DenseCoordinateSpace((matrix.shape[1],), ctx)
        codomain = sc.DenseCoordinateSpace((matrix.shape[0],), ctx)
        operator = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
        b_backend = ctx.asarray(b)

        result = sc.lsqr(operator, b_backend, tol=1e-12, maxiter=20, check_every=1)
        expected, *_ = np.linalg.lstsq(matrix, b, rcond=None)
        residual = operator.apply(result.x) - b_backend
        normal_residual = operator.H.apply(residual)

        np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            to_numpy(result.residual_norm), to_numpy(codomain.norm(residual)), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            to_numpy(result.normal_residual_norm),
            to_numpy(domain.norm(normal_residual)),
            rtol=1e-10,
            atol=1e-10,
        )
        assert bool(to_numpy(result.converged))

    def test_solves_complex_least_squares(self):
        ctx = make_ctx(dtype=np.complex128)
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        matrix = np.array([[1.0 + 1.0j, 0.0], [0.0, 2.0 - 1.0j], [1.0, 1.0j]], dtype=np.complex128)
        A = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
        b = ctx.asarray([1.0 - 1.0j, 2.0 + 0.5j, 3.0j])

        result = sc.lsqr(A, b, tol=1e-10, maxiter=20)

        expected, *_ = np.linalg.lstsq(matrix, to_numpy(b), rcond=None)
        np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-7, atol=1e-7)
        assert bool(to_numpy(result.converged))


# ===========================================================================
# Matrix-free adjoint routing
# ===========================================================================
class TestMatrixFree:
    def test_uses_rapply_for_adjoint_products(self):
        ctx = make_ctx()
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        matrix = ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        calls = {"rapply": 0}

        def apply(x):
            return matrix @ x

        def rapply(y):
            calls["rapply"] += 1
            return matrix.T @ y

        A = sc.MatrixFreeLinOp(apply, rapply, domain, codomain, ctx)
        result = sc.lsqr(A, ctx.asarray([1.0, 2.0, 3.0]), tol=1e-8, maxiter=10)

        np.testing.assert_allclose(to_numpy(result.x), [1.0, 2.0], rtol=1e-6, atol=1e-6)
        assert calls["rapply"] > 0


# ===========================================================================
# residual_mode
# ===========================================================================
class TestResidualMode:
    def _matrix_free(self, ctx, matrix, calls):
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)

        def apply(x):
            calls["apply"] += 1
            return matrix @ x

        def rapply(y):
            calls["rapply"] += 1
            return matrix.T @ y

        return sc.MatrixFreeLinOp(apply, rapply, domain, codomain, ctx)

    def test_recurrence_mode_avoids_extra_check_applications(self):
        ctx = make_ctx()
        matrix = ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = ctx.asarray([1.0, 2.0, 4.0])

        def run(mode):
            calls = {"apply": 0, "rapply": 0}
            A = self._matrix_free(ctx, matrix, calls)
            sc.lsqr(A, b, tol=0.0, maxiter=1, check_every=1, residual_mode=mode)
            return calls

        exact_calls = run("exact")
        recurrence_calls = run("recurrence")

        assert exact_calls["apply"] == recurrence_calls["apply"] + 1
        assert exact_calls["rapply"] == recurrence_calls["rapply"] + 2
        assert recurrence_calls == {"apply": 2, "rapply": 2}

    def test_recurrence_mode_matches_exact_solution_and_diagnostics(self):
        ctx = make_ctx()
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        A = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
        b = ctx.asarray([1.0, 2.0, 4.0])

        exact = sc.lsqr(A, b, tol=1e-12, maxiter=10, check_every=1, residual_mode="exact")
        recurrence = sc.lsqr(A, b, tol=1e-12, maxiter=10, check_every=1, residual_mode="recurrence")

        np.testing.assert_allclose(to_numpy(recurrence.x), to_numpy(exact.x), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            to_numpy(recurrence.residual_norm), to_numpy(exact.residual_norm), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            to_numpy(recurrence.normal_residual_norm), to_numpy(exact.normal_residual_norm), atol=1e-12
        )
        assert bool(to_numpy(recurrence.converged))

    def test_rejects_unknown_residual_mode(self):
        ctx = make_ctx()
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)
        with pytest.raises(ValueError, match="residual_mode"):
            sc.lsqr(A, ctx.asarray([1.0, 2.0, 3.0]), residual_mode="cheap")


# ===========================================================================
# Geometry awareness (weighted metric adjoint)
# ===========================================================================
class TestWeightedGeometry:
    def test_uses_metric_adjoint_on_weighted_spaces(self):
        ctx = make_ctx(check_level="standard")
        domain_weights = ctx.asarray([2.0, 7.0])
        codomain_weights = ctx.asarray([3.0, 5.0, 11.0])
        domain = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(domain_weights))
        codomain = sc.DenseCoordinateSpace(
            (3,), ctx, geometry=sc.WeightedInnerProduct(codomain_weights)
        )
        matrix = np.asarray([[1.0, 0.5], [-0.25, 2.0], [1.5, -1.0]])
        operator = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
        b = ctx.asarray([1.0, -2.0, 0.75])

        result = sc.lsqr(operator, b, tol=1e-12, maxiter=10, check_every=1)
        weighted_normal = matrix.T @ np.diag(to_numpy(codomain_weights))
        expected = np.linalg.solve(weighted_normal @ matrix, weighted_normal @ to_numpy(b))
        residual = operator.apply(result.x) - b

        np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(to_numpy(operator.H.apply(residual)), 0.0, atol=1e-9)
        # The metric adjoint is genuinely different from the coordinate transpose.
        y_probe = ctx.asarray([1.0, -0.5, 2.0])
        assert not np.allclose(to_numpy(operator.rapply(y_probe)), matrix.T @ to_numpy(y_probe))
        assert bool(to_numpy(result.converged))


# ===========================================================================
# JAX jit
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_lsqr_jit_compiles_with_operator_argument():
    import jax

    ctx = make_ctx("jax", jax_real_dtype())
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)

    solve = jax.jit(lambda A, b: sc.lsqr(A, b, maxiter=10).x)
    x = solve(op, ctx.asarray([1.0, 2.0, 4.0]))

    np.testing.assert_allclose(to_numpy(x), [1.33333333, 2.33333333], rtol=1e-5, atol=1e-5)
