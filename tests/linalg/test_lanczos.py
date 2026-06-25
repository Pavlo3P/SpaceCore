"""Tests for :func:`spacecore.lanczos_smallest` and its Krylov basis builder.

Checklist section 8, ``lanczos_smallest`` + internal ``_LanczosBasisResult``:

* Approximates the smallest eigenpair (real, complex Hermitian, large-scale).
* Geometry awareness: weighted / custom inner products give the metric pair.
* Zero initial vector falls back to a deterministic coordinate vector.
* Rejects invalid ``max_iter``, structurally non-Hermitian, and rectangular
  operators.
* ``krylov_dim`` reflects true breakdown; the basis is orthonormal and the
  inactive-tail sentinel masks ghost iterations.
* JAX ``jit`` round-trip preserves correctness.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from spacecore.linalg._lanczos import _lanczos_basis_and_tridiag
from tests._helpers import has_jax, jax_real_dtype, to_numpy
from tests.linalg._helpers import backend_params, make_ctx, numpy_jax_params


# ===========================================================================
# Smallest eigenpair
# ===========================================================================
class TestSmallestEigenpair:
    @pytest.mark.parametrize("backend_name,dtype", backend_params())
    def test_approximates_smallest_eigenpair(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

        result = sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=2, tol=1e-8)

        np.testing.assert_allclose(to_numpy(result.eigenvalue), 2.0, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.abs(to_numpy(result.eigenvector)), [1.0, 0.0], rtol=1e-5, atol=1e-5)
        assert bool(to_numpy(result.converged))
        np.testing.assert_allclose(to_numpy(result.residual_norm), 0.0, atol=1e-5)
        assert int(to_numpy(result.krylov_dim)) == 2

    def test_returns_result_object(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
        result = sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=2, tol=1e-8)
        assert isinstance(result, sc.LanczosResult)
        np.testing.assert_allclose(result.eigenvalue, 2.0)

    def test_handles_complex_hermitian_operator(self):
        ctx = make_ctx(dtype=np.complex128)
        space = sc.DenseCoordinateSpace((2,), ctx)
        matrix = np.array([[2.0, 1.0 + 2.0j], [1.0 - 2.0j, 5.0]], dtype=np.complex128)
        op = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)

        result = sc.lanczos_smallest(op, ctx.asarray([1.0 + 0.0j, 1.0j]), max_iter=2, tol=1e-10)

        expected = np.linalg.eigvalsh(matrix)[0]
        np.testing.assert_allclose(to_numpy(result.eigenvalue), expected, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(
            to_numpy(op.apply(result.eigenvector)),
            to_numpy(result.eigenvalue) * to_numpy(result.eigenvector),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_handles_eigenvalues_larger_than_1e10(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray(np.diag([2.0e12, 3.0e12])), space, space, ctx)

        result = sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=4, tol=1e-8)

        np.testing.assert_allclose(to_numpy(result.eigenvalue), 2.0e12, rtol=1e-6)
        np.testing.assert_allclose(np.abs(to_numpy(result.eigenvector)), [1.0, 0.0], atol=1e-5)

    def test_matrix_free_reference_has_normalized_ritz_vector(self):
        ctx = make_ctx(check_level="standard")
        space = sc.DenseCoordinateSpace((3,), ctx)
        matrix = np.asarray([[4.0, 1.0, 0.5], [1.0, 3.0, -0.25], [0.5, -0.25, 2.0]])
        backend = ctx.asarray(matrix)
        operator = sc.MatrixFreeLinOp(lambda x: backend @ x, lambda y: backend.T @ y, space, space, ctx)

        result = sc.lanczos_smallest(
            operator, ctx.asarray([1.0, -0.5, 0.75]), max_iter=3, tol=1e-12, check_every=1
        )
        expected = np.linalg.eigvalsh(matrix)[0]
        residual = operator.apply(result.eigenvector) - result.eigenvalue * result.eigenvector

        np.testing.assert_allclose(to_numpy(result.eigenvalue), expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(to_numpy(space.norm(result.eigenvector)), 1.0, atol=1e-12)
        np.testing.assert_allclose(to_numpy(space.norm(residual)), 0.0, atol=1e-10)
        assert tuple(result.eigenvector.shape) == space.shape
        assert int(to_numpy(result.krylov_dim)) == 3


# ===========================================================================
# Geometry & zero-vector fallback
# ===========================================================================
class TestGeometryAndFallback:
    def test_uses_domain_geometry_for_weighted_inner_product(self):
        ctx = make_ctx()

        class WeightedVectorSpace(sc.DenseCoordinateSpace):
            def __init__(self, weights, ctx):
                weights = ctx.asarray(weights)
                super().__init__(tuple(weights.shape), ctx)
                self.weights = weights

            def inner(self, x, y):
                if self._enable_checks:
                    self._check_member(x)
                    self._check_member(y)
                return self.ops.vdot(x, self.weights * y)

            def _convert(self, new_ctx):
                return WeightedVectorSpace(new_ctx.asarray(self.weights), new_ctx)

        space = WeightedVectorSpace([1.0, 4.0], ctx)
        assert type(space) is not sc.DenseCoordinateSpace
        matrix = ctx.asarray([[2.0, 1.0], [0.25, 0.75]])
        op = sc.MatrixFreeLinOp(lambda x: matrix @ x, lambda x: matrix @ x, space, space, ctx)

        result = sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=2, tol=1e-12)

        expected = np.min(np.linalg.eigvals(to_numpy(matrix)).real)
        np.testing.assert_allclose(to_numpy(result.eigenvalue), expected, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(
            to_numpy(op.apply(result.eigenvector)),
            to_numpy(result.eigenvalue) * to_numpy(result.eigenvector),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_uses_e0_for_zero_initial_vector(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

        result = sc.lanczos_smallest(op, ctx.asarray([0.0, 0.0]), max_iter=2, tol=1e-8)

        np.testing.assert_allclose(result.eigenvalue, 2.0, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(result.eigenvector, [1.0, 0.0], rtol=1e-6, atol=1e-6)


# ===========================================================================
# Rejections
# ===========================================================================
class TestRejections:
    def test_rejects_invalid_max_iter(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((1,), ctx)
        op = sc.IdentityLinOp(space, ctx)
        with pytest.raises(ValueError, match="max_iter"):
            sc.lanczos_smallest(op, ctx.asarray([1.0]), max_iter=0)

    def test_rejects_structurally_non_hermitian_operator(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, 3.0]]), space, space, ctx)
        with pytest.raises(ValueError, match="Hermitian"):
            sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=2)

    def test_rejects_rectangular_operator(self):
        ctx = make_ctx()
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)
        with pytest.raises(ValueError, match="square LinOp"):
            sc.lanczos_smallest(A, ctx.asarray([1.0, 2.0]))


# ===========================================================================
# Krylov dimension / breakdown
# ===========================================================================
class TestKrylovDimension:
    @pytest.mark.parametrize("backend_name,dtype", numpy_jax_params())
    def test_uses_true_krylov_dim_after_delayed_breakdown(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        space = sc.DenseCoordinateSpace((3,), ctx)
        op = sc.DiagonalLinOp(ctx.asarray([1.5, 2.0, 3.0]), space, ctx)

        result = sc.lanczos_smallest(
            op, ctx.asarray([1.0, 1.0, 1.0]), max_iter=20, tol=1e-5, check_every=10
        )

        assert int(to_numpy(result.krylov_dim)) <= 3
        np.testing.assert_allclose(to_numpy(result.eigenvalue), 1.5, rtol=1e-5, atol=1e-5)


# ===========================================================================
# Internal: _lanczos_basis_and_tridiag / _LanczosBasisResult
# ===========================================================================
class TestBasisBuilder:
    @pytest.mark.parametrize("backend_name,dtype", numpy_jax_params())
    def test_sentinel_masks_ghost_iterations_after_breakdown(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        space = sc.DenseCoordinateSpace((3,), ctx)
        op = sc.DiagonalLinOp(ctx.asarray([1.5, 2.0, 3.0]), space, ctx)

        basis = _lanczos_basis_and_tridiag(
            op,
            ctx.asarray([1.0, 1.0, 1.0]),
            max_iter=20,
            tol=1e-5,
            real_dtype=ctx.ops.real_dtype(ctx.dtype),
            check_every=10,
        )

        krylov_dim = int(to_numpy(basis.krylov_dim))
        T_diag = np.diag(to_numpy(basis.T))
        assert krylov_dim == 3
        # The inactive tail is filled with a large sentinel, not real Ritz values.
        assert np.all(T_diag[krylov_dim:] > 3.0)
        np.testing.assert_allclose(np.linalg.eigvalsh(to_numpy(basis.T))[0], 1.5, rtol=1e-5, atol=1e-5)

    def test_krylov_basis_is_orthonormal_before_breakdown(self):
        ctx = make_ctx(check_level="standard")
        space = sc.DenseCoordinateSpace((4,), ctx)
        matrix = ctx.asarray(
            [[5.0, 1.0, 0.0, 0.25], [1.0, 4.0, -0.5, 0.0],
             [0.0, -0.5, 3.0, 0.75], [0.25, 0.0, 0.75, 2.0]]
        )
        operator = sc.DenseLinOp(matrix, space, space, ctx)

        basis = _lanczos_basis_and_tridiag(
            operator,
            ctx.asarray([1.0, -0.5, 0.75, 1.5]),
            max_iter=4,
            tol=1e-12,
            real_dtype=ctx.ops.real_dtype(ctx.dtype),
            check_every=1,
        )
        krylov_dim = int(to_numpy(basis.krylov_dim))
        vectors = to_numpy(basis.V[:krylov_dim])

        np.testing.assert_allclose(vectors @ vectors.conj().T, np.eye(krylov_dim), atol=1e-11)


# ===========================================================================
# JAX jit
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_lanczos_smallest_jit_compiles_with_operator_argument():
    import jax

    ctx = make_ctx("jax", jax_real_dtype())
    space = sc.DenseCoordinateSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    def run(A, initial):
        result = sc.lanczos_smallest(A, initial, max_iter=2, tol=1e-8)
        return result.eigenvalue, result.eigenvector

    eigenvalue, eigenvector = jax.jit(run)(op, ctx.asarray([1.0, 1.0]))

    np.testing.assert_allclose(to_numpy(eigenvalue), 2.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.abs(to_numpy(eigenvector)), [1.0, 0.0], rtol=1e-5, atol=1e-5)
