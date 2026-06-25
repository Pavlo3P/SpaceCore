"""Tests for the ADR-019 spectral (Schatten) ``p``-norm functional."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


def _hermitian(ctx, matrix):
    """Return a symmetrized backend element and its NumPy form."""
    m = np.asarray(matrix, dtype=np.float64)
    m = 0.5 * (m + m.T)
    return ctx.asarray(m), m


_M = [[3.0, 1.0, 0.0], [1.0, 2.0, -1.0], [0.0, -1.0, 4.0]]


class TestSpectralValue:
    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
    def test_value_is_schatten_p_norm(self, numpy_ctx, p):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        A, m = _hermitian(numpy_ctx, _M)
        evals = np.linalg.eigvalsh(m)
        f = sc.SpectralLpNormFunctional(X, p)
        expected = np.sum(np.abs(evals) ** p) ** (1.0 / p)
        np.testing.assert_allclose(to_numpy(f.value(A)), expected)

    def test_schatten_2_is_frobenius_norm(self, numpy_ctx):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        A, m = _hermitian(numpy_ctx, _M)
        f = sc.SpectralLpNormFunctional(X, 2.0)
        np.testing.assert_allclose(to_numpy(f.value(A)), np.linalg.norm(m, "fro"))

    def test_nuclear_norm_is_sum_of_singular_values(self, numpy_ctx):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        A, m = _hermitian(numpy_ctx, _M)
        f = sc.NuclearNormFunctional(X)
        np.testing.assert_allclose(to_numpy(f.value(A)), np.sum(np.abs(np.linalg.eigvalsh(m))))

    def test_nuclear_norm_is_p1_spectral(self, numpy_ctx):
        X = sc.HermitianSpace(2, ctx=numpy_ctx)
        f = sc.NuclearNormFunctional(X)
        assert isinstance(f, sc.SpectralLpNormFunctional)
        assert f.p == 1.0


class TestSpectralGradient:
    def test_schatten_2_gradient_is_normalized_matrix(self, numpy_ctx):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        A, m = _hermitian(numpy_ctx, _M)
        f = sc.SpectralLpNormFunctional(X, 2.0)
        np.testing.assert_allclose(
            to_numpy(f.grad(A)), m / np.linalg.norm(m, "fro"), atol=1e-12
        )

    def test_nuclear_gradient_is_matrix_sign(self, numpy_ctx):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        A, m = _hermitian(numpy_ctx, _M)
        w, U = np.linalg.eigh(m)
        expected = U @ np.diag(np.sign(w)) @ U.T
        np.testing.assert_allclose(to_numpy(sc.NuclearNormFunctional(X).grad(A)), expected, atol=1e-10)

    def test_gradient_is_hermitian(self, numpy_ctx):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        A, _ = _hermitian(numpy_ctx, _M)
        g = to_numpy(sc.SpectralLpNormFunctional(X, 3.0).grad(A))
        np.testing.assert_allclose(g, g.T, atol=1e-12)

    @pytest.mark.parametrize("p", [1.5, 2.0, 3.0])
    def test_gradient_satisfies_directional_derivative_identity(self, numpy_ctx, p):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        A, m = _hermitian(numpy_ctx, _M)
        f = sc.SpectralLpNormFunctional(X, p)
        d = np.array([[0.2, 0.5, -0.1], [0.5, -0.3, 0.4], [-0.1, 0.4, 0.7]])
        d = 0.5 * (d + d.T)
        eps = 1e-6
        plus = float(f.value(numpy_ctx.asarray(m + eps * d)))
        minus = float(f.value(numpy_ctx.asarray(m - eps * d)))
        fd = (plus - minus) / (2.0 * eps)
        metric = float(numpy_ctx.ops.real(X.inner(f.grad(A), numpy_ctx.asarray(d))))
        np.testing.assert_allclose(fd, metric, rtol=1e-5)

    def test_gradient_at_zero_is_zero(self, numpy_ctx):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        for p in (1.0, 2.0, 3.0):
            g = to_numpy(sc.SpectralLpNormFunctional(X, p).grad(X.zeros()))
            assert np.all(np.isfinite(g))
            np.testing.assert_allclose(g, 0.0)


class TestSpectralReductionAndValidation:
    def test_reduces_to_coordinate_lp_on_elementwise_jordan(self, numpy_ctx):
        # On an elementwise Jordan space the spectrum is the coordinates, so the
        # spectral p-norm coincides with the coordinate p-norm.
        J = sc.EuclideanElementwiseJordanSpace((4,), numpy_ctx)
        v = numpy_ctx.asarray([1.0, -2.0, 0.5, 3.0])
        spectral = sc.SpectralLpNormFunctional(J, 1.5)
        coordinate = sc.LpNormFunctional(J, 1.5)
        np.testing.assert_allclose(to_numpy(spectral.value(v)), to_numpy(coordinate.value(v)))
        np.testing.assert_allclose(to_numpy(spectral.grad(v)), to_numpy(coordinate.grad(v)))

    def test_rejects_non_jordan_domain(self, numpy_ctx):
        plain = sc.DenseCoordinateSpace((3,), numpy_ctx)
        with pytest.raises(TypeError, match="Jordan"):
            sc.SpectralLpNormFunctional(plain, 2.0)

    def test_rejects_p_below_one(self, numpy_ctx):
        X = sc.HermitianSpace(2, ctx=numpy_ctx)
        with pytest.raises(ValueError):
            sc.SpectralLpNormFunctional(X, 0.5)

    def test_convert_preserves_p_and_value(self, numpy_ctx, numpy_f32_ctx):
        X = sc.HermitianSpace(3, ctx=numpy_ctx)
        A, m = _hermitian(numpy_ctx, _M)
        f = sc.SpectralLpNormFunctional(X, 2.0)
        g = f.convert(numpy_f32_ctx)
        assert g.p == 2.0 and g.ctx == numpy_f32_ctx
        a32 = numpy_f32_ctx.asarray(m.astype(np.float32))
        np.testing.assert_allclose(
            to_numpy(g.value(a32)), np.linalg.norm(m, "fro"), rtol=2e-5
        )


class TestSpectralComplex:
    def test_complex_hermitian_value_and_hermitian_gradient(self, numpy_complex_ctx):
        X = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        m = np.array([[2.0, 1.0 + 1.0j], [1.0 - 1.0j, 3.0]], dtype=np.complex128)
        A = numpy_complex_ctx.asarray(m)
        evals = np.linalg.eigvalsh(m)
        f = sc.SpectralLpNormFunctional(X, 1.0)
        np.testing.assert_allclose(to_numpy(f.value(A)), np.sum(np.abs(evals)))
        g = to_numpy(f.grad(A))
        np.testing.assert_allclose(g, g.conj().T, atol=1e-12)
