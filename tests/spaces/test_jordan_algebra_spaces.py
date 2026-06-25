"""Tests for :class:`spacecore.JordanAlgebraSpace` and
:class:`spacecore.EuclideanJordanAlgebraSpace` — the Jordan-product bases.

Checklist items 9 and 10:

* :class:`JordanAlgebraSpace` — ``jordan(x, y) = jordan(y, x)`` commutativity,
  ``spectrum`` / ``spectral_decompose`` / ``from_spectrum`` round-trip,
  ``spectral_apply(x, f)`` matches ``f`` on the spectrum.
* :class:`EuclideanJordanAlgebraSpace` — trace-form inner product
  ``⟨x, y⟩ = trace(x ∘ y)``, Jordan associativity
  ``⟨x ∘ y, z⟩ = ⟨y, x ∘ z⟩``.

Gap-fill (per audit): trace-form inner identity on Hermitian + Elementwise.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

import spacecore as sc

from tests._helpers import to_numpy


# ===========================================================================
# Jordan commutativity
# ===========================================================================
class TestCommutativity:
    def test_elementwise_jordan_commutativity_real(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        y = numpy_ctx.asarray([0.5, -3.0, 4.0])
        np.testing.assert_allclose(space.jordan(x, y), space.jordan(y, x))

    def test_hermitian_jordan_commutativity(self, numpy_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        A = space.symmetrize(numpy_ctx.asarray([[1.0, 0.25], [0.25, 2.0]]))
        B = space.symmetrize(numpy_ctx.asarray([[0.5, -0.75], [-0.75, 3.0]]))
        np.testing.assert_allclose(space.jordan(A, B), space.jordan(B, A))


# ===========================================================================
# Spectrum / spectral_decompose / from_spectrum round-trip
# ===========================================================================
class TestSpectralRoundTrip:
    def test_elementwise_spectrum_equals_x(self, numpy_ctx):
        """For ElementwiseJordanSpace, spectrum(x) = x."""
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([2.0, -1.0, 0.5])
        np.testing.assert_allclose(space.spectrum(x), x)

    def test_elementwise_spectral_decompose_round_trip(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([2.0, -1.0, 0.5])
        eigvals, frame = space.spectral_decompose(x)
        recon = space.from_spectrum(eigvals, frame)
        np.testing.assert_allclose(recon, x)

    def test_hermitian_spectral_decompose_round_trip(self):
        """Reconstruction from spectrum is bit-inexact; run at check_level=none."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        space = sc.HermitianSpace(3, ctx=ctx)
        rng = np.random.default_rng(0)
        M = ctx.asarray(rng.standard_normal((3, 3)))
        H = space.symmetrize(M)
        eigvals, frame = space.spectral_decompose(H)
        recon = space.from_spectrum(eigvals, frame)
        np.testing.assert_allclose(recon, H, atol=1e-10)


# ===========================================================================
# spectral_apply: matches f on the spectrum
# ===========================================================================
class TestSpectralApply:
    def test_spectral_apply_squares_on_elementwise(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        out = space.spectral_apply(x, lambda t: t * t)
        np.testing.assert_allclose(out, np.asarray(to_numpy(x)) ** 2)

    def test_spectral_apply_exp_on_hermitian(self):
        """``spectral_apply(H, exp)`` matches ``scipy.linalg.expm(H)``.

        The reference is an INDEPENDENT ground truth: ``scipy.linalg.expm`` does
        not share ``spectral_apply``'s ``eigh -> exp -> eig_to_dense``
        reconstruction path, so a bug in that reconstruction (eigenvector
        handling, the ``eig_to_dense`` einsum, or frame ordering) is genuinely
        caught here rather than cancelling out against a self-referential
        ``from_spectrum`` expected value. Reconstruction is not bit-exact
        symmetric, so the strict Hermitian membership gate would refuse it; run
        with check_level=none.
        """
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        space = sc.HermitianSpace(2, ctx=ctx)
        H = space.symmetrize(ctx.asarray([[2.0, 0.5], [0.5, 1.0]]))
        applied = space.spectral_apply(H, lambda t: ctx.ops.exp(t))
        expected = scipy.linalg.expm(to_numpy(H))
        np.testing.assert_allclose(to_numpy(applied), expected, atol=1e-10)


# ===========================================================================
# EuclideanJordanAlgebraSpace: Jordan associativity ⟨x ∘ y, z⟩ = ⟨y, x ∘ z⟩
# ===========================================================================
class TestJordanAssociativity:
    def test_associativity_on_elementwise_real(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        assert isinstance(space, sc.EuclideanJordanAlgebraSpace)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        y = numpy_ctx.asarray([0.5, -3.0, 4.0])
        z = numpy_ctx.asarray([2.0, 1.0, 0.25])
        lhs = space.inner(space.jordan(x, y), z)
        rhs = space.inner(y, space.jordan(x, z))
        np.testing.assert_allclose(lhs, rhs)

    def test_associativity_on_hermitian(self, numpy_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        A = space.symmetrize(numpy_ctx.asarray([[1.0, 0.25], [0.25, 2.0]]))
        B = space.symmetrize(numpy_ctx.asarray([[0.5, -0.75], [-0.75, 3.0]]))
        C = space.symmetrize(numpy_ctx.asarray([[2.0, 1.0], [1.0, -1.0]]))
        np.testing.assert_allclose(
            space.inner(space.jordan(A, B), C),
            space.inner(B, space.jordan(A, C)),
        )


# ===========================================================================
# EuclideanJordanAlgebraSpace: trace-form inner product
# Gap-fill: explicit ⟨x, y⟩ = sum/trace(x ∘ y) check.
# ===========================================================================
class TestTraceFormInner:
    def test_trace_form_inner_on_real_elementwise(self, numpy_ctx):
        """For ElementwiseJordanSpace, ``⟨x, y⟩ = Σ x_i · y_i = vdot(x, y)``."""
        space = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, -1.0])
        y = numpy_ctx.asarray([0.5, -3.0, 4.0])
        # Trace form: inner = sum(x ∘ y) where x ∘ y = x*y elementwise.
        expected = float(np.sum(to_numpy(x) * to_numpy(y)))
        np.testing.assert_allclose(float(to_numpy(space.inner(x, y))), expected)

    def test_trace_form_inner_on_hermitian(self, numpy_ctx):
        """For HermitianSpace, ``⟨A, B⟩ = trace(A · B)`` for symmetric A, B."""
        space = sc.HermitianSpace(2, ctx=numpy_ctx)
        A = space.symmetrize(numpy_ctx.asarray([[1.0, 0.5], [0.5, 2.0]]))
        B = space.symmetrize(numpy_ctx.asarray([[3.0, -1.0], [-1.0, 4.0]]))
        expected = float(np.trace(to_numpy(A) @ to_numpy(B)))
        np.testing.assert_allclose(float(to_numpy(space.inner(A, B))), expected)


# ===========================================================================
# Geometry compatibility — real vs complex vs weighted Elementwise
# (moved from test_space_hierarchy.py)
# ===========================================================================
class TestGeometryCompatibility:
    def test_real_elementwise_is_euclidean_jordan(self, numpy_ctx):
        space = sc.ElementwiseJordanSpace((2,), numpy_ctx)
        assert isinstance(space, sc.EuclideanJordanAlgebraSpace)

    def test_complex_elementwise_is_not_euclidean_jordan(self, numpy_complex_ctx):
        space = sc.ElementwiseJordanSpace((2,), numpy_complex_ctx)
        assert not isinstance(space, sc.EuclideanJordanAlgebraSpace)

    def test_weighted_elementwise_is_not_euclidean_jordan(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        weighted = sc.ElementwiseJordanSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        assert isinstance(weighted, sc.JordanAlgebraSpace)
        assert not isinstance(weighted, sc.EuclideanJordanAlgebraSpace)

    def test_weighted_dense_vector_is_not_jordan(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        weighted = sc.DenseVectorSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        assert not isinstance(weighted, sc.JordanAlgebraSpace)

    def test_dense_coordinate_is_not_jordan_even_weighted(self, numpy_ctx):
        weights = numpy_ctx.asarray([2.0, 3.0])
        weighted = sc.DenseCoordinateSpace(
            (2,), numpy_ctx, geometry=sc.WeightedInnerProduct(weights),
        )
        assert not isinstance(weighted, sc.JordanAlgebraSpace)
