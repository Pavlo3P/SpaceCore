"""Tests for :class:`spacecore.HermitianSpace`.

Checklist item 15:

* Construction: ``n`` property, ``shape = (n, n)``, ``size = n*n``.
* Hermitian membership check rejects non-Hermitian, accepts Hermitian.
* ``symmetrize`` produces a strictly Hermitian element from any input.
* ``star = conj-transpose``.
* ``jordan(x, y) = (x·y + y·x) / 2``.
* ``spectrum`` / ``spectral_decompose`` via ``eigh``, ``from_spectrum`` reconstructs.
* ``psd_proj`` projects onto the PSD cone (eigvals ≥ 0).
* ``unflatten`` returns a square Hermitian matrix.
* ``eig_to_dense`` reconstructs from eigendecomposition.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_complex_dtype, to_numpy


# ===========================================================================
# Construction
# ===========================================================================
class TestConstruction:
    def test_n_property_and_shape(self, numpy_complex_ctx):
        H = sc.HermitianSpace(3, ctx=numpy_complex_ctx)
        assert H.n == 3
        assert H.shape == (3, 3)
        assert H.size == 9

    def test_is_jordan_star_inner_product_space(self, numpy_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_ctx)
        assert isinstance(H, sc.HermitianSpace)
        assert isinstance(H, sc.StarSpace)
        assert isinstance(H, sc.JordanAlgebraSpace)
        assert isinstance(H, sc.InnerProductSpace)
        assert isinstance(H, sc.EuclideanJordanAlgebraSpace)


# ===========================================================================
# Membership + symmetrize
# ===========================================================================
class TestMembershipAndSymmetrize:
    def test_hermitian_input_passes_membership(self, numpy_complex_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        good = numpy_complex_ctx.asarray([[1 + 0j, 2 - 1j], [2 + 1j, 3 + 0j]])
        H.check_member(good)

    def test_non_hermitian_input_rejected(self, numpy_complex_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        bad = numpy_complex_ctx.asarray([[1 + 0j, 2 + 1j], [2 + 1j, 3 + 0j]])
        with pytest.raises(Exception, match="(?i)hermitian"):
            H.check_member(bad)

    def test_symmetrize_makes_membership_pass(self, numpy_complex_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        bad = numpy_complex_ctx.asarray([[1 + 0j, 2 + 1j], [2 + 1j, 3 + 0j]])
        sym = H.symmetrize(bad)
        H.check_member(sym)

    def test_symmetrize_on_already_hermitian_is_identity(self, numpy_complex_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        good = numpy_complex_ctx.asarray([[1 + 0j, 2 - 1j], [2 + 1j, 3 + 0j]])
        np.testing.assert_allclose(H.symmetrize(good), good)


# ===========================================================================
# star: conj-transpose
# ===========================================================================
class TestStar:
    def test_star_is_conj_transpose(self, numpy_complex_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        A = numpy_complex_ctx.asarray([[1 + 0j, 2 - 1j], [2 + 1j, 3 + 0j]])
        np.testing.assert_allclose(H.star(A), np.conj(to_numpy(A).T))


# ===========================================================================
# jordan = (x·y + y·x) / 2
# ===========================================================================
class TestJordan:
    def test_jordan_is_anticommutator_half(self, numpy_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_ctx)
        A = H.symmetrize(numpy_ctx.asarray([[1.0, 0.5], [0.5, 2.0]]))
        B = H.symmetrize(numpy_ctx.asarray([[3.0, -1.0], [-1.0, 4.0]]))
        expected = (to_numpy(A) @ to_numpy(B) + to_numpy(B) @ to_numpy(A)) / 2.0
        np.testing.assert_allclose(H.jordan(A, B), expected)


# ===========================================================================
# Spectrum / spectral_decompose / from_spectrum / eig_to_dense
# ===========================================================================
class TestSpectrum:
    def test_spectral_decompose_eigenvalues_match_eigvalsh(self, numpy_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_ctx)
        x = numpy_ctx.asarray([[1.0, 0.0], [0.0, -2.0]])
        evals, _ = H.spectral_decompose(x)
        np.testing.assert_allclose(np.sort(to_numpy(evals)), [-2.0, 1.0])

    def test_spectral_decompose_reconstruction(self):
        """A·v_i = λ_i·v_i identity. Run at check_level=none to avoid
        the strict Hermitian membership gate on the reconstructed matrix.
        """
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        H = sc.HermitianSpace(3, ctx=ctx)
        rng = np.random.default_rng(0)
        A = H.symmetrize(ctx.asarray(rng.standard_normal((3, 3))))
        evals, evecs = H.spectral_decompose(A)
        recon = H.from_spectrum(evals, evecs)
        np.testing.assert_allclose(recon, A, atol=1e-10)

    def test_eig_to_dense_reconstructs_via_independent_formula(self):
        """``eig_to_dense(λ, V)`` equals the independent ``V diag(λ) Vᴴ``.

        Compared against a NumPy reconstruction rather than ``from_spectrum``
        (which merely delegates to ``eig_to_dense``, so that comparison was a
        tautology) — this genuinely verifies the reconstruction einsum and
        eigenvector handling.
        """
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        H = sc.HermitianSpace(3, ctx=ctx)
        rng = np.random.default_rng(1)
        A = H.symmetrize(ctx.asarray(rng.standard_normal((3, 3))))
        evals, evecs = H.spectral_decompose(A)
        via_eig = to_numpy(H.eig_to_dense(evals, evecs))
        V = to_numpy(evecs)
        independent = V @ np.diag(to_numpy(evals)) @ V.conj().T
        np.testing.assert_allclose(via_eig, independent, atol=1e-10)
        # And it reconstructs the original Hermitian matrix.
        np.testing.assert_allclose(via_eig, to_numpy(A), atol=1e-10)


# ===========================================================================
# psd_proj
# ===========================================================================
class TestPsdProj:
    def test_psd_proj_yields_nonneg_spectrum(self, numpy_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_ctx)
        x = H.symmetrize(numpy_ctx.asarray([[1.0, 0.0], [0.0, -2.0]]))
        y = H.psd_proj(x)
        evals = np.linalg.eigvalsh(to_numpy(y))
        assert evals.min() >= -1e-8

    def test_psd_proj_is_idempotent_on_psd_input(self):
        """psd_proj is a projector — applying it twice equals once."""
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        H = sc.HermitianSpace(2, ctx=ctx)
        # SPD input
        M = ctx.asarray([[2.0, 0.5], [0.5, 3.0]])
        A = H.symmetrize(M)
        once = H.psd_proj(A)
        twice = H.psd_proj(once)
        np.testing.assert_allclose(to_numpy(twice), to_numpy(once), atol=1e-10)


# ===========================================================================
# Unflatten
# ===========================================================================
class TestUnflatten:
    def test_unflatten_returns_square_matrix(self, numpy_ctx):
        H = sc.HermitianSpace(3, ctx=numpy_ctx)
        v = numpy_ctx.asarray(np.arange(9.0))
        M = H.unflatten(v)
        assert to_numpy(M).shape == (3, 3)


# ===========================================================================
# Conversion
# ===========================================================================
class TestConvert:
    def test_convert_uses_target_dtype(self, numpy_complex_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        dst = sc.Context(sc.NumpyOps(), dtype=np.complex64)
        K = H.convert(dst)
        assert K.dtype == np.dtype(np.complex64)
        assert K.shape == (2, 2)

    def test_convert_preserves_euclidean_geometry(self, numpy_ctx, numpy_f32_ctx):
        H = sc.HermitianSpace(2, ctx=numpy_ctx)
        K = H.convert(numpy_f32_ctx)
        assert K.is_euclidean is True
        assert type(K.geometry) is sc.EuclideanInnerProduct

    @pytest.mark.skipif(not has_jax(), reason="jax is not installed")
    def test_convert_to_jax(self):
        dt = jax_complex_dtype()
        H = sc.HermitianSpace(2, ctx=sc.Context(sc.NumpyOps(), dtype=dt))
        K = H.convert(sc.Context(sc.JaxOps(), dtype=dt))
        assert K.ctx.ops.family == "jax"
