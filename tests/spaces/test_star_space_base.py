"""Tests for :class:`spacecore.StarSpace` — the conjugation/transpose base.

Checklist item 8:

* ``star(star(x)) == x`` involution law.
* ``star(α·x) == conj(α) · star(x)`` conjugate linearity.
* Tested on DenseVectorSpace (per-element conj on complex) and HermitianSpace
  (conj-transpose) as representative concrete StarSpace subclasses.
"""
from __future__ import annotations

import numpy as np

import spacecore as sc

from tests._helpers import to_numpy


# ===========================================================================
# Involution: star(star(x)) = x
# ===========================================================================
class TestInvolution:
    def test_involution_on_real_dense_vector(self, numpy_ctx):
        space = sc.DenseVectorSpace((4,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, -2.0, 3.0, 4.5])
        np.testing.assert_allclose(space.star(space.star(x)), x)

    def test_involution_on_complex_dense_vector(self, numpy_complex_ctx):
        space = sc.DenseVectorSpace((4,), numpy_complex_ctx)
        x = numpy_complex_ctx.asarray([1 + 2j, -3 + 0.5j, 4 + 0j, 0.25 - 1j])
        np.testing.assert_allclose(space.star(space.star(x)), x)

    def test_involution_on_hermitian_space(self, numpy_complex_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        H = numpy_complex_ctx.asarray([[1 + 0j, 2 - 1j], [2 + 1j, 3 + 0j]])
        np.testing.assert_allclose(space.star(space.star(H)), H)


# ===========================================================================
# Conjugate linearity: star(α·x) = conj(α)·star(x)
# ===========================================================================
class TestConjugateLinearity:
    def test_conjugate_linear_on_real(self, numpy_ctx):
        """On a real space, ``conj(α) = α``, so star is linear."""
        space = sc.DenseVectorSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, -2.0, 3.0])
        alpha = 2.5
        np.testing.assert_allclose(
            space.star(alpha * x),
            np.conj(alpha) * to_numpy(space.star(x)),
        )

    def test_conjugate_linear_on_complex(self, numpy_complex_ctx):
        space = sc.DenseVectorSpace((3,), numpy_complex_ctx)
        x = numpy_complex_ctx.asarray([1 + 2j, -3 + 0.5j, 4 - 1j])
        alpha = 2.0 - 3.0j
        np.testing.assert_allclose(
            space.star(alpha * x),
            np.conj(alpha) * to_numpy(space.star(x)),
        )


# ===========================================================================
# Concrete star implementations
# ===========================================================================
class TestConcreteStar:
    def test_dense_vector_real_star_is_identity(self, numpy_ctx):
        space = sc.DenseVectorSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, -2.0, 3.0])
        np.testing.assert_allclose(space.star(x), x)

    def test_dense_vector_complex_star_is_conj(self, numpy_complex_ctx):
        space = sc.DenseVectorSpace((3,), numpy_complex_ctx)
        x = numpy_complex_ctx.asarray([1 + 2j, -3 + 0.5j, 4 + 0j])
        np.testing.assert_allclose(space.star(x), np.conj(to_numpy(x)))

    def test_hermitian_star_is_conj_transpose(self, numpy_complex_ctx):
        space = sc.HermitianSpace(2, ctx=numpy_complex_ctx)
        A = numpy_complex_ctx.asarray([[1 + 0j, 2 - 1j], [2 + 1j, 3 + 0j]])
        np.testing.assert_allclose(space.star(A), np.conj(to_numpy(A).T))
