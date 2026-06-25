"""Tests for :class:`spacecore.TreeSpectralDecomposition`.

Checklist item 19:

* Construction from per-leaf eigvals + frames tuples.
* Frozen dataclass: ``eigvals`` and ``frames`` are immutable.
* Leafwise access by index.
* ``tree_flatten`` / ``tree_unflatten`` JAX pytree round-trip.
* Produced by :meth:`TreeSpace.spectral_decompose` on Jordan-leaf trees.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype, to_numpy


# ===========================================================================
# Construction and access
# ===========================================================================
class TestConstruction:
    def test_constructed_from_eigvals_and_frames(self, numpy_ctx):
        eigvals = (numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0]))
        frames = (None, None)
        decomp = sc.TreeSpectralDecomposition(eigvals=eigvals, frames=frames)
        assert decomp.eigvals == eigvals
        assert decomp.frames == frames

    def test_is_frozen_dataclass(self, numpy_ctx):
        decomp = sc.TreeSpectralDecomposition(eigvals=(), frames=())
        with pytest.raises((AttributeError, Exception)):
            decomp.eigvals = ()  # type: ignore[misc]

    def test_leaves_accessible_by_index(self, numpy_ctx):
        ev0 = numpy_ctx.asarray([1.0, 2.0])
        ev1 = numpy_ctx.asarray([3.0])
        decomp = sc.TreeSpectralDecomposition(eigvals=(ev0, ev1), frames=(None, None))
        np.testing.assert_allclose(decomp.eigvals[0], ev0)
        np.testing.assert_allclose(decomp.eigvals[1], ev1)


# ===========================================================================
# Produced by TreeSpace.spectral_decompose
# ===========================================================================
class TestProducedByTreeSpace:
    def test_spectral_decompose_returns_decomposition(self, numpy_ctx):
        # Use Elementwise leaves where spectrum(x) = x.
        leaves = (
            sc.ElementwiseJordanSpace((2,), numpy_ctx),
            sc.ElementwiseJordanSpace((3,), numpy_ctx),
        )
        product = sc.TreeSpace.from_leaf_spaces(leaves, numpy_ctx)
        x = (numpy_ctx.asarray([1.0, -2.0]), numpy_ctx.asarray([3.0, 4.0, -1.0]))
        decomp = product.spectral_decompose(x)
        assert isinstance(decomp, sc.TreeSpectralDecomposition)
        np.testing.assert_allclose(to_numpy(decomp.eigvals[0]), [1.0, -2.0])
        np.testing.assert_allclose(to_numpy(decomp.eigvals[1]), [3.0, 4.0, -1.0])


# ===========================================================================
# JAX pytree round-trip
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
class TestJaxPytree:
    def test_round_trip(self):
        import jax
        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        decomp = sc.TreeSpectralDecomposition(
            eigvals=(ctx.asarray([1.0, 2.0]), ctx.asarray([3.0])),
            frames=(None, None),
        )
        leaves, treedef = jax.tree_util.tree_flatten(decomp)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, sc.TreeSpectralDecomposition)
        for actual, expected in zip(rebuilt.eigvals, decomp.eigvals):
            np.testing.assert_allclose(to_numpy(actual), to_numpy(expected))
