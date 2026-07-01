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
import optree
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

    def test_round_trip_preserves_treedef(self):
        import jax

        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        leaves = (sc.ElementwiseJordanSpace((2,), ctx), sc.ElementwiseJordanSpace((1,), ctx))
        space = sc.TreeSpace.from_leaf_spaces(leaves, ctx)
        x = space.element((ctx.asarray([1.0, 2.0]), ctx.asarray([3.0])))
        decomp = space.spectral_decompose(x)
        flat, treedef = jax.tree_util.tree_flatten(decomp)
        rebuilt = jax.tree_util.tree_unflatten(treedef, flat)
        # treedef is static aux, so it survives the JAX pytree round-trip.
        assert rebuilt.treedef == space.treedef


# ===========================================================================
# W3 — structure-preserving spectrum + treedef-shaped decomposition
# ===========================================================================
def _flat_jordan_tree(ctx):
    leaves = (sc.ElementwiseJordanSpace((2,), ctx), sc.ElementwiseJordanSpace((3,), ctx))
    space = sc.TreeSpace.from_leaf_spaces(leaves, ctx)
    x = space.element((ctx.asarray([1.0, -2.0]), ctx.asarray([3.0, 4.0, -5.0])))
    return space, x


class TestStructuredSpectrum:
    def test_structured_spectrum_matches_treedef_and_recovers_leaves(self, numpy_ctx):
        space, x = _flat_jordan_tree(numpy_ctx)
        structured = space.spectrum(x, structured=True)
        assert optree.tree_structure(structured) == space.treedef
        parts = space.flatten_tree(structured)
        np.testing.assert_allclose(to_numpy(parts[0]), [1.0, -2.0])
        np.testing.assert_allclose(to_numpy(parts[1]), [3.0, 4.0, -5.0])

    def test_flat_spectrum_unchanged_and_equals_structured_concat(self, numpy_ctx):
        space, x = _flat_jordan_tree(numpy_ctx)
        flat = to_numpy(space.spectrum(x))  # default: flat concatenation
        assert flat.shape == (5,)
        parts = [to_numpy(p) for p in space.flatten_tree(space.spectrum(x, structured=True))]
        np.testing.assert_allclose(flat, np.concatenate(parts))


class TestTreedefShapedDecomposition:
    def test_decomposition_carries_treedef_and_to_tree(self, numpy_ctx):
        space, x = _flat_jordan_tree(numpy_ctx)
        decomp = space.spectral_decompose(x)
        assert decomp.treedef == space.treedef
        tree = decomp.to_tree()
        assert optree.tree_structure(tree) == space.treedef
        parts = space.flatten_tree(tree)
        np.testing.assert_allclose(to_numpy(parts[0]), [1.0, -2.0])
        np.testing.assert_allclose(to_numpy(parts[1]), [3.0, 4.0, -5.0])

    def test_to_tree_without_treedef_raises(self, numpy_ctx):
        decomp = sc.TreeSpectralDecomposition(eigvals=(numpy_ctx.asarray([1.0]),), frames=(None,))
        with pytest.raises(ValueError, match="treedef"):
            decomp.to_tree()

    def test_two_arg_construction_is_backward_compatible(self, numpy_ctx):
        # The added treedef field defaults to None; 2-arg construction still works.
        decomp = sc.TreeSpectralDecomposition((numpy_ctx.asarray([1.0]),), (None,))
        assert decomp.treedef is None


class TestNestedTree:
    def _nested(self, ctx):
        leaves = (
            sc.ElementwiseJordanSpace((2,), ctx),
            sc.ElementwiseJordanSpace((1,), ctx),
            sc.ElementwiseJordanSpace((2,), ctx),
        )
        space = sc.TreeSpace.from_template((0, [0, (0,)]), leaves, ctx=ctx)
        x = space.unflatten_tree(
            (ctx.asarray([1.0, -2.0]), ctx.asarray([3.0]), ctx.asarray([5.0, -6.0]))
        )
        return space, x

    def test_nested_decompose_round_trip(self, numpy_ctx):
        space, x = self._nested(numpy_ctx)
        decomp = space.spectral_decompose(x)
        assert decomp.treedef == space.treedef
        rebuilt = space.from_spectrum(decomp)
        np.testing.assert_allclose(to_numpy(space.flatten(rebuilt)), to_numpy(space.flatten(x)))

    def test_nested_structured_results_match_deep_treedef(self, numpy_ctx):
        space, x = self._nested(numpy_ctx)
        assert optree.tree_structure(space.spectrum(x, structured=True)) == space.treedef
        assert optree.tree_structure(space.spectral_decompose(x).to_tree()) == space.treedef

    def test_nested_hermitian_leaf_round_trip(self, numpy_ctx):
        # A Hermitian leaf exercises non-None frames through the nested round-trip.
        leaves = (sc.HermitianSpace(2, ctx=numpy_ctx), sc.ElementwiseJordanSpace((2,), numpy_ctx))
        space = sc.TreeSpace.from_template((0, (0,)), leaves, ctx=numpy_ctx)
        x = space.unflatten_tree(
            (numpy_ctx.asarray([[2.0, 0.3], [0.3, 1.0]]), numpy_ctx.asarray([3.0, -4.0]))
        )
        decomp = space.spectral_decompose(x)
        assert decomp.frames[0] is not None  # Hermitian leaf carries an eigenvector frame
        rebuilt = space.from_spectrum(decomp)
        np.testing.assert_allclose(to_numpy(space.flatten(rebuilt)), to_numpy(space.flatten(x)))


class TestFlatSpectralRegression:
    """W3 must not disturb SpectralLpNormFunctional on flat (non-tree) Jordan domains."""

    def test_hermitian_nuclear_norm_unchanged(self, numpy_ctx):
        X = sc.HermitianSpace(2, ctx=numpy_ctx)
        A = numpy_ctx.asarray([[2.0, 0.0], [0.0, -3.0]])
        f = sc.SpectralLpNormFunctional(X, 1)  # nuclear norm |2| + |-3|
        assert float(f.value(A)) == 5.0

    def test_elementwise_frobenius_unchanged(self, numpy_ctx):
        X = sc.ElementwiseJordanSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([3.0, -4.0, 0.0])
        f = sc.SpectralLpNormFunctional(X, 2)  # sqrt(9 + 16) = 5
        assert float(f.value(x)) == pytest.approx(5.0)


class TestStructuredSpectrumEdges:
    def test_single_leaf_tree_flat_vs_structured(self, numpy_ctx):
        # Arity 1: flat returns the bare leaf array; structured still returns a
        # treedef-shaped (1-leaf) pytree, exercising the structured branch's lack
        # of the len==1 shortcut.
        space = sc.TreeSpace.from_leaf_spaces(
            (sc.ElementwiseJordanSpace((3,), numpy_ctx),), numpy_ctx
        )
        x = space.element((numpy_ctx.asarray([2.0, -1.0, 4.0]),))
        np.testing.assert_allclose(to_numpy(space.spectrum(x)), [2.0, -1.0, 4.0])  # bare array
        structured = space.spectrum(x, structured=True)
        assert optree.tree_structure(structured) == space.treedef
        (leaf,) = space.flatten_tree(structured)
        np.testing.assert_allclose(to_numpy(leaf), [2.0, -1.0, 4.0])

    def test_structured_spectrum_is_not_a_space_member(self):
        # A Hermitian leaf's spectrum (n,) differs from its element shape (n, n), so
        # the structured spectrum matches the treedef but is NOT a valid member.
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
        leaves = (sc.HermitianSpace(2, ctx=ctx), sc.ElementwiseJordanSpace((2,), ctx))
        space = sc.TreeSpace.from_template((0, (0,)), leaves, ctx=ctx)
        x = space.unflatten_tree(
            (ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), ctx.asarray([1.0, -1.0]))
        )
        structured = space.spectrum(x, structured=True)
        assert optree.tree_structure(structured) == space.treedef
        with pytest.raises(sc.SpaceValidationError):
            space.check_member(structured)

    def test_complex_hermitian_nested_round_trip(self):
        # Complex eigenvalues/frames survive the structure-preserving round-trip.
        ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128, check_level="standard")
        leaves = (sc.HermitianSpace(2, ctx=ctx), sc.ElementwiseJordanSpace((2,), ctx))
        space = sc.TreeSpace.from_template((0, (0,)), leaves, ctx=ctx)
        herm = ctx.asarray([[2.0, 0.5 + 0.5j], [0.5 - 0.5j, 1.0]])
        x = space.unflatten_tree((herm, ctx.asarray([1.0 + 0j, -2.0 + 0j])))
        decomp = space.spectral_decompose(x)
        assert decomp.treedef == space.treedef
        rebuilt = space.from_spectrum(decomp)
        np.testing.assert_allclose(to_numpy(space.flatten(rebuilt)), to_numpy(space.flatten(x)))
