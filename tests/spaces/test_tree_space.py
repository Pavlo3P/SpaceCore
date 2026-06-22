"""Tests for :class:`spacecore.TreeSpace`, :class:`spacecore.TreeElement`,
and the tree-mixin capability dispatch.

Checklist items 17, 18, 21:

* :class:`TreeSpace` —
  - ``__new__`` capability dispatch (inner-product / star / jordan mixin variants)
  - ``from_leaf_spaces`` (tuple-style) and ``from_template`` (named-tuple/pytree)
  - ``treedef`` / ``leaf_paths`` / ``arity`` properties
  - ``flatten_tree`` ↔ ``unflatten_tree`` (pytree-shape round-trip)
  - ``element(value)`` returns a ``TreeElement`` bound to the space
  - ``zero`` / ``zeros`` / ``ones`` / leafwise ``add`` / ``scale``
  - Batched ``add_batch`` / ``flatten_batch`` / ``unflatten_batch``
  - ``flatten`` / ``unflatten`` to a dense 1-D vector
  - ``convert`` / ``convert_element`` to a new context
  - ``tree_flatten`` / ``tree_unflatten`` JAX pytree round-trip
  - Constructor validates leaf types and leaf count
  - Componentwise capability is the intersection of the leaves
* :class:`TreeElement` — ``value`` property, ``tree_flatten`` / ``tree_unflatten``
* Tree mixin spaces — InnerProduct / Star / Jordan dispatch from the leaves
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import has_jax, jax_real_dtype, to_numpy


class _State(NamedTuple):
    a: object
    b: object


class _OtherState(NamedTuple):
    a: object
    b: object


class _CoordinateOnlySpace(sc.CoordinateSpace):
    """Minimal coordinate space without inner-product capability."""

    def zeros(self):
        return self.ops.zeros(self.shape, dtype=self.dtype)

    def add(self, x, y):
        return x + y

    def scale(self, a, x):
        return a * x

    def flatten(self, x):
        return x.reshape((-1,))

    def unflatten(self, vector):
        return vector.reshape(self.shape)

    def _convert(self, new_ctx):
        return type(self)(self.shape, new_ctx)


def _three_spaces(ctx):
    return (
        sc.DenseCoordinateSpace((2,), ctx),
        sc.DenseCoordinateSpace((1,), ctx),
        sc.DenseCoordinateSpace((2, 2), ctx),
    )


def _nested_template():
    return {"bias": 0, "model": (0, {"weight": 0})}


def _nested_value(ctx):
    return {
        "model": (
            ctx.asarray([1.0]),
            {"weight": ctx.asarray([[2.0, 3.0], [4.0, 5.0]])},
        ),
        "bias": ctx.asarray([6.0, 7.0]),
    }


# ===========================================================================
# Construction and validation
# ===========================================================================
class TestConstruction:
    def test_from_leaf_spaces_tuple_style(self, numpy_ctx):
        leaves = (
            sc.DenseCoordinateSpace((2,), numpy_ctx),
            sc.DenseCoordinateSpace((3,), numpy_ctx),
        )
        tree = sc.TreeSpace.from_leaf_spaces(leaves, numpy_ctx)
        assert isinstance(tree, sc.TreeSpace)
        assert tree.shape == (5,)
        assert tree.arity == 2
        assert len(tree.leaf_spaces) == 2

    def test_from_template_named_tuple(self, numpy_ctx):
        template = _State(0, 0)
        leaves = (
            sc.DenseCoordinateSpace((2,), numpy_ctx),
            sc.DenseCoordinateSpace((3,), numpy_ctx),
        )
        tree = sc.TreeSpace.from_template(template, leaves, ctx=numpy_ctx)
        assert isinstance(tree, sc.TreeSpace)
        assert tree.shape == (5,)

    def test_nested_dict_with_explicit_treedef(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        assert tree.leaf_paths == (("bias",), ("model", 0), ("model", 1, "weight"))

    def test_rejects_non_coordinate_leaf(self, numpy_ctx):
        with pytest.raises(TypeError, match="every leaf to be a CoordinateSpace"):
            sc.TreeSpace((0,), (object(),), ctx=numpy_ctx)

    def test_rejects_leaf_count_mismatch(self, numpy_ctx):
        leaves = _three_spaces(numpy_ctx)
        with pytest.raises(ValueError, match="leaf-count mismatch"):
            sc.TreeSpace((0, 0), leaves[:1], ctx=numpy_ctx)

    def test_rejects_non_sequence_leaves(self, numpy_ctx):
        leaves = _three_spaces(numpy_ctx)
        with pytest.raises(TypeError, match="sequence of CoordinateSpace leaves"):
            sc.TreeSpace((0,), leaves[0], ctx=numpy_ctx)


# ===========================================================================
# Properties: treedef / leaf_paths / arity
# ===========================================================================
class TestStructureProperties:
    def test_arity_matches_leaf_count(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        assert tree.arity == 3

    def test_leaf_paths_are_deterministic(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        # Sorted by optree's flatten order — pinning the result locks the
        # contract.
        assert tree.leaf_paths == (("bias",), ("model", 0), ("model", 1, "weight"))


# ===========================================================================
# flatten_tree ↔ unflatten_tree round-trip
# ===========================================================================
class TestFlattenTreeRoundTrip:
    def test_round_trip_on_nested_dict(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        value = _nested_value(numpy_ctx)
        leaves = tree.flatten_tree(value)
        rebuilt = tree.unflatten_tree(leaves)
        np.testing.assert_allclose(rebuilt["bias"], value["bias"])
        np.testing.assert_allclose(rebuilt["model"][1]["weight"], value["model"][1]["weight"])

    def test_unflatten_wrong_leaf_count_raises(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        with pytest.raises(ValueError, match="expected 3 leaves, got 2"):
            tree.unflatten_tree((numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0])))


# ===========================================================================
# TreeElement: value property, JAX pytree round-trip
# ===========================================================================
class TestTreeElement:
    def test_value_property_reconstructs_input(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        value = _nested_value(numpy_ctx)
        element = tree.element(value)
        assert element.space == tree
        # ``element.value`` rebuilds the pytree.
        rebuilt = element.value
        np.testing.assert_allclose(rebuilt["bias"], value["bias"])

    def test_element_leaves_match_flatten_tree(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        value = _nested_value(numpy_ctx)
        element = tree.element(value)
        assert element.leaves == tree.flatten_tree(value)

    def test_element_with_wrong_leaf_count_raises(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        with pytest.raises(ValueError, match="TreeElement expected 3 leaves, got 2"):
            sc.TreeElement(tree, (numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0])))


# ===========================================================================
# zero / zeros / ones / add / scale (leafwise)
# ===========================================================================
class TestLeafwiseOperations:
    def test_zero_returns_leafwise_zeros(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        zero = tree.zero()
        for leaf in tree.flatten_tree(zero):
            np.testing.assert_allclose(leaf, 0.0)

    def test_add_is_leafwise(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        x = tree.element(_nested_value(numpy_ctx))
        doubled = tree.add(x, x)
        scaled = tree.scale(0.5, doubled)
        for actual, expected in zip(tree.flatten_tree(scaled), x.leaves):
            np.testing.assert_allclose(actual, expected)


# ===========================================================================
# Flatten/unflatten to a dense 1-D vector
# ===========================================================================
class TestFlattenDense:
    def test_flatten_yields_concatenated_leaves(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        x = tree.element(_nested_value(numpy_ctx))
        # Order follows leaf_paths: bias (2,), model[0] (1,), model[1].weight (2, 2).
        flat = tree.flatten(x)
        np.testing.assert_allclose(flat, [6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_unflatten_round_trip(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        x = tree.element(_nested_value(numpy_ctx))
        flat = tree.flatten(x)
        rebuilt = tree.unflatten(flat)
        for actual, expected in zip(tree.flatten_tree(rebuilt), x.leaves):
            np.testing.assert_allclose(actual, expected)


# ===========================================================================
# Batched variants
# ===========================================================================
class TestBatched:
    def test_flatten_batch_round_trip(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        x = tree.element(_nested_value(numpy_ctx))
        batch = sc.TreeElement(tree, tuple(np.stack([leaf, leaf]) for leaf in x.leaves))
        flat_batch = tree.flatten_batch(batch)
        assert flat_batch.shape == (2, 7)
        rebuilt = tree.unflatten_batch(flat_batch)
        for actual, expected in zip(tree.flatten_tree(rebuilt), batch.leaves):
            np.testing.assert_allclose(actual, expected)

    def test_named_tuple_batch_add_is_leafwise(self, numpy_ctx):
        template = _State(numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0, 4.0, 5.0]))
        leaves = (sc.DenseCoordinateSpace((2,), numpy_ctx),
                  sc.DenseCoordinateSpace((3,), numpy_ctx))
        tree = sc.TreeSpace.from_template(template, leaves, ctx=numpy_ctx)
        batch = _State(
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0]]),
            numpy_ctx.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
        )
        doubled = tree.add_batch(batch, batch)
        assert isinstance(doubled, _State)
        np.testing.assert_allclose(to_numpy(doubled.a), [[2, 4], [6, 8]])
        np.testing.assert_allclose(to_numpy(doubled.b), [[10, 12, 14], [16, 18, 20]])


# ===========================================================================
# check_member — reports leaf path
# ===========================================================================
class TestCheck:
    def test_check_reports_leaf_path(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        leaves = list(tree.element(_nested_value(numpy_ctx)).leaves)
        leaves[2] = numpy_ctx.asarray([1.0, 2.0])
        invalid = sc.TreeElement(tree, leaves)
        with pytest.raises(sc.SpaceValidationError,
                           match=r"\$\.model\[1\]\.weight.*Expected shape"):
            tree.check(invalid)

    def test_check_skipped_when_check_level_none(self, numpy_ctx):
        none_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
        tree = sc.TreeSpace(_nested_template(), _three_spaces(none_ctx), ctx=none_ctx)
        leaves = list(tree.element(_nested_value(none_ctx)).leaves)
        leaves[2] = none_ctx.asarray([1.0, 2.0])  # wrong shape
        # No raise — checks are disabled.
        tree.check(sc.TreeElement(tree, leaves))


# ===========================================================================
# Convert / convert_element
# ===========================================================================
class TestConvert:
    def test_convert_preserves_structure_and_paths(self, numpy_ctx, numpy_f32_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        converted = tree.convert(numpy_f32_ctx)
        assert converted.treedef == tree.treedef
        assert converted.leaf_paths == tree.leaf_paths
        assert all(space.dtype == numpy_f32_ctx.dtype for space in converted.leaf_spaces)

    def test_convert_element_changes_dtype(self, numpy_ctx, numpy_f32_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        x = tree.element(_nested_value(numpy_ctx))
        converted = tree.convert_element(x, numpy_f32_ctx)
        assert all(leaf.dtype == numpy_f32_ctx.dtype
                   for leaf in tree.convert(numpy_f32_ctx).flatten_tree(converted))


# ===========================================================================
# Tuple-style TreeSpace (absorbed from test_tree_tuple_space.py)
# ===========================================================================
class TestTupleStyle:
    def test_tuple_elements_preserved(self, numpy_ctx):
        leaves = (
            sc.DenseCoordinateSpace((2,), numpy_ctx),
            sc.DenseCoordinateSpace((3,), numpy_ctx),
        )
        product = sc.TreeSpace.from_leaf_spaces(leaves, numpy_ctx)
        x = (numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0, 4.0, 5.0]))
        assert isinstance(product.zeros(), tuple)
        assert isinstance(product.add(x, x), tuple)
        np.testing.assert_allclose(product.flatten(x), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_check_member_rejects_wrong_arity(self, numpy_ctx):
        leaves = (
            sc.DenseCoordinateSpace((2,), numpy_ctx),
            sc.DenseCoordinateSpace((3,), numpy_ctx),
        )
        product = sc.TreeSpace.from_leaf_spaces(leaves, numpy_ctx)
        product.check_member((numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0, 4.0, 5.0])))
        with pytest.raises(Exception):
            product.check_member((numpy_ctx.asarray([1.0, 2.0]),))

    def test_uses_resolved_context_dtype_for_leaves(self, numpy_ctx, numpy_f32_ctx):
        leaves = (sc.DenseCoordinateSpace((2,), numpy_f32_ctx),
                  sc.DenseCoordinateSpace((3,), numpy_ctx))
        product = sc.TreeSpace.from_leaf_spaces(leaves, numpy_ctx)
        assert product.dtype == numpy_ctx.dtype
        assert all(sp.dtype == numpy_ctx.dtype for sp in product.leaf_spaces)


# ===========================================================================
# Named-tuple TreeSpace (absorbed from test_tree_structures.py)
# ===========================================================================
class TestNamedTupleStructure:
    def _tree(self, ctx):
        template = _State(ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0]))
        leaves = (sc.DenseCoordinateSpace((2,), ctx),
                  sc.DenseCoordinateSpace((3,), ctx))
        return sc.TreeSpace.from_template(template, leaves, ctx=ctx), template

    def test_add_scale_inner_preserve_named_tuple(self, numpy_ctx):
        tree, x = self._tree(numpy_ctx)
        y = _State(numpy_ctx.asarray([10.0, 20.0]), numpy_ctx.asarray([30.0, 40.0, 50.0]))
        out = tree.add(x, y)
        assert isinstance(out, _State)
        np.testing.assert_allclose(to_numpy(out.a), [11, 22])
        np.testing.assert_allclose(to_numpy(out.b), [33, 44, 55])
        assert np.allclose(tree.inner(x, x), 55.0)

    def test_structure_mismatch_clear_message(self, numpy_ctx):
        tree, x = self._tree(numpy_ctx)
        with pytest.raises(TypeError, match="structure mismatch"):
            tree.check_member(_OtherState(x.a, x.b))
        with pytest.raises(TypeError, match="structure mismatch"):
            tree.check_member((x.a, x.b))

    def test_equality_distinguishes_named_tuple_vs_tuple(self, numpy_ctx):
        tree, _ = self._tree(numpy_ctx)
        tuple_tree = sc.TreeSpace.from_leaf_spaces(
            (sc.DenseCoordinateSpace((2,), numpy_ctx),
             sc.DenseCoordinateSpace((3,), numpy_ctx)),
            numpy_ctx,
        )
        assert tree != tuple_tree


# ===========================================================================
# Capability dispatch (items 17 dispatch + 21 tree mixins)
# ===========================================================================
class TestCapabilityDispatch:
    def test_baseline_tree_has_no_inner_product(self, numpy_ctx):
        product = sc.TreeSpace.from_leaf_spaces(
            (_CoordinateOnlySpace((2,), numpy_ctx),
             _CoordinateOnlySpace((1,), numpy_ctx)),
            numpy_ctx,
        )
        assert type(product) is sc.TreeSpace
        assert isinstance(product, sc.CoordinateSpace)
        assert not isinstance(product, sc.InnerProductSpace)
        assert not isinstance(product, sc.StarSpace)
        assert not isinstance(product, sc.JordanAlgebraSpace)

    def test_inner_leaves_lift_inner_product(self, numpy_ctx):
        product = sc.TreeSpace.from_leaf_spaces(
            (sc.DenseCoordinateSpace((2,), numpy_ctx),
             sc.DenseCoordinateSpace((1,), numpy_ctx)),
            numpy_ctx,
        )
        assert isinstance(product, sc.InnerProductSpace)
        assert not isinstance(product, sc.JordanAlgebraSpace)

    def test_jordan_leaves_lift_jordan(self, numpy_ctx):
        product = sc.TreeSpace.from_leaf_spaces(
            (sc.ElementwiseJordanSpace((2,), numpy_ctx),
             sc.HermitianSpace(2, ctx=numpy_ctx)),
            numpy_ctx,
        )
        assert isinstance(product, sc.StarSpace)
        assert isinstance(product, sc.EuclideanJordanAlgebraSpace)

    def test_capability_is_intersection_of_components(self, numpy_ctx):
        """Two inner-product leaves with different geometry → still inner-product."""
        euclidean = sc.DenseCoordinateSpace((2,), numpy_ctx)
        weighted = sc.DenseCoordinateSpace(
            (2,), numpy_ctx,
            geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0])),
        )
        product = sc.TreeSpace.from_leaf_spaces((euclidean, weighted), numpy_ctx)
        assert isinstance(product, sc.InnerProductSpace)
        assert product.is_euclidean is False

    def test_mixed_leaves_drop_inner(self, numpy_ctx):
        """If any leaf lacks inner-product, the tree lacks inner-product."""
        mixed = sc.TreeSpace(
            (0, 0),
            (sc.DenseCoordinateSpace((1,), numpy_ctx),
             _CoordinateOnlySpace((1,), numpy_ctx)),
            ctx=numpy_ctx,
        )
        assert not isinstance(mixed, sc.InnerProductSpace)


# ===========================================================================
# Inner-product values: leafwise sum + riesz componentwise
# ===========================================================================
class TestInnerProduct:
    def test_inner_sums_leafwise_components(self, numpy_ctx):
        tree = sc.TreeSpace(_nested_template(), _three_spaces(numpy_ctx), ctx=numpy_ctx)
        x = tree.element(_nested_value(numpy_ctx))
        # bias·bias + model[0]·model[0] + model[1].weight·model[1].weight
        # = 6²+7² + 1² + 2²+3²+4²+5² = 36+49 + 1 + 4+9+16+25 = 140
        assert np.allclose(tree.inner(x, x), 140.0)
        assert np.allclose(tree.norm(x), np.sqrt(140.0))

    def test_riesz_componentwise_on_mixed_geometry(self, numpy_ctx):
        weighted = sc.DenseCoordinateSpace(
            (2,), numpy_ctx,
            geometry=sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0])),
        )
        product = sc.TreeSpace.from_leaf_spaces(
            (sc.DenseCoordinateSpace((2,), numpy_ctx), weighted), numpy_ctx,
        )
        x = (numpy_ctx.asarray([1.0, 2.0]), numpy_ctx.asarray([3.0, 4.0]))
        dual = product.riesz(x)
        np.testing.assert_allclose(dual[0], x[0])
        np.testing.assert_allclose(dual[1], [6.0, 12.0])
        roundtrip = product.riesz_inverse(dual)
        np.testing.assert_allclose(roundtrip[0], x[0])
        np.testing.assert_allclose(roundtrip[1], x[1])


# ===========================================================================
# JAX pytree round-trip
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
class TestJaxPytree:
    def test_tree_space_round_trip(self):
        import jax
        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        tree = sc.TreeSpace(_nested_template(), _three_spaces(ctx), ctx=ctx)
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert leaves == []
        assert rebuilt == tree
        assert rebuilt.treedef == tree.treedef

    def test_tree_element_round_trip(self):
        import jax
        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        tree = sc.TreeSpace(_nested_template(), _three_spaces(ctx), ctx=ctx)
        element = tree.element(_nested_value(ctx))
        leaves, treedef = jax.tree_util.tree_flatten(element)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt.space == tree
        for actual, expected in zip(rebuilt.leaves, element.leaves):
            np.testing.assert_allclose(to_numpy(actual), to_numpy(expected))

    def test_named_tuple_structure_preserved_through_jax(self):
        import jax
        ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), check_level="none")
        template = _State(ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0]))
        leaves = (sc.DenseCoordinateSpace((2,), ctx),
                  sc.DenseCoordinateSpace((3,), ctx))
        tree = sc.TreeSpace.from_template(template, leaves, ctx=ctx)
        flat, treedef = jax.tree_util.tree_flatten(tree)
        rebuilt = jax.tree_util.tree_unflatten(treedef, flat)
        assert rebuilt == tree
