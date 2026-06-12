import numpy as np
import pytest

import spacecore as sc
from tests._helpers import has_jax, to_numpy


class CoordinateOnlySpace(sc.CoordinateSpace):
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


def _ctx(dtype=np.float64, *, check_level="standard"):
    return sc.Context(sc.NumpyOps(), dtype=dtype, check_level=check_level)


def _spaces(ctx):
    return (
        sc.DenseCoordinateSpace((2,), ctx),
        sc.DenseCoordinateSpace((1,), ctx),
        sc.DenseCoordinateSpace((2, 2), ctx),
    )


def _template():
    return {"bias": 0, "model": (0, {"weight": 0})}


def _value(ctx):
    return {
        "model": (
            ctx.asarray([1.0]),
            {"weight": ctx.asarray([[2.0, 3.0], [4.0, 5.0]])},
        ),
        "bias": ctx.asarray([6.0, 7.0]),
    }


def test_constructor_validates_leaf_spaces_and_leaf_count():
    ctx = _ctx()
    spaces = _spaces(ctx)

    with pytest.raises(TypeError, match="sequence of CoordinateSpace leaves"):
        sc.TreeSpace((0,), spaces[0], ctx=ctx)
    with pytest.raises(TypeError, match="every leaf to be a CoordinateSpace"):
        sc.TreeSpace((0,), (object(),), ctx=ctx)
    with pytest.raises(ValueError, match="leaf-count mismatch"):
        sc.TreeSpace((0, 0), spaces[:1], ctx=ctx)


def test_nested_structure_round_trip_and_deterministic_paths():
    ctx = _ctx()
    tree = sc.TreeSpace(_template(), _spaces(ctx), ctx=ctx)
    value = _value(ctx)

    assert tree.leaf_paths == (("bias",), ("model", 0), ("model", 1, "weight"))
    element = tree.element(value)
    assert element.space == tree
    assert element.leaves == tree.flatten_tree(value)
    rebuilt = element.value
    np.testing.assert_allclose(rebuilt["bias"], value["bias"])
    np.testing.assert_allclose(rebuilt["model"][1]["weight"], value["model"][1]["weight"])


def test_structure_and_leaf_count_mismatch_errors_are_clear():
    ctx = _ctx()
    tree = sc.TreeSpace(_template(), _spaces(ctx), ctx=ctx)

    with pytest.raises(TypeError, match="structure mismatch"):
        tree.element({"bias": ctx.asarray([1.0, 2.0]), "model": [ctx.asarray([3.0])]})
    with pytest.raises(ValueError, match="expected 3 leaves, got 2"):
        tree.unflatten_tree((ctx.asarray([1.0, 2.0]), ctx.asarray([3.0])))
    with pytest.raises(ValueError, match="TreeElement expected 3 leaves, got 2"):
        sc.TreeElement(tree, (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0])))


def test_zero_add_scale_flatten_and_batch_round_trips():
    ctx = _ctx()
    tree = sc.TreeSpace(_template(), _spaces(ctx), ctx=ctx)
    x = tree.element(_value(ctx))

    zero = tree.zero()
    assert all(np.allclose(leaf, 0.0) for leaf in tree.flatten_tree(zero))
    doubled = tree.add(x, x)
    scaled = tree.scale(0.5, doubled)
    for actual, expected in zip(tree.flatten_tree(scaled), x.leaves):
        np.testing.assert_allclose(actual, expected)

    flat = tree.flatten(x)
    np.testing.assert_allclose(flat, [6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    roundtrip = tree.unflatten(flat)
    for actual, expected in zip(tree.flatten_tree(roundtrip), x.leaves):
        np.testing.assert_allclose(actual, expected)

    batch = sc.TreeElement(tree, tuple(np.stack([leaf, leaf]) for leaf in x.leaves))
    flat_batch = tree.flatten_batch(batch)
    assert flat_batch.shape == (2, 7)
    batch_roundtrip = tree.unflatten_batch(flat_batch)
    for actual, expected in zip(tree.flatten_tree(batch_roundtrip), batch.leaves):
        np.testing.assert_allclose(actual, expected)


def test_check_reports_leaf_path_and_respects_check_level():
    ctx = _ctx(dtype=np.float32)
    tree = sc.TreeSpace(_template(), _spaces(ctx), ctx=ctx)
    leaves = list(tree.element(_value(ctx)).leaves)
    leaves[2] = ctx.asarray([1.0, 2.0])
    invalid = sc.TreeElement(tree, leaves)

    with pytest.raises(sc.SpaceValidationError, match=r"\$\.model\[1\]\.weight.*Expected shape"):
        tree.check(invalid)

    unchecked_ctx = _ctx(dtype=np.float32, check_level="none")
    unchecked = sc.TreeSpace(_template(), _spaces(unchecked_ctx), ctx=unchecked_ctx)
    unchecked.check(sc.TreeElement(unchecked, leaves))


def test_convert_preserves_structure_paths_and_converts_elements():
    ctx = _ctx(dtype=np.float64)
    target_ctx = _ctx(dtype=np.float32)
    tree = sc.TreeSpace(_template(), _spaces(ctx), ctx=ctx)
    x = tree.element(_value(ctx))

    converted = tree.convert(target_ctx)
    assert converted.treedef == tree.treedef
    assert converted.leaf_paths == tree.leaf_paths
    assert all(space.dtype == np.dtype(np.float32) for space in converted.leaf_spaces)

    converted_x = tree.convert_element(x, target_ctx)
    assert all(
        leaf.dtype == np.dtype(np.float32)
        for leaf in converted.flatten_tree(converted_x)
    )
    np.testing.assert_allclose(converted_x["bias"], [6.0, 7.0])


def test_inner_norm_and_capability_dispatch():
    ctx = _ctx()
    tree = sc.TreeSpace(_template(), _spaces(ctx), ctx=ctx)
    x = tree.element(_value(ctx))

    assert isinstance(tree, sc.InnerProductSpace)
    assert np.allclose(tree.inner(x, x), 140.0)
    assert np.allclose(tree.norm(x), np.sqrt(140.0))

    mixed = sc.TreeSpace(
        (0, 0),
        (sc.DenseCoordinateSpace((1,), ctx), CoordinateOnlySpace((1,), ctx)),
        ctx=ctx,
    )
    assert not isinstance(mixed, sc.InnerProductSpace)


def test_tuple_tree_space_preserves_plain_tuple_elements():
    ctx = _ctx()
    product = sc.TreeSpace.from_leaf_spaces(_spaces(ctx)[:2], ctx)
    x = (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0]))

    assert isinstance(product, sc.TreeSpace)
    assert isinstance(product.zeros(), tuple)
    assert isinstance(product.add(x, x), tuple)
    np.testing.assert_allclose(product.flatten(x), [1.0, 2.0, 3.0])


@pytest.mark.skipif(not has_jax(), reason="JAX is not installed")
def test_tree_space_and_element_are_registered_jax_pytrees():
    import jax

    ctx = _ctx()
    tree = sc.TreeSpace(_template(), _spaces(ctx), ctx=ctx)
    element = tree.element(_value(ctx))

    leaves, treedef = jax.tree_util.tree_flatten(element)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.space == tree
    for actual, expected in zip(rebuilt.leaves, element.leaves):
        np.testing.assert_allclose(to_numpy(actual), to_numpy(expected))

    space_leaves, space_treedef = jax.tree_util.tree_flatten(tree)
    rebuilt_space = jax.tree_util.tree_unflatten(space_treedef, space_leaves)
    assert space_leaves == []
    assert rebuilt_space == tree
