from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import has_jax, jax_real_dtype, to_numpy


class State(NamedTuple):
    a: object
    b: object


class OtherState(NamedTuple):
    a: object
    b: object


def _np_ctx(enable_checks=True):
    return sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=enable_checks)


def _jax_ctx(enable_checks=True):
    return sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=enable_checks)


def _spaces(ctx):
    return sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((3,), ctx)


def _state(ctx, a=(1.0, 2.0), b=(3.0, 4.0, 5.0)):
    return State(ctx.asarray(a), ctx.asarray(b))


def _assert_state_allclose(x, a, b):
    assert isinstance(x, State)
    np.testing.assert_allclose(to_numpy(x.a), a)
    np.testing.assert_allclose(to_numpy(x.b), b)


def test_tuple_tree_accepts_and_returns_plain_tuples():
    ctx = _np_ctx()
    tree = sc.TreeSpace.from_leaf_spaces(_spaces(ctx), ctx)
    x = (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0]))

    tree.check_member(x)
    assert isinstance(tree.zeros(), tuple)
    assert isinstance(tree.add(x, x), tuple)
    np.testing.assert_allclose(tree.flatten(x), [1.0, 2.0, 3.0, 4.0, 5.0])
    rebuilt = tree.unflatten(tree.flatten(x))
    assert isinstance(rebuilt, tuple)
    np.testing.assert_allclose(rebuilt[0], x[0])
    np.testing.assert_allclose(rebuilt[1], x[1])


def test_named_tuple_tree_numpy_operations_and_flat_boundary():
    ctx = _np_ctx()
    x = _state(ctx)
    y = State(ctx.asarray([10.0, 20.0]), ctx.asarray([30.0, 40.0, 50.0]))
    tree = sc.TreeSpace.from_template(x, _spaces(ctx), ctx=ctx)

    _assert_state_allclose(tree.add(x, y), [11.0, 22.0], [33.0, 44.0, 55.0])
    _assert_state_allclose(tree.scale(2.0, x), [2.0, 4.0], [6.0, 8.0, 10.0])
    assert np.allclose(tree.inner(x, x), 55.0)
    _assert_state_allclose(tree.zeros(), [0.0, 0.0], [0.0, 0.0, 0.0])
    _assert_state_allclose(tree.ones(), [1.0, 1.0], [1.0, 1.0, 1.0])

    np.testing.assert_allclose(tree.flatten(x), [1.0, 2.0, 3.0, 4.0, 5.0])
    _assert_state_allclose(tree.unflatten(tree.flatten(x)), [1.0, 2.0], [3.0, 4.0, 5.0])


@pytest.mark.skipif(not has_jax(), reason="JAX is not installed")
def test_named_tuple_tree_jax_operations():
    ctx = _jax_ctx()
    x = _state(ctx)
    tree = sc.TreeSpace.from_template(x, _spaces(ctx), ctx=ctx)

    _assert_state_allclose(tree.add(x, x), [2.0, 4.0], [6.0, 8.0, 10.0])
    _assert_state_allclose(tree.scale(3.0, x), [3.0, 6.0], [9.0, 12.0, 15.0])
    assert np.allclose(to_numpy(tree.inner(x, x)), 55.0)


def test_structure_mismatch_errors_are_clear():
    ctx = _np_ctx()
    x = _state(ctx)
    tree = sc.TreeSpace.from_template(x, _spaces(ctx), ctx=ctx)

    with pytest.raises(TypeError, match="structure mismatch"):
        tree.check_member(OtherState(x.a, x.b))
    with pytest.raises(TypeError, match="structure mismatch"):
        tree.check_member((x.a, x.b))


def test_equality_and_conversion_include_tree_structure():
    ctx = _np_ctx()
    target = sc.Context(sc.NumpyOps(), dtype=np.float32)
    x = _state(ctx)
    same_shape = State(ctx.asarray([9.0, 8.0]), ctx.asarray([7.0, 6.0, 5.0]))
    spaces = _spaces(ctx)

    tree = sc.TreeSpace.from_template(x, spaces, ctx=ctx)
    same_tree = sc.TreeSpace.from_template(same_shape, spaces, ctx=ctx)
    tuple_tree = sc.TreeSpace.from_leaf_spaces(spaces, ctx)

    assert tree == same_tree
    assert tree != tuple_tree
    converted = tree.convert(target)
    assert converted.treedef == tree.treedef
    assert converted != tuple_tree.convert(target)


def test_batch_semantics_are_tree_of_batched_leaves():
    ctx = _np_ctx()
    tree = sc.TreeSpace.from_template(_state(ctx), _spaces(ctx), ctx=ctx)
    batch = State(
        ctx.asarray([[1.0, 2.0], [3.0, 4.0]]),
        ctx.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
    )

    doubled = tree.add_batch(batch, batch)
    _assert_state_allclose(doubled, [[2.0, 4.0], [6.0, 8.0]], [[10.0, 12.0, 14.0], [16.0, 18.0, 20.0]])
    flat_batch = tree.flatten_batch(batch)
    np.testing.assert_allclose(flat_batch, [[1.0, 2.0, 5.0, 6.0, 7.0], [3.0, 4.0, 8.0, 9.0, 10.0]])
    _assert_state_allclose(tree.unflatten_batch(flat_batch), batch.a, batch.b)


def test_solver_smoke_with_named_tuple_tree_space():
    ctx = _np_ctx()
    tree = sc.TreeSpace.from_template(_state(ctx), _spaces(ctx), ctx=ctx)
    operator = sc.MatrixFreeLinOp(lambda x: x, lambda x: x, tree, tree, ctx)
    result = sc.cg(operator, _state(ctx), tol=1e-12, maxiter=5, check_every=1)

    assert bool(result.converged)
    _assert_state_allclose(result.x, [1.0, 2.0], [3.0, 4.0, 5.0])


@pytest.mark.skipif(not has_jax(), reason="JAX is not installed")
def test_tree_space_jax_registration_preserves_static_structure():
    import jax

    ctx = _jax_ctx()
    tree = sc.TreeSpace.from_template(_state(ctx), _spaces(ctx), ctx=ctx)
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert leaves == []
    assert rebuilt == tree
    assert rebuilt.treedef == tree.treedef
