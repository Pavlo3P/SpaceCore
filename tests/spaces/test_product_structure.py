from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import has_jax, jax_real_dtype, to_numpy

jax = pytest.importorskip("jax")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class State:
    a: object
    b: object

    def tree_flatten(self):
        return (self.a, self.b), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OtherState:
    a: object
    b: object

    def tree_flatten(self):
        return (self.a, self.b), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NestedState:
    left: object
    right: object

    def tree_flatten(self):
        return ((self.left,), self.right), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        left, right = children
        return cls(left[0], right)


@dataclass(frozen=True)
class UnregisteredState:
    a: object
    b: object


def _np_ctx(enable_checks=True):
    return sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=enable_checks)


def _jax_ctx(enable_checks=True):
    dtype = jax_real_dtype()
    return sc.Context(sc.JaxOps(), dtype=dtype, enable_checks=enable_checks)


def _spaces(ctx):
    return sc.VectorSpace((2,), ctx), sc.VectorSpace((3,), ctx)


def _state(ctx, a=(1.0, 2.0), b=(3.0, 4.0, 5.0)):
    return State(ctx.asarray(a), ctx.asarray(b))


def _assert_state_allclose(x, a, b):
    assert isinstance(x, State)
    np.testing.assert_allclose(to_numpy(x.a), a)
    np.testing.assert_allclose(to_numpy(x.b), b)


def test_tuple_default_regression_accepts_and_returns_plain_tuples():
    ctx = _np_ctx()
    P = sc.ProductSpace(_spaces(ctx), ctx)
    x = (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0]))

    P.check_member(x)
    assert isinstance(P.zeros(), tuple)
    assert isinstance(P.add(x, x), tuple)
    np.testing.assert_allclose(P.flatten(x), [1.0, 2.0, 3.0, 4.0, 5.0])
    y = P.unflatten(P.flatten(x))
    assert isinstance(y, tuple)
    np.testing.assert_allclose(y[0], x[0])
    np.testing.assert_allclose(y[1], x[1])


def test_registered_pytree_element_numpy_operations_and_flat_boundary():
    ctx = _np_ctx()
    spaces = _spaces(ctx)
    x = _state(ctx)
    y = State(ctx.asarray([10.0, 20.0]), ctx.asarray([30.0, 40.0, 50.0]))
    P = sc.ProductSpace.from_template(spaces, x, ctx)
    tuple_product = sc.ProductSpace(spaces, ctx)

    _assert_state_allclose(P.add(x, y), [11.0, 22.0], [33.0, 44.0, 55.0])
    _assert_state_allclose(P.scale(2.0, x), [2.0, 4.0], [6.0, 8.0, 10.0])
    assert np.allclose(P.inner(x, x), 55.0)
    _assert_state_allclose(P.zeros(), [0.0, 0.0], [0.0, 0.0, 0.0])
    _assert_state_allclose(P.ones(), [1.0, 1.0], [1.0, 1.0, 1.0])

    flat_state = P.flatten(x)
    flat_tuple = tuple_product.flatten((x.a, x.b))
    np.testing.assert_allclose(flat_state, flat_tuple)
    xr = P.unflatten(flat_state)
    _assert_state_allclose(xr, [1.0, 2.0], [3.0, 4.0, 5.0])


def test_registered_pytree_element_jax_operations_when_available():
    if not has_jax():
        pytest.skip("JAX is not available")
    ctx = _jax_ctx()
    spaces = _spaces(ctx)
    x = _state(ctx)
    P = sc.ProductSpace.from_template(spaces, x, ctx)

    _assert_state_allclose(P.add(x, x), [2.0, 4.0], [6.0, 8.0, 10.0])
    _assert_state_allclose(P.scale(3.0, x), [3.0, 6.0], [9.0, 12.0, 15.0])
    assert np.allclose(to_numpy(P.inner(x, x)), 55.0)
    _assert_state_allclose(P.unflatten(P.flatten(x)), [1.0, 2.0], [3.0, 4.0, 5.0])


def test_round_trip_law_for_tuple_and_registered_pytree_structures():
    ctx = _np_ctx()
    x_tuple = (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0]))
    tuple_structure = sc.TupleStructure()
    tuple_parts = tuple_structure.to_components(x_tuple, arity=2)
    assert tuple_structure.from_components(tuple_parts, arity=2) == x_tuple

    x = _state(ctx)
    structure = sc.PytreeStructure(x)
    parts = structure.to_components(x, arity=2)
    y = structure.from_components(parts, arity=2)
    assert isinstance(y, State)
    np.testing.assert_allclose(y.a, x.a)
    np.testing.assert_allclose(y.b, x.b)


def test_mismatch_errors_are_clear():
    ctx = _np_ctx()
    P = sc.ProductSpace(_spaces(ctx), ctx)
    x = _state(ctx)
    structure = sc.PytreeStructure(x)

    with pytest.raises(ValueError, match="Expected tuple of length 2, got 1"):
        P.check_member((ctx.asarray([1.0, 2.0]),))
    with pytest.raises(TypeError, match="ProductSpace element must be a tuple"):
        P.check_member([ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0, 5.0])])
    with pytest.raises(TypeError, match="pytree structure mismatch"):
        structure.to_components(OtherState(x.a, x.b), arity=2)
    with pytest.raises(ValueError, match="register it as a JAX pytree"):
        sc.PytreeStructure(UnregisteredState(x.a, x.b)).to_components(
            UnregisteredState(x.a, x.b),
            arity=2,
        )
    with pytest.raises(ValueError, match="Expected 2 product components, got 1"):
        structure.from_components((x.a,), arity=2)


def test_structure_equality_and_product_equality_include_structure():
    ctx = _np_ctx()
    x = _state(ctx)
    same_shape_different_values = State(ctx.asarray([9.0, 8.0]), ctx.asarray([7.0, 6.0, 5.0]))

    assert sc.TupleStructure() == sc.TupleStructure()
    assert sc.PytreeStructure(x) == sc.PytreeStructure(same_shape_different_values)
    assert sc.PytreeStructure(x) != sc.PytreeStructure(NestedState(x.a, x.b))
    assert sc.TupleStructure() != sc.PytreeStructure(x)

    spaces = _spaces(ctx)
    tuple_product = sc.ProductSpace(spaces, ctx)
    pytree_product = sc.ProductSpace.from_template(spaces, x, ctx)
    pytree_product_same = sc.ProductSpace.from_template(spaces, same_shape_different_values, ctx)
    assert pytree_product == pytree_product_same
    assert tuple_product != pytree_product


def test_convert_preserves_tuple_and_pytree_structure():
    ctx = _np_ctx()
    target = sc.Context(sc.NumpyOps(), dtype=np.float32)
    spaces = _spaces(ctx)
    tuple_product = sc.ProductSpace(spaces, ctx)
    pytree_product = sc.ProductSpace.from_template(spaces, _state(ctx), ctx)

    assert isinstance(tuple_product.convert(target).structure, sc.TupleStructure)
    converted = pytree_product.convert(target)
    assert converted.structure == pytree_product.structure
    assert converted != tuple_product.convert(target)


def test_batch_semantics_are_structure_of_batched_components():
    ctx = _np_ctx()
    spaces = _spaces(ctx)
    template = _state(ctx)
    P = sc.ProductSpace.from_template(spaces, template, ctx)
    batch = State(
        ctx.asarray([[1.0, 2.0], [3.0, 4.0]]),
        ctx.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
    )

    doubled = P.add_batch(batch, batch)
    assert isinstance(doubled, State)
    np.testing.assert_allclose(doubled.a, [[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(doubled.b, [[10.0, 12.0, 14.0], [16.0, 18.0, 20.0]])
    scaled = P.scale_batch(0.5, batch)
    assert isinstance(scaled, State)
    np.testing.assert_allclose(scaled.a, [[0.5, 1.0], [1.5, 2.0]])
    flat_batch = P.flatten_batch(batch)
    np.testing.assert_allclose(flat_batch, [[1.0, 2.0, 5.0, 6.0, 7.0], [3.0, 4.0, 8.0, 9.0, 10.0]])
    unflat = P.unflatten_batch(flat_batch)
    assert isinstance(unflat, State)
    np.testing.assert_allclose(unflat.b, batch.b)

    with pytest.raises(ValueError, match="Invalid batched product structure"):
        from spacecore._batching import _check_batched

        _check_batched(P, [State(batch.a[0], batch.b[0]), State(batch.a[1], batch.b[1])])


def test_solver_smoke_with_registered_pytree_product_space():
    ctx = _np_ctx()
    spaces = _spaces(ctx)
    P = sc.ProductSpace.from_template(spaces, _state(ctx), ctx)

    def identity(x):
        return x

    A = sc.MatrixFreeLinOp(identity, identity, P, P, ctx)
    b = _state(ctx)
    result = sc.cg(A, b, tol=1e-12, maxiter=5, check_every=1)

    assert bool(result.converged)
    _assert_state_allclose(result.x, [1.0, 2.0], [3.0, 4.0, 5.0])


def test_product_space_jax_pytree_registration_preserves_structure_static_aux():
    if not has_jax():
        pytest.skip("JAX is not available")
    ctx = _jax_ctx()
    template = _state(ctx)
    P = sc.ProductSpace.from_template(_spaces(ctx), template, ctx)

    leaves, treedef = jax.tree_util.tree_flatten(P)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert leaves == []
    assert rebuilt == P
    assert rebuilt.structure == P.structure
    assert not hasattr(rebuilt.structure, "_leaves")
