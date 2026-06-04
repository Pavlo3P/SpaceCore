from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import has_jax, jax_real_dtype, to_numpy

if has_jax():
    import jax

    _register_pytree = jax.tree_util.register_pytree_node_class
else:
    _register_pytree = lambda cls: cls


@_register_pytree
@dataclass(frozen=True)
class State:
    a: object
    b: object

    def tree_flatten(self):
        return (self.a, self.b), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def _ctx(kind: str):
    if kind == "jax":
        pytest.importorskip("jax")
        return sc.Context(sc.JaxOps(), dtype=jax_real_dtype())
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _assert_allclose(actual, expected):
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=1e-7, atol=1e-7)


def _assert_state_allclose(actual, expected_a, expected_b):
    assert isinstance(actual, State)
    _assert_allclose(actual.a, expected_a)
    _assert_allclose(actual.b, expected_b)


def _block_parts(ctx):
    x1 = sc.VectorSpace((2,), ctx)
    x2 = sc.VectorSpace((3,), ctx)
    y1 = sc.VectorSpace((2,), ctx)
    y2 = sc.VectorSpace((1,), ctx)
    return (
        sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), x1, y1, ctx),
        sc.DenseLinOp(ctx.asarray([[5.0, 6.0, 7.0]]), x2, y2, ctx),
    )


def _stacked_parts(ctx):
    x = sc.VectorSpace((2,), ctx)
    y1 = sc.VectorSpace((2,), ctx)
    y2 = sc.VectorSpace((1,), ctx)
    return (
        sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), x, y1, ctx),
        sc.DenseLinOp(ctx.asarray([[5.0, 6.0]]), x, y2, ctx),
    )


def _sum_parts(ctx):
    x1 = sc.VectorSpace((2,), ctx)
    x2 = sc.VectorSpace((3,), ctx)
    y = sc.VectorSpace((2,), ctx)
    return (
        sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), x1, y, ctx),
        sc.DenseLinOp(ctx.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), x2, y, ctx),
    )


def _structured_product(spaces, template, ctx):
    if not has_jax():
        pytest.skip("structured ProductSpace tests require JAX tree utilities")
    return sc.ProductSpace.from_template(tuple(spaces), template, ctx)


@pytest.mark.parametrize("kind", ["numpy", "jax"])
def test_block_diagonal_accepts_and_returns_structured_product_elements(kind):
    ctx = _ctx(kind)
    parts = _block_parts(ctx)
    dom = _structured_product(
        (parts[0].domain, parts[1].domain),
        State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0, 0.0, 0.0])),
        ctx,
    )
    cod = _structured_product(
        (parts[0].codomain, parts[1].codomain),
        State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0])),
        ctx,
    )
    op = sc.BlockDiagonalLinOp(dom, cod, parts, ctx)

    x = State(ctx.asarray([10.0, 20.0]), ctx.asarray([1.0, 2.0, 3.0]))
    y = op.apply(x)
    _assert_state_allclose(y, [50.0, 110.0], [38.0])

    xr = op.rapply(State(ctx.asarray([2.0, -1.0]), ctx.asarray([3.0])))
    _assert_state_allclose(xr, [-1.0, 0.0], [15.0, 18.0, 21.0])

    _assert_state_allclose(op.H.apply(y), [380.0, 540.0], [190.0, 228.0, 266.0])


def test_block_diagonal_tuple_default_remains_tuple():
    ctx = _ctx("numpy")
    op = sc.BlockDiagonalLinOp.from_operators(_block_parts(ctx))

    y = op.apply((ctx.asarray([10.0, 20.0]), ctx.asarray([1.0, 2.0, 3.0])))
    assert isinstance(y, tuple)
    _assert_allclose(y[0], [50.0, 110.0])
    _assert_allclose(y[1], [38.0])

    x = op.rapply((ctx.asarray([2.0, -1.0]), ctx.asarray([3.0])))
    assert isinstance(x, tuple)
    _assert_allclose(x[0], [-1.0, 0.0])
    _assert_allclose(x[1], [15.0, 18.0, 21.0])


@pytest.mark.parametrize("kind", ["numpy", "jax"])
def test_stacked_linop_returns_structured_codomain_and_accepts_it_for_adjoint(kind):
    ctx = _ctx(kind)
    parts = _stacked_parts(ctx)
    cod = _structured_product(
        (parts[0].codomain, parts[1].codomain),
        State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0])),
        ctx,
    )
    op = sc.StackedLinOp(parts[0].domain, cod, parts, ctx)

    y = op.apply(ctx.asarray([10.0, 20.0]))
    _assert_state_allclose(y, [50.0, 110.0], [170.0])

    x = op.rapply(State(ctx.asarray([2.0, -1.0]), ctx.asarray([3.0])))
    _assert_allclose(x, [14.0, 18.0])
    _assert_allclose(op.H.apply(y), [1230.0, 1560.0])


@pytest.mark.parametrize("kind", ["numpy", "jax"])
def test_sum_to_single_accepts_structured_domain_and_returns_it_for_adjoint(kind):
    ctx = _ctx(kind)
    parts = _sum_parts(ctx)
    dom = _structured_product(
        (parts[0].domain, parts[1].domain),
        State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0, 0.0, 0.0])),
        ctx,
    )
    op = sc.SumToSingleLinOp(dom, parts[0].codomain, parts, ctx)

    x = State(ctx.asarray([10.0, 20.0]), ctx.asarray([1.0, 2.0, 3.0]))
    y = op.apply(x)
    _assert_allclose(y, [88.0, 166.0])

    xr = op.rapply(ctx.asarray([2.0, -1.0]))
    _assert_state_allclose(xr, [-1.0, 0.0], [2.0, 3.0, 4.0])
    _assert_state_allclose(op.H.apply(y), [586.0, 840.0], [1768.0, 2022.0, 2276.0])


@pytest.mark.parametrize("kind", ["numpy", "jax"])
def test_product_linop_batch_paths_use_structure_of_batched_components(kind):
    ctx = _ctx(kind)

    block_parts = _block_parts(ctx)
    block_dom = _structured_product(
        (block_parts[0].domain, block_parts[1].domain),
        State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0, 0.0, 0.0])),
        ctx,
    )
    block_cod = _structured_product(
        (block_parts[0].codomain, block_parts[1].codomain),
        State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0])),
        ctx,
    )
    block = sc.BlockDiagonalLinOp(block_dom, block_cod, block_parts, ctx)
    xb = State(
        ctx.asarray([[10.0, 20.0], [1.0, 2.0]]),
        ctx.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    _assert_state_allclose(block.vapply(xb), [[50.0, 110.0], [5.0, 11.0]], [[38.0], [92.0]])
    yb = State(ctx.asarray([[2.0, -1.0], [1.0, 1.0]]), ctx.asarray([[3.0], [2.0]]))
    _assert_state_allclose(
        block.rvapply(yb),
        [[-1.0, 0.0], [4.0, 6.0]],
        [[15.0, 18.0, 21.0], [10.0, 12.0, 14.0]],
    )

    stacked_parts = _stacked_parts(ctx)
    stacked_cod = _structured_product(
        (stacked_parts[0].codomain, stacked_parts[1].codomain),
        State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0])),
        ctx,
    )
    stacked = sc.StackedLinOp(stacked_parts[0].domain, stacked_cod, stacked_parts, ctx)
    _assert_state_allclose(
        stacked.vapply(ctx.asarray([[10.0, 20.0], [1.0, 2.0]])),
        [[50.0, 110.0], [5.0, 11.0]],
        [[170.0], [17.0]],
    )
    _assert_allclose(
        stacked.rvapply(
            State(
                ctx.asarray([[2.0, -1.0], [1.0, 2.0]]),
                ctx.asarray([[3.0], [-1.0]]),
            )
        ),
        [[14.0, 18.0], [2.0, 4.0]],
    )

    sum_parts = _sum_parts(ctx)
    sum_dom = _structured_product(
        (sum_parts[0].domain, sum_parts[1].domain),
        State(ctx.asarray([0.0, 0.0]), ctx.asarray([0.0, 0.0, 0.0])),
        ctx,
    )
    summed = sc.SumToSingleLinOp(sum_dom, sum_parts[0].codomain, sum_parts, ctx)
    _assert_allclose(summed.vapply(xb), [[88.0, 166.0], [97.0, 148.0]])
    _assert_state_allclose(
        summed.rvapply(ctx.asarray([[2.0, -1.0], [1.0, 2.0]])),
        [[-1.0, 0.0], [7.0, 10.0]],
        [[2.0, 3.0, 4.0], [21.0, 24.0, 27.0]],
    )
