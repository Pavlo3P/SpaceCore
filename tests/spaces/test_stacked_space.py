import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, jax_real_dtype, to_numpy


def _contexts():
    sc = importlib.import_module("spacecore")
    yield pytest.param(sc.Context(sc.NumpyOps(), dtype=np.float64), id="numpy")
    yield pytest.param(
        sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=False),
        marks=pytest.mark.skipif(not has_jax(), reason="jax is not installed"),
        id="jax",
    )


@pytest.mark.parametrize("ctx", list(_contexts()))
def test_stacked_space_core_contract_euclidean(ctx):
    sc = importlib.import_module("spacecore")
    base = sc.VectorSpace((3,), ctx)
    space = sc.StackedSpace(base, 4, ctx)
    x = ctx.asarray(np.arange(12.0).reshape(4, 3))
    y = ctx.asarray(np.ones((4, 3)))

    assert space.shape == (4, 3)
    assert space.count == 4
    assert space.base == base
    np.testing.assert_allclose(to_numpy(space.inner(x, y)), np.vdot(to_numpy(x), to_numpy(y)))
    np.testing.assert_allclose(to_numpy(space.norm(x)), np.linalg.norm(to_numpy(x).reshape(-1)))
    np.testing.assert_allclose(to_numpy(space.add(x, y)), to_numpy(x + y))
    np.testing.assert_allclose(to_numpy(space.scale(2.0, x)), to_numpy(2.0 * x))
    np.testing.assert_allclose(to_numpy(space.zeros()), np.zeros((4, 3)))
    np.testing.assert_allclose(to_numpy(space.ones()), np.ones((4, 3)))
    np.testing.assert_allclose(to_numpy(space.unflatten(space.flatten(x))), to_numpy(x))


@pytest.mark.parametrize("ctx", list(_contexts()))
def test_stacked_space_weighted_geometry_lifts_elementwise(ctx):
    sc = importlib.import_module("spacecore")
    weights = ctx.asarray([2.0, 5.0, 11.0])
    base = sc.VectorSpace((3,), ctx, geometry=sc.WeightedInnerProduct(weights))
    space = base.stacked(2)
    x = ctx.asarray([[1.0, -2.0, 0.5], [3.0, 1.5, -1.0]])
    y = ctx.asarray([[0.25, 2.0, -1.0], [-2.0, 0.5, 4.0]])

    expected_inner = np.vdot(to_numpy(x), to_numpy(weights) * to_numpy(y))
    np.testing.assert_allclose(to_numpy(space.inner(x, y)), expected_inner)
    np.testing.assert_allclose(to_numpy(space.riesz(x)), to_numpy(weights) * to_numpy(x))
    np.testing.assert_allclose(to_numpy(space.riesz_inverse(space.riesz(x))), to_numpy(x))
    assert space.is_euclidean is False


def test_stacked_space_equality_conversion_and_composition():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    base = sc.VectorSpace((2,), src)
    a = base.stacked(3)
    b = sc.StackedSpace(sc.VectorSpace((2,), src), 3, src)
    c = base.stacked(4)

    assert a == b
    assert a != c
    converted = a.convert(dst)
    assert converted.ctx == dst
    assert converted.base.ctx == dst
    assert converted.shape == (3, 2)
    assert a.stacked(5) == sc.StackedSpace(base, 15, src)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_stacked_space_jax_pytree_roundtrip():
    import jax

    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=False)
    space = sc.VectorSpace((2,), ctx).stacked(3)
    leaves, treedef = jax.tree_util.tree_flatten(space)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert leaves == []
    assert rebuilt == space


def test_product_space_stacked_nests_products_outside_stacks():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    x = sc.VectorSpace((2,), ctx)
    y = sc.VectorSpace((3,), ctx)
    product = sc.ProductSpace((x, y), ctx)
    stacked = product.stacked(4)

    assert isinstance(stacked, sc.ProductSpace)
    assert all(isinstance(s, sc.StackedSpace) for s in stacked.spaces)
    assert stacked.spaces[0].shape == (4, 2)
    assert stacked.spaces[1].shape == (4, 3)
    with pytest.raises(TypeError, match="ProductSpace"):
        sc.StackedSpace(product, 4, ctx)


@pytest.mark.parametrize("ctx", list(_contexts()))
def test_operator_vapply_on_stacked_element_and_adjoint_identity(ctx):
    sc = importlib.import_module("spacecore")
    weights = ctx.asarray([2.0, 5.0])
    base = sc.VectorSpace((2,), ctx, geometry=sc.WeightedInnerProduct(weights))
    stacked = base.stacked(3)
    matrix = ctx.asarray([[3.0, 0.0], [0.0, 7.0]])
    op = sc.DenseLinOp(matrix, base, base, ctx)
    xs = ctx.asarray([[1.0, -2.0], [3.0, 4.0], [-1.0, 0.5]])
    ys = ctx.asarray([[0.25, 2.0], [-2.0, 0.5], [1.5, -1.0]])

    axs = op.vapply(xs)
    ahys = op.rvapply(ys)
    expected = np.stack([to_numpy(op.apply(x)) for x in xs], axis=0)
    np.testing.assert_allclose(to_numpy(axs), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        to_numpy(stacked.inner(axs, ys)),
        to_numpy(stacked.inner(xs, ahys)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_cg_solves_diagonal_operator_on_stacked_space():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    base = sc.VectorSpace((2,), ctx)
    stacked = base.stacked(3)
    diagonal = ctx.asarray([[2.0, 4.0], [3.0, 5.0], [7.0, 11.0]])
    op = sc.DiagonalLinOp(diagonal, stacked, ctx)
    b = ctx.asarray([[2.0, 8.0], [9.0, 20.0], [14.0, 44.0]])

    result = sc.cg(op, b, tol=1e-12, maxiter=8, check_every=1)

    expected = to_numpy(b) / to_numpy(diagonal)
    np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-8, atol=1e-8)
    residual = b - op.apply(result.x)
    np.testing.assert_allclose(to_numpy(result.residual_norm), to_numpy(stacked.norm(residual)), atol=1e-12)
