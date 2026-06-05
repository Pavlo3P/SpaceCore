import importlib
import warnings

import numpy as np
import pytest

from tests._helpers import has_jax, jax_real_dtype, to_numpy


def _contexts():
    sc = importlib.import_module("spacecore")
    yield pytest.param(sc.Context(sc.NumpyOps(), dtype=np.float64), 1e-6, 2e-5, id="numpy")
    yield pytest.param(
        sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=False),
        1e-3,
        5e-3,
        marks=pytest.mark.skipif(not has_jax(), reason="jax is not installed"),
        id="jax",
    )


def _weighted_geometry(weights):
    sc = importlib.import_module("spacecore")
    return sc.WeightedInnerProduct(weights)


def _weighted_space(ctx):
    weights = ctx.asarray([2.0, 5.0, 11.0])
    sc = importlib.import_module("spacecore")
    return sc.DenseCoordinateSpace((3,), ctx, geometry=_weighted_geometry(weights))


def _weighted_vector_space(ctx, weights):
    sc = importlib.import_module("spacecore")
    weights = ctx.asarray(weights)
    return sc.DenseCoordinateSpace(tuple(to_numpy(weights).shape), ctx, geometry=_weighted_geometry(weights))


def _self_adjoint_metric_matrix(ctx):
    weights = np.asarray([2.0, 5.0, 11.0])
    symmetric = np.asarray(
        [
            [4.0, 1.0, -0.5],
            [1.0, 6.0, 2.0],
            [-0.5, 2.0, 3.0],
        ]
    )
    return ctx.asarray(symmetric / weights[:, None])


def _finite_difference(functional, x, v, eps):
    xp = x + eps * v
    xm = x - eps * v
    return (functional.value(xp) - functional.value(xm)) / (2.0 * eps)


def _assert_gradient_identity(functional, x, v, eps, atol):
    grad = functional.grad(x)
    lhs = functional.domain.inner(grad, v)
    rhs = _finite_difference(functional, x, v, eps)
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=5e-3, atol=atol)


@pytest.mark.parametrize("ctx,eps,atol", list(_contexts()))
def test_linop_quadratic_gradient_identity_on_weighted_space(ctx, eps, atol):
    sc = importlib.import_module("spacecore")
    space = _weighted_space(ctx)
    Q = sc.DenseLinOp(_self_adjoint_metric_matrix(ctx), space, space, ctx)
    c = ctx.asarray([0.25, -1.5, 2.0])
    functional = sc.LinOpQuadraticForm(Q, sc.InnerProductFunctional(c, space, ctx), 1.25, ctx)
    x = ctx.asarray([0.5, -1.0, 2.0])
    v = ctx.asarray([1.25, 0.75, -0.5])

    _assert_gradient_identity(functional, x, v, eps, atol)


@pytest.mark.parametrize("ctx,eps,atol", list(_contexts()))
def test_inner_product_functional_gradient_identity_on_weighted_space(ctx, eps, atol):
    sc = importlib.import_module("spacecore")
    space = _weighted_space(ctx)
    c = ctx.asarray([0.25, -1.5, 2.0])
    functional = sc.InnerProductFunctional(c, space, ctx)
    x = ctx.asarray([0.5, -1.0, 2.0])
    v = ctx.asarray([1.25, 0.75, -0.5])

    _assert_gradient_identity(functional, x, v, eps, atol)
    np.testing.assert_allclose(to_numpy(functional.grad(x)), to_numpy(c))


def test_euclidean_quadratic_gradient_behavior_is_unchanged():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseCoordinateSpace((2,), ctx)
    Q = sc.DenseLinOp(ctx.asarray([[2.0, 1.0], [1.0, 4.0]]), space, space, ctx)
    c = ctx.asarray([1.0, -1.0])
    functional = sc.LinOpQuadraticForm(Q, sc.InnerProductFunctional(c, space, ctx), 3.0, ctx)
    x = ctx.asarray([2.0, -1.0])

    np.testing.assert_allclose(to_numpy(functional.grad(x)), [4.0, -3.0])
    np.testing.assert_allclose(to_numpy(functional.hess_apply(x)), [3.0, -2.0])


@pytest.mark.parametrize("ctx,eps,atol", list(_contexts()))
def test_inner_product_functional_compose_uses_metric_adjoint_pullback(ctx, eps, atol):
    sc = importlib.import_module("spacecore")
    domain = _weighted_vector_space(ctx, [2.0, 5.0])
    codomain = _weighted_vector_space(ctx, [3.0, 7.0, 11.0])
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    A = sc.DenseLinOp(matrix, domain, codomain, ctx)
    c = ctx.asarray([0.25, -1.5, 2.0])
    functional = sc.InnerProductFunctional(c, codomain, ctx)
    composed = functional.compose(A)
    x = ctx.asarray([0.5, -1.0])
    v = ctx.asarray([1.25, 0.75])

    np.testing.assert_allclose(
        to_numpy(composed.value(x)),
        to_numpy(functional.value(A.apply(x))),
        rtol=5e-6,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(composed.representer),
        to_numpy(A.H.apply(c)),
        rtol=5e-6,
        atol=atol,
    )
    _assert_gradient_identity(composed, x, v, eps, atol)


@pytest.mark.parametrize("ctx,eps,atol", list(_contexts()))
@pytest.mark.parametrize("weighted", [False, True])
def test_inner_product_functional_vectorized_batches_match_elementwise(ctx, eps, atol, weighted):
    sc = importlib.import_module("spacecore")
    space = _weighted_space(ctx) if weighted else sc.DenseCoordinateSpace((3,), ctx)
    c = ctx.asarray([0.25, -1.5, 2.0])
    functional = sc.InnerProductFunctional(c, space, ctx)
    xs = ctx.asarray(
        [
            [0.5, -1.0, 2.0],
            [1.25, 0.75, -0.5],
            [-2.0, 0.25, 1.5],
        ]
    )

    expected_values = functional.ops.stack(tuple(functional.value(x) for x in xs), axis=0)
    expected_grads = functional.ops.stack(tuple(functional.grad(x) for x in xs), axis=0)

    np.testing.assert_allclose(to_numpy(functional.vvalue(xs)), to_numpy(expected_values), rtol=5e-6, atol=atol)
    np.testing.assert_allclose(to_numpy(functional.vgrad(xs)), to_numpy(expected_grads), rtol=5e-6, atol=atol)


@pytest.mark.parametrize("ctx,eps,atol", list(_contexts()))
@pytest.mark.parametrize("weighted", [False, True])
def test_linop_quadratic_form_vectorized_batches_match_elementwise(ctx, eps, atol, weighted):
    sc = importlib.import_module("spacecore")
    space = _weighted_space(ctx) if weighted else sc.DenseCoordinateSpace((3,), ctx)
    matrix = _self_adjoint_metric_matrix(ctx) if weighted else ctx.asarray(
        [
            [4.0, 1.0, -0.5],
            [1.0, 6.0, 2.0],
            [-0.5, 2.0, 3.0],
        ]
    )
    Q = sc.DenseLinOp(matrix, space, space, ctx)
    linear = sc.InnerProductFunctional(ctx.asarray([0.25, -1.5, 2.0]), space, ctx)
    functional = sc.LinOpQuadraticForm(Q, linear, 1.25, ctx)
    xs = ctx.asarray(
        [
            [0.5, -1.0, 2.0],
            [1.25, 0.75, -0.5],
            [-2.0, 0.25, 1.5],
        ]
    )

    expected_values = functional.ops.stack(tuple(functional.value(x) for x in xs), axis=0)
    expected_grads = functional.ops.stack(tuple(functional.grad(x) for x in xs), axis=0)

    np.testing.assert_allclose(to_numpy(functional.vvalue(xs)), to_numpy(expected_values), rtol=5e-6, atol=atol)
    np.testing.assert_allclose(to_numpy(functional.vgrad(xs)), to_numpy(expected_grads), rtol=5e-6, atol=atol)


def test_matrix_free_functional_vvalue_python_loop_warns_once_on_numpy():
    sc = importlib.import_module("spacecore")
    base = importlib.import_module("spacecore.functional._base")
    base._VMAP_FALLBACK_WARNED.clear()
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    space = sc.DenseCoordinateSpace((2,), ctx)
    c = ctx.asarray([1.0, -2.0])
    functional = sc.MatrixFreeLinearFunctional(lambda x: space.inner(c, x), space, ctx)
    xs = ctx.asarray(np.arange(80.0).reshape(40, 2))

    with pytest.warns(RuntimeWarning, match="falls back to a Python loop"):
        functional.vvalue(xs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        functional.vvalue(xs)
    assert caught == []


def test_vectorized_functionals_do_not_warn_on_numpy():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    space = sc.DenseCoordinateSpace((2,), ctx)
    c = ctx.asarray([1.0, -2.0])
    functional = sc.InnerProductFunctional(c, space, ctx)
    xs = ctx.asarray(np.arange(80.0).reshape(40, 2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        functional.vvalue(xs)
        functional.vgrad(xs)
    assert caught == []


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_matrix_free_functional_vvalue_does_not_warn_on_native_vmap_backend():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=False)
    space = sc.DenseCoordinateSpace((2,), ctx)
    c = ctx.asarray([1.0, -2.0])
    functional = sc.MatrixFreeLinearFunctional(lambda x: space.inner(c, x), space, ctx)
    xs = ctx.asarray(np.arange(80.0).reshape(40, 2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        functional.vvalue(xs)
    assert caught == []
