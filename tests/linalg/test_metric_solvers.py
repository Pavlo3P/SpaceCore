import importlib

import numpy as np

from tests._helpers import to_numpy


sc = importlib.import_module("spacecore")


def _weighted_space(ctx):
    return sc.VectorSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0, 11.0])))


def _metric_spd_matrix(ctx):
    weights = np.asarray([2.0, 5.0, 11.0])
    symmetric_spd = np.asarray(
        [
            [6.0, 1.0, 0.5],
            [1.0, 5.0, -0.25],
            [0.5, -0.25, 4.0],
        ]
    )
    return ctx.asarray(symmetric_spd / weights[:, None])


def test_cg_uses_weighted_inner_products_and_residual_norms():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = _weighted_space(ctx)
    A = sc.DenseLinOp(_metric_spd_matrix(ctx), space, space, ctx)
    x_true = ctx.asarray([1.0, -2.0, 0.5])
    b = A.apply(x_true)

    result = sc.cg(A, b, tol=1e-12, maxiter=8, check_every=2)
    residual = space.add(A.apply(result.x), space.scale(-1.0, b))

    np.testing.assert_allclose(to_numpy(result.x), to_numpy(x_true), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(to_numpy(result.residual_norm), to_numpy(space.norm(residual)), atol=1e-12)
    assert bool(result.converged)


def test_power_iteration_uses_metric_rayleigh_quotient_and_norm():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = _weighted_space(ctx)
    matrix = _metric_spd_matrix(ctx)
    A = sc.DenseLinOp(matrix, space, space, ctx)
    x0 = ctx.asarray([1.0, 0.25, -0.5])

    result = sc.power_iteration(A, x0=x0, tol=1e-10, maxiter=80)
    Ax = A.apply(result.eigenvector)
    rayleigh = space.inner(result.eigenvector, Ax) / space.inner(result.eigenvector, result.eigenvector)
    residual = space.add(Ax, space.scale(-result.eigenvalue, result.eigenvector))
    expected = max(np.linalg.eigvals(to_numpy(matrix)).real)

    np.testing.assert_allclose(to_numpy(result.eigenvalue), to_numpy(rayleigh), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(to_numpy(result.eigenvalue), expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(to_numpy(result.residual_norm), to_numpy(space.norm(residual)), atol=1e-12)
    assert bool(result.converged)
