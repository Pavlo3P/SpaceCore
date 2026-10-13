"""S3: dual norms, dual geometries and metric factors."""
import numpy as np
import pytest

import spacecore as sc

N = 3


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _M(n=N):
    G = np.random.default_rng(0).standard_normal((n, n))
    return G @ G.T + n * np.eye(n)


def _op(ctx, A):
    E = sc.DenseCoordinateSpace((N,), ctx)
    return sc.DenseLinOp(ctx.asarray(A), E, E, ctx)


def _geometries(ctx):
    M = _M()
    return {
        "euclidean": (sc.EuclideanInnerProduct(), np.eye(N)),
        "weighted": (sc.WeightedInnerProduct(ctx.asarray([1.0, 2.0, 5.0])), np.diag([1.0, 2.0, 5.0])),
        "operator": (sc.OperatorInnerProduct(_op(ctx, M), _op(ctx, np.linalg.inv(M)),
                                             factor=_op(ctx, np.linalg.cholesky(M).T),
                                             factor_inverse=_op(ctx, np.linalg.inv(np.linalg.cholesky(M).T))), M),
    }


@pytest.mark.parametrize("kind", ["euclidean", "weighted", "operator"])
def test_dual_norm_and_dual_geometry(ctx, kind):
    G, R = _geometries(ctx)[kind]
    X = sc.DenseCoordinateSpace((N,), ctx, geometry=G)
    xi, x = np.array([1.0, -2.0, 0.5]), np.array([0.3, 1.0, -1.0])
    expected = np.sqrt(xi @ np.linalg.solve(R, xi))
    assert X.dual_norm(xi) == pytest.approx(expected)
    assert G.dual_norm(X.ops, xi) == pytest.approx(expected)
    # The dual norm is the sup of the pairing over the unit ball, attained at R^{-1} xi.
    x_star = np.linalg.solve(R, xi)
    assert X.pair(xi, x_star / X.norm(x_star)) == pytest.approx(expected)
    assert X.dual_norm(X.riesz(x)) == pytest.approx(X.norm(x))
    assert G.dual().norm(X.ops, xi) == pytest.approx(expected)
    assert G.norm(X.ops, x) == pytest.approx(X.norm(x))


@pytest.mark.parametrize("kind", ["euclidean", "weighted", "operator"])
def test_metric_factor_is_an_isometry(ctx, kind):
    G, _ = _geometries(ctx)[kind]
    X = sc.DenseCoordinateSpace((N,), ctx, geometry=G, check_level="strict")
    x = np.array([0.3, 1.0, -1.0])
    xhat = G.factor(X.ops, x)
    assert np.linalg.norm(xhat) == pytest.approx(X.norm(x))
    np.testing.assert_allclose(G.factor_inverse(X.ops, xhat), x)


def test_wrong_factor_is_caught_at_strict(ctx):
    M = _M()
    G = sc.OperatorInnerProduct(_op(ctx, M), factor=_op(ctx, np.eye(N)))
    with pytest.raises(ValueError, match="factor"):
        sc.DenseCoordinateSpace((N,), ctx, geometry=G, check_level="strict")
    sc.DenseCoordinateSpace((N,), ctx, geometry=G, check_level="standard")


def test_dual_needs_the_inverse(ctx):
    M = _M()
    G = sc.OperatorInnerProduct(_op(ctx, M), _op(ctx, np.linalg.inv(M)))
    xi = np.array([1.0, -2.0, 0.5])
    # The dual geometry is an ordinary metric on covectors.
    covectors = sc.DenseCoordinateSpace((N,), ctx, geometry=G.dual())
    assert covectors.norm(xi) == pytest.approx(np.sqrt(xi @ np.linalg.solve(M, xi)))
    with pytest.raises(sc.CapabilityError, match="M_inverse"):
        sc.OperatorInnerProduct(_op(ctx, M)).dual()
    with pytest.raises(TypeError):
        sc.OperatorInnerProduct()


def test_conversion_keeps_factors(ctx):
    G, _ = _geometries(ctx)["operator"]
    X = sc.DenseCoordinateSpace((N,), ctx, geometry=G)
    Y = X.convert(sc.Context(sc.NumpyOps(), dtype=np.float32))
    assert Y.geometry.factor_op.dtype == np.dtype("float32")
    assert X == sc.DenseCoordinateSpace((N,), ctx, geometry=_geometries(ctx)["operator"][0])
