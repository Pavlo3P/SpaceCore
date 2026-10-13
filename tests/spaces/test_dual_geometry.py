"""The dual norm of a covector."""
import numpy as np
import pytest

import spacecore as sc

N = 3


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _geometries(ctx):
    G = np.random.default_rng(0).standard_normal((N, N))
    M = G @ G.T + N * np.eye(N)
    E = sc.DenseCoordinateSpace((N,), ctx)
    op = lambda A: sc.DenseLinOp(ctx.asarray(A), E, E, ctx)  # noqa: E731
    return {
        "euclidean": (sc.EuclideanInnerProduct(), np.eye(N)),
        "weighted": (sc.WeightedInnerProduct(ctx.asarray([1.0, 2.0, 5.0])), np.diag([1.0, 2.0, 5.0])),
        "operator": (sc.OperatorInnerProduct(op(M), op(np.linalg.inv(M))), M),
    }


@pytest.mark.parametrize("kind", ["euclidean", "weighted", "operator"])
def test_dual_norm(ctx, kind):
    G, R = _geometries(ctx)[kind]
    X = sc.DenseCoordinateSpace((N,), ctx, geometry=G)
    xi, x = np.array([1.0, -2.0, 0.5]), np.array([0.3, 1.0, -1.0])
    expected = np.sqrt(xi @ np.linalg.solve(R, xi))
    assert X.dual_norm(xi) == pytest.approx(expected)
    # The dual norm is the sup of the pairing over the unit ball, attained at R^{-1} xi.
    x_star = np.linalg.solve(R, xi)
    assert X.pair(xi, x_star / X.norm(x_star)) == pytest.approx(expected)
    assert X.dual_norm(X.riesz(x)) == pytest.approx(X.norm(x))


def test_operator_metric_needs_M_and_converts(ctx):
    with pytest.raises(TypeError):
        sc.OperatorInnerProduct()
    G, _ = _geometries(ctx)["operator"]
    X = sc.DenseCoordinateSpace((N,), ctx, geometry=G)
    Y = X.convert(sc.Context(sc.NumpyOps(), dtype=np.float32))
    assert Y.geometry.M_inverse.dtype == np.dtype("float32")
    assert X == sc.DenseCoordinateSpace((N,), ctx, geometry=_geometries(ctx)["operator"][0])
