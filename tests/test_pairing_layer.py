"""S2: the metric-free pairing and the differential covector."""
import numpy as np
import pytest

import spacecore as sc


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _spaces(ctx, n=3):
    rng = np.random.default_rng(0)
    G = rng.standard_normal((n, n))
    M_np = G @ G.T + n * np.eye(n)
    E = sc.DenseCoordinateSpace((n,), ctx)
    M = sc.DenseLinOp(ctx.asarray(M_np), E, E, ctx)
    Minv = sc.DenseLinOp(ctx.asarray(np.linalg.inv(M_np)), E, E, ctx)
    return {
        "euclidean": E,
        "weighted": sc.DenseCoordinateSpace((n,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([1.0, 2.0, 5.0]))),
        "operator": sc.DenseCoordinateSpace((n,), ctx, geometry=sc.OperatorInnerProduct(M, Minv)),
    }


@pytest.mark.parametrize("kind", ["euclidean", "weighted", "operator"])
def test_pairing_is_metric_free_and_represents_the_inner_product(ctx, kind):
    X = _spaces(ctx)[kind]
    x, y = np.array([1.0, -2.0, 0.5]), np.array([0.3, 1.0, -1.0])
    assert X.pair(x, y) == pytest.approx(x @ y)
    assert X.real_inner(x, y) == pytest.approx(X.pair(X.riesz(x), y))
    xs = np.stack([y, 2 * y, -y])
    np.testing.assert_allclose(X.vpair(x, xs), xs @ x)


@pytest.mark.parametrize("kind", ["euclidean", "weighted", "operator"])
def test_differential_represents_the_derivative(ctx, kind):
    X = _spaces(ctx)[kind]
    x, h = np.array([1.0, -2.0, 0.5]), np.array([0.3, 1.0, -1.0])
    for f in (sc.SquaredL2NormFunctional(X), sc.InnerProductFunctional(ctx.asarray([1.0, 2.0, 3.0]), X)):
        df = f.differential(x)
        assert X.pair(df, h) == pytest.approx(X.real_inner(f.grad(x), h))
        np.testing.assert_allclose(df, X.riesz(f.grad(x)))
    # Another geometry turns the same covector into its own gradient.
    W = _spaces(ctx)["weighted"]
    f = sc.SquaredL2NormFunctional(X)
    g_W = W.geometry.riesz_inverse(W.ops, f.differential(x))
    assert W.real_inner(g_W, h) == pytest.approx(X.pair(f.differential(x), h))
