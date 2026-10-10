"""S4: a geometry passed per call — preconditioned CG and ``adjoint_in``."""
import numpy as np
import pytest

import spacecore as sc
from spacecore.linalg import cg

N = 6


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _spd(seed, n=N, shift=1.0):
    G = np.random.default_rng(seed).standard_normal((n, n))
    return G @ G.T + shift * np.eye(n)


def _operator_geometry(ctx, E, M_np, *, metric=True):
    M = sc.DenseLinOp(ctx.asarray(M_np), E, E, ctx) if metric else None
    Minv = sc.DenseLinOp(ctx.asarray(np.linalg.inv(M_np)), E, E, ctx)
    return sc.OperatorInnerProduct(M, Minv)


@pytest.mark.parametrize("metric", [True, False], ids=["metric", "cometric"])
def test_preconditioned_cg_solves_and_saves_iterations(ctx, metric):
    E = sc.DenseCoordinateSpace((N,), ctx)
    A_np = _spd(0) @ np.diag(np.logspace(0, 4, N)) @ _spd(0).T + np.eye(N)
    A = sc.DenseLinOp(ctx.asarray(A_np), E, E, ctx)
    b = np.arange(1.0, N + 1)
    # The exact preconditioner A^{-1}: PCG must converge in one step.
    G = _operator_geometry(ctx, E, A_np, metric=metric)
    pre = cg(A, b, geometry=G, tol=1e-10, maxiter=50, check_every=1)
    plain = cg(A, b, tol=1e-10, maxiter=50, check_every=1)
    np.testing.assert_allclose(pre.x, np.linalg.solve(A_np, b), rtol=1e-8)
    assert bool(pre.converged) and int(pre.num_iters) <= 2 < int(plain.num_iters)
    np.testing.assert_allclose(pre.residual_norm, np.linalg.norm(b - A_np @ pre.x), atol=1e-8)


def test_cg_with_space_own_geometry_matches_plain_cg(ctx):
    E = sc.DenseCoordinateSpace((N,), ctx)
    A_np = _spd(1)
    A = sc.DenseLinOp(ctx.asarray(A_np), E, E, ctx)
    b = np.ones(N)
    a = cg(A, b, tol=1e-12, maxiter=40, check_every=1)
    g = cg(A, b, geometry=E.geometry, tol=1e-12, maxiter=40, check_every=1)
    np.testing.assert_allclose(a.x, g.x)
    assert int(a.num_iters) == int(g.num_iters)


def test_adjoint_in_satisfies_the_geometry_adjoint_identity(ctx):
    X = sc.DenseCoordinateSpace((N,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(np.arange(1.0, N + 1))))
    Y = sc.DenseCoordinateSpace((3,), ctx)
    A_np = np.random.default_rng(2).standard_normal((3, N))
    A = sc.DenseLinOp(ctx.asarray(A_np), X, Y, ctx)
    gx = _operator_geometry(ctx, sc.DenseCoordinateSpace((N,), ctx), _spd(3))
    gy = sc.WeightedInnerProduct(ctx.asarray([2.0, 3.0, 7.0]))
    B = A.adjoint_in(gx, gy)
    x, y = np.linspace(-1, 1, N), np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(gy.inner(ctx.ops, A.apply(x), y), gx.inner(ctx.ops, x, B.apply(y)))
    # B's own adjoint is A again, in the same pair of geometries.
    np.testing.assert_allclose(B.adjoint_in(gy, gx).apply(x), A.apply(x))
    # With the spaces' own geometries it is the metric adjoint.
    np.testing.assert_allclose(A.adjoint_in(X.geometry, Y.geometry).apply(y), A.rapply(y))
    # rapply is B's metric adjoint in the spaces' own geometries.
    np.testing.assert_allclose(X.inner(B.apply(y), x), Y.inner(y, B.rapply(x)))


def test_prox_in_a_per_call_weighted_geometry(ctx):
    E = sc.DenseCoordinateSpace((N,), ctx)
    w = np.arange(1.0, N + 1)
    G = sc.WeightedInnerProduct(ctx.asarray(w))
    x = np.linspace(-3, 3, N)
    l1 = sc.LpNormFunctional(E, p=1)
    # argmin |y|_1 + sum w (y - x)^2 / (2 s): soft threshold at s / w.
    expected = np.sign(x) * np.maximum(np.abs(x) - 0.5 / w, 0)
    np.testing.assert_allclose(l1.prox(x, 0.5, geometry=G), expected)
    np.testing.assert_allclose((2.0 * l1).prox(x, 0.25, geometry=G), expected)
    # Moreau: the conjugate prox in G is the projection onto the G-dual unit ball.
    np.testing.assert_allclose(l1.conjugate_prox(x, 0.5, geometry=G), x - 0.5 * l1.prox(x / 0.5, 2.0, geometry=G))
    orthant = sc.IndicatorFunctional(sc.NonnegativeOrthant(E))
    np.testing.assert_allclose(orthant.prox(x, 1.0, geometry=G), np.maximum(x, 0))


def test_prox_refuses_a_foreign_geometry_without_a_closed_form(ctx):
    E = sc.DenseCoordinateSpace((N,), ctx)
    G = sc.WeightedInnerProduct(ctx.asarray(np.arange(1.0, N + 1)))
    x = np.ones(N)
    with pytest.raises(sc.CapabilityError):
        sc.SquaredL2NormFunctional(E).prox(x, 1.0, geometry=G)
    np.testing.assert_allclose(sc.SquaredL2NormFunctional(E).prox(x, 1.0, geometry=E.geometry), x / 2)
    with pytest.raises(sc.CapabilityError):
        sc.NonnegativeOrthant(E).project(x, geometry=_operator_geometry(ctx, E, _spd(4)))


@pytest.fixture
def generalized(ctx):
    """``A = M^{-1} K`` is self-adjoint in the ``M`` metric, not in the Euclidean one."""
    E = sc.DenseCoordinateSpace((N,), ctx)
    K = _spd(5, shift=0.1)
    M_np = _spd(6)
    A = sc.DenseLinOp(ctx.asarray(np.linalg.solve(M_np, K)), E, E, ctx, check_level="standard")
    eigenvalues, vectors = np.linalg.eig(np.linalg.solve(M_np, K))
    order = np.argsort(eigenvalues.real)
    return A, _operator_geometry(ctx, E, M_np), eigenvalues.real[order], vectors.real[:, order]


def test_eigensolvers_solve_the_generalized_problem(generalized):
    from spacecore.linalg import lanczos_smallest, power_iteration

    A, G, eigenvalues, _ = generalized
    with pytest.raises(ValueError):
        power_iteration(A)
    small = lanczos_smallest(A, np.ones(N), geometry=G, max_iter=N, tol=1e-10)
    np.testing.assert_allclose(small.eigenvalue, eigenvalues[0], rtol=1e-8)
    large = power_iteration(A, geometry=G, maxiter=3000, tol=1e-9)
    np.testing.assert_allclose(large.eigenvalue, eigenvalues[-1], rtol=1e-6)


def test_expm_multiply_in_the_metric_of_self_adjointness(generalized):
    from spacecore.linalg import expm_multiply

    A, G, eigenvalues, V = generalized
    v = np.linspace(1, 2, N)
    expected = V @ (np.exp(0.1 * eigenvalues) * np.linalg.solve(V, v))
    np.testing.assert_allclose(expm_multiply(A, v, 0.1, geometry=G, max_iter=N).result, expected, rtol=1e-8)


def test_lsqr_minimizes_in_the_codomain_geometry(ctx):
    from spacecore.linalg import lsqr

    X, Y = sc.DenseCoordinateSpace((3,), ctx), sc.DenseCoordinateSpace((N,), ctx)
    A_np = np.random.default_rng(7).standard_normal((N, 3))
    A = sc.DenseLinOp(ctx.asarray(A_np), X, Y, ctx)
    b = np.arange(1.0, N + 1)
    w = np.arange(1.0, N + 1) ** 2
    expected = np.linalg.solve(A_np.T @ (w[:, None] * A_np), A_np.T @ (w * b))
    for gx in (None, _operator_geometry(ctx, X, _spd(8, n=3))):
        result = lsqr(A, b, domain_geometry=gx, codomain_geometry=sc.WeightedInnerProduct(ctx.asarray(w)),
                      tol=1e-12, maxiter=50)
        np.testing.assert_allclose(result.x, expected, rtol=1e-7)
