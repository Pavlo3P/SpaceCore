"""A problem and algorithm definitions survive a change of native storage."""
import numpy as np
import pytest

from examples.structured_optimization import (
    Coefficients, admm, build_problem, pdhg, prepare_banded_solve,
    prepare_cg, proximal_gradient,
)


def coordinates(space, x, count):
    """Explicit test-only extraction; never used by an algorithm."""
    return np.array([space.component(x, i) for i in range(count)])


@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_problem_algebra_adjoint_gradient_and_optimality_certificate(representation):
    problem, optimum, diagonal, lower = build_problem(representation, count=12)
    X, K = problem.X, problem.K
    matrix = np.diag(diagonal) + np.diag(np.full(11, lower), -1)
    rng = np.random.default_rng(26)
    x_values, h_values = rng.normal(size=(2, 12))
    x, h = X.from_components(x_values), X.from_components(h_values)
    np.testing.assert_allclose(coordinates(X, K(x), 12), matrix @ x_values)
    np.testing.assert_allclose(coordinates(X, K.rapply(h), 12), matrix.T @ h_values)
    np.testing.assert_allclose(X.inner(K(x), h), X.inner(x, K.rapply(h)))
    np.testing.assert_allclose(
        coordinates(X, X.axpy(0.3, x, h), 12), 0.3 * x_values + h_values
    )
    epsilon = 1e-6
    derivative = (
        problem.smooth(X.axpy(epsilon, h, x))
        - problem.smooth(X.axpy(-epsilon, h, x))
    ) / (2 * epsilon)
    np.testing.assert_allclose(derivative, X.inner(problem.smooth.grad(x), h), rtol=1e-7)
    np.testing.assert_allclose(problem.objective(x), problem.g(x) + problem.f(K(x)))

    # Independent dense KKT certificate: full-rank K makes this optimum unique.
    target = coordinates(X, optimum, 12)
    gradient = matrix.T @ (matrix @ target - coordinates(X, problem.f.b, 12))
    active = target != 0
    np.testing.assert_allclose(
        gradient[active] + problem.g.weight * np.sign(target[active]), 0, atol=1e-14
    )
    assert np.max(np.abs(gradient[~active])) < problem.g.weight
    if representation == "sparse":
        assert not hasattr(X, "shape") and not hasattr(X, "flatten")
        with pytest.raises(TypeError, match="cannot be converted"):
            np.asarray(x)


@pytest.mark.parametrize("representation", ["dense", "sparse"])
@pytest.mark.parametrize("method", ["pg", "pdhg", "admm_cg", "admm_direct"])
def test_all_algorithms_reach_the_same_certified_solution(representation, method):
    problem, optimum, diagonal, lower = build_problem(representation, count=12)
    if method == "pg":
        result = proximal_gradient(problem)
    elif method == "pdhg":
        result = pdhg(problem)
    else:
        if method == "admm_cg":
            solve, work = prepare_cg(problem, 1.0, maxiter=24)
        else:
            solve, work = prepare_banded_solve(problem.X, diagonal, lower, 1.0)
        result = admm(problem, solve)
        assert work["solves"] == result.iterations
        assert max(result.primal_residual, result.dual_residual) <= 1e-8
    assert result.stationarity <= 1e-8
    assert problem.X.norm(problem.X.axpy(-1, optimum, result.x)) < 2e-7
    np.testing.assert_allclose(problem.objective(result.x), problem.objective(optimum), atol=1e-10)
    if representation == "sparse":
        assert isinstance(result.x, Coefficients)


@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_prepared_subproblem_matches_dense_solve_and_reuses_factorization(representation):
    problem, _, diagonal, lower = build_problem(representation, count=12)
    matrix = np.diag(diagonal) + np.diag(np.full(11, lower), -1)
    rng = np.random.default_rng(9)
    for rho in (0.2, 3.0):
        direct, work = prepare_banded_solve(problem.X, diagonal, lower, rho)
        cg, _ = prepare_cg(problem, rho, maxiter=24)
        for values in rng.normal(size=(3, 12)):
            rhs = problem.X.from_components(values)
            expected = np.linalg.solve(matrix.T @ matrix + rho * np.eye(12), values)
            for solve in (direct, cg):
                result = solve(rhs, problem.X.zeros())
                np.testing.assert_allclose(coordinates(problem.X, result, 12), expected, atol=1e-10)
            np.testing.assert_array_equal(coordinates(problem.X, rhs, 12), values)
        assert work == {"factorizations": 1, "solves": 3}


def test_optimized_admm_agrees_with_generic_execution():
    problem, _, diagonal, lower = build_problem("sparse", count=12)
    cg, cg_work = prepare_cg(problem, 1.0, maxiter=24)
    direct, direct_work = prepare_banded_solve(problem.X, diagonal, lower, 1.0)
    baseline, optimized = admm(problem, cg), admm(problem, direct)
    assert baseline.iterations == optimized.iterations
    assert problem.X.norm(problem.X.axpy(-1, baseline.x, optimized.x)) < 1e-10
    assert cg_work["krylov_iterations"] > direct_work["solves"]
    assert direct_work["factorizations"] == 1
