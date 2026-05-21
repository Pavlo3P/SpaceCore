import importlib

import numpy as np
import pytest

from tests._helpers import has_cupy, has_jax, has_torch, jax_real_dtype, to_numpy
from tests._helpers import torch_real_dtype


def _backend_params():
    return [
        pytest.param("numpy", np.float64, id="numpy"),
        pytest.param(
            "jax",
            jax_real_dtype(),
            marks=pytest.mark.skipif(not has_jax(), reason="jax is not installed"),
            id="jax",
        ),
        pytest.param(
            "torch",
            torch_real_dtype(),
            marks=pytest.mark.skipif(not has_torch(), reason="torch is not installed"),
            id="torch",
        ),
        pytest.param(
            "cupy",
            np.float64,
            marks=pytest.mark.skipif(not has_cupy(), reason="cupy is not installed"),
            id="cupy",
        ),
    ]


def _ops_for_backend(name):
    sc = importlib.import_module("spacecore")
    if name == "numpy":
        return sc.NumpyOps()
    if name == "jax":
        return sc.JaxOps()
    if name == "torch":
        return sc.TorchOps()
    if name == "cupy":
        return sc.CuPyOps()
    raise ValueError(f"Unknown backend {name!r}.")


def _ctx(backend_name="numpy", dtype=np.float64):
    sc = importlib.import_module("spacecore")
    return sc.Context(_ops_for_backend(backend_name), dtype=dtype, enable_checks=False)


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_cg_solves_spd_system(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    space = sc.VectorSpace((2,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
    b = ctx.asarray([1.0, 2.0])

    result = sc.cg(A, b, tol=1e-7, maxiter=10)

    np.testing.assert_allclose(
        to_numpy(result.x),
        np.linalg.solve(np.array([[4.0, 1.0], [1.0, 3.0]]), np.array([1.0, 2.0])),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(to_numpy(A.apply(result.x)), to_numpy(b), rtol=1e-5, atol=1e-5)
    assert bool(to_numpy(result.converged))


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_lsqr_solves_rectangular_least_squares(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    A = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    b = ctx.asarray([1.0, 2.0, 4.0])

    result = sc.lsqr(A, b, tol=1e-7, maxiter=10)

    expected, *_ = np.linalg.lstsq(matrix, np.array([1.0, 2.0, 4.0]), rcond=None)
    np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(A.H.apply(A.apply(result.x) - b)), [0.0, 0.0], atol=1e-5)
    assert bool(to_numpy(result.converged))


def test_lsqr_works_with_matrix_free_linop_and_uses_rapply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    calls = {"rapply": 0}

    def apply(x):
        return matrix @ x

    def rapply(y):
        calls["rapply"] += 1
        return matrix.T @ y

    A = sc.MatrixFreeLinOp(apply, rapply, domain, codomain, ctx)
    b = ctx.asarray([1.0, 2.0, 3.0])

    result = sc.lsqr(A, b, tol=1e-8, maxiter=10)

    np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-6, atol=1e-6)
    assert calls["rapply"] > 0


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_power_iteration_estimates_dominant_eigenpair(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    space = sc.VectorSpace((2,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    x0 = ctx.asarray([1.0, 1.0])

    result = sc.power_iteration(A, x0=x0, tol=1e-5, maxiter=60)

    np.testing.assert_allclose(to_numpy(result.eigenvalue), 5.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.abs(to_numpy(result.eigenvector)),
        [0.0, 1.0],
        rtol=1e-4,
        atol=1e-4,
    )
    assert bool(to_numpy(result.converged))


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_stochastic_lanczos_approximates_smallest_eigenpair(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    initial = ctx.asarray([1.0, 1.0])

    eigenvalue, eigenvector = sc.stochastic_lanczos(
        op,
        initial,
        max_iter=2,
        tol=1e-8,
    )

    np.testing.assert_allclose(to_numpy(eigenvalue), 2.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.abs(to_numpy(eigenvector)),
        [1.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_stochastic_lanczos_uses_e0_for_zero_initial_vector():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    initial = ctx.asarray([0.0, 0.0])

    eigenvalue, eigenvector = sc.stochastic_lanczos(
        op,
        initial,
        max_iter=2,
        tol=1e-8,
    )

    np.testing.assert_allclose(eigenvalue, 2.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(eigenvector, [1.0, 0.0], rtol=1e-6, atol=1e-6)


def test_stochastic_lanczos_rejects_invalid_max_iter():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((1,), ctx)
    op = sc.IdentityLinOp(space, ctx)

    with pytest.raises(ValueError, match="max_iter"):
        sc.stochastic_lanczos(op, ctx.asarray([1.0]), max_iter=0)


def test_iterative_solvers_poll_convergence_on_check_interval():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    spd = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
    rectangular = sc.DenseLinOp(
        ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        space,
        sc.VectorSpace((3,), ctx),
        ctx,
    )
    diagonal = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    cg_result = sc.cg(spd, ctx.asarray([1.0, 2.0]), maxiter=65)
    lsqr_result = sc.lsqr(rectangular, ctx.asarray([1.0, 2.0, 3.0]), maxiter=65)
    power_result = sc.power_iteration(diagonal, x0=ctx.asarray([1.0, 1.0]), maxiter=65)

    assert cg_result.num_iters == 64
    assert lsqr_result.num_iters == 64
    assert power_result.num_iters == 64
    np.testing.assert_allclose(cg_result.residual_norm, 0.0, atol=1e-12)
    np.testing.assert_allclose(lsqr_result.normal_residual_norm, 0.0, atol=1e-12)
    np.testing.assert_allclose(power_result.residual_norm, 0.0, atol=1e-12)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_cg_jit_compiles_with_operator_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)

    solve = jax.jit(lambda A, b: sc.cg(A, b, maxiter=10).x)
    x = solve(op, ctx.asarray([1.0, 2.0]))

    np.testing.assert_allclose(to_numpy(x), [0.09090909, 0.63636364], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_lsqr_jit_compiles_with_operator_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)

    solve = jax.jit(lambda A, b: sc.lsqr(A, b, maxiter=10).x)
    x = solve(op, ctx.asarray([1.0, 2.0, 4.0]))

    np.testing.assert_allclose(to_numpy(x), [1.33333333, 2.33333333], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_power_iteration_jit_compiles_with_operator_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    run = jax.jit(lambda A, x: sc.power_iteration(A, x0=x, maxiter=60).eigenvalue)
    eigenvalue = run(op, ctx.asarray([1.0, 1.0]))

    np.testing.assert_allclose(to_numpy(eigenvalue), 5.0, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_stochastic_lanczos_jit_compiles_with_operator_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    def run(A, initial):
        return sc.stochastic_lanczos(
            A,
            initial,
            max_iter=2,
            tol=1e-8,
        )

    eigenvalue, eigenvector = jax.jit(run)(op, ctx.asarray([1.0, 1.0]))

    np.testing.assert_allclose(to_numpy(eigenvalue), 2.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.abs(to_numpy(eigenvector)),
        [1.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_cg_and_power_iteration_reject_rectangular_operator():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)

    with pytest.raises(ValueError, match="square LinOp"):
        sc.cg(A, ctx.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="square LinOp"):
        sc.power_iteration(A)
    with pytest.raises(ValueError, match="square LinOp"):
        sc.stochastic_lanczos(A, ctx.asarray([1.0, 2.0]))
