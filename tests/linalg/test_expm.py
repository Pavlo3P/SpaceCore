import importlib

import numpy as np
import pytest
import scipy.linalg

from tests._helpers import has_jax, has_torch, jax_complex_dtype, jax_real_dtype, to_numpy
from tests._helpers import torch_real_dtype


def _ops_for_backend(name):
    sc = importlib.import_module("spacecore")
    if name == "numpy":
        return sc.NumpyOps()
    if name == "jax":
        return sc.JaxOps()
    if name == "torch":
        return sc.TorchOps()
    raise ValueError(f"Unknown backend {name!r}.")


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
    ]


def _ctx(backend_name="numpy", dtype=np.float64, enable_checks=False):
    sc = importlib.import_module("spacecore")
    return sc.Context(_ops_for_backend(backend_name), dtype=dtype, enable_checks=enable_checks)


def _operator(ctx, matrix):
    sc = importlib.import_module("spacecore")
    space = sc.VectorSpace((matrix.shape[0],), ctx)
    return sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)


def _ground_truth(matrix, vector, t):
    return scipy.linalg.expm(t * matrix) @ vector


def test_expm_multiply_t_zero_returns_input():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    matrix = np.array([[2.0, 0.5], [0.5, 3.0]])
    A = _operator(ctx, matrix)
    v = ctx.asarray([1.0, -2.0])

    result = sc.expm_multiply(A, v, t=0.0, max_iter=4)

    np.testing.assert_allclose(result.result, v, rtol=1e-12, atol=1e-12)
    assert isinstance(result, sc.ExpmMultiplyResult)


def test_expm_multiply_rejects_structurally_non_hermitian_operator():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    A = _operator(ctx, np.array([[1.0, 2.0], [0.0, 3.0]]))
    v = ctx.asarray([1.0, -2.0])

    with pytest.raises(ValueError, match="Hermitian"):
        sc.expm_multiply(A, v, t=0.1, max_iter=4)


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_expm_multiply_matches_dense_ground_truth(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    matrix = np.array([[1.0, 0.25, 0.0], [0.25, 2.0, -0.5], [0.0, -0.5, 3.0]])
    A = _operator(ctx, matrix)
    v_np = np.array([1.0, -2.0, 0.5])
    v = ctx.asarray(v_np)

    result = sc.expm_multiply(A, v, t=-0.2, max_iter=8, tol=1e-12)

    np.testing.assert_allclose(
        to_numpy(result.result),
        _ground_truth(matrix, v_np, -0.2),
        rtol=1e-5,
        atol=1e-5,
    )
    assert bool(to_numpy(result.converged))


def test_expm_multiply_is_linear_in_vector():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    A = _operator(ctx, np.array([[1.0, 0.5], [0.5, 2.0]]))
    v1 = ctx.asarray([1.0, -1.0])
    v2 = ctx.asarray([0.5, 2.0])
    alpha = 1.5
    beta = -0.25

    combined = sc.expm_multiply(A, alpha * v1 + beta * v2, t=0.3, max_iter=6).result
    expected = (
        alpha * sc.expm_multiply(A, v1, t=0.3, max_iter=6).result
        + beta * sc.expm_multiply(A, v2, t=0.3, max_iter=6).result
    )

    np.testing.assert_allclose(combined, expected, rtol=1e-10, atol=1e-10)


def test_expm_multiply_group_property():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    A = _operator(ctx, np.array([[1.0, 0.5], [0.5, 2.0]]))
    v = ctx.asarray([1.0, -1.0])
    t1 = 0.2
    t2 = -0.35

    first = sc.expm_multiply(A, v, t=t1, max_iter=6).result
    sequential = sc.expm_multiply(A, first, t=t2, max_iter=6).result
    direct = sc.expm_multiply(A, v, t=t1 + t2, max_iter=6).result

    np.testing.assert_allclose(sequential, direct, rtol=1e-10, atol=1e-10)


def test_expm_multiply_complex_time_is_unitary_for_hermitian_generator():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.complex128)
    matrix = np.array([[1.0, 0.5 - 0.25j], [0.5 + 0.25j, 2.0]], dtype=np.complex128)
    A = _operator(ctx, matrix)
    v = ctx.asarray([1.0 + 0.5j, -0.25 + 0.75j])

    result = sc.expm_multiply(A, v, t=-0.5j, max_iter=6, tol=1e-12).result

    np.testing.assert_allclose(
        to_numpy(A.domain.norm(result)),
        to_numpy(A.domain.norm(v)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_expm_multiply_complex_time_matches_dense_truth():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.complex128)
    matrix = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.complex128)
    A = _operator(ctx, matrix)
    v_np = np.array([1.0 - 0.5j, 0.25 + 0.75j])
    v = ctx.asarray(v_np)

    result = sc.expm_multiply(A, v, t=-0.5j, max_iter=6, tol=1e-12)

    np.testing.assert_allclose(
        to_numpy(result.result),
        _ground_truth(matrix, v_np, -0.5j),
        rtol=1e-10,
        atol=1e-10,
    )


def test_expm_multiply_residual_estimate_decreases_with_more_iterations():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    matrix = np.array([[1.0, 0.25, 0.1], [0.25, 2.0, -0.5], [0.1, -0.5, 4.0]])
    A = _operator(ctx, matrix)
    v = ctx.asarray([1.0, -2.0, 0.5])

    low = sc.expm_multiply(A, v, t=0.4, max_iter=1, tol=1e-12)
    high = sc.expm_multiply(A, v, t=0.4, max_iter=3, tol=1e-12)

    assert to_numpy(high.residual_estimate) <= to_numpy(low.residual_estimate)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_expm_multiply_jit_matches_eager():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_complex_dtype())
    matrix = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.complex128)
    A = _operator(ctx, matrix)
    v = ctx.asarray([1.0 - 0.5j, 0.25 + 0.75j])

    eager = sc.expm_multiply(A, v, t=-0.5j, max_iter=6).result
    run = jax.jit(lambda op, x: sc.expm_multiply(op, x, t=-0.5j, max_iter=6).result)
    compiled = run(A, v)

    np.testing.assert_allclose(to_numpy(compiled), to_numpy(eager), rtol=1e-6, atol=1e-6)
