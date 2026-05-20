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


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_fori_loop_accumulates_indices(backend_name, dtype):
    ops = _ops_for_backend(backend_name)

    def body_fun(i, carry):
        return carry + ops.asarray(i, dtype=dtype)

    out = ops.fori_loop(0, 5, body_fun, ops.asarray(0.0, dtype=dtype))

    np.testing.assert_allclose(to_numpy(out), 10.0)


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_while_loop_reaches_terminal_state(backend_name, dtype):
    ops = _ops_for_backend(backend_name)
    limit = ops.asarray(4.0, dtype=dtype)

    def cond_fun(carry):
        return carry < limit

    def body_fun(carry):
        return carry + ops.asarray(1.0, dtype=dtype)

    out = ops.while_loop(cond_fun, body_fun, ops.asarray(0.0, dtype=dtype))

    np.testing.assert_allclose(to_numpy(out), 4.0)


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_scan_accumulates_and_stacks_outputs(backend_name, dtype):
    ops = _ops_for_backend(backend_name)
    xs = ops.asarray([1.0, 2.0, 3.0, 4.0], dtype=dtype)

    def body_fun(carry, x):
        new_carry = carry + x
        return new_carry, new_carry * ops.asarray(2.0, dtype=dtype)

    final, ys = ops.scan(body_fun, ops.asarray(0.0, dtype=dtype), xs)

    np.testing.assert_allclose(to_numpy(final), 10.0)
    np.testing.assert_allclose(to_numpy(ys), [2.0, 6.0, 12.0, 20.0])


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_scan_without_xs_uses_explicit_length(backend_name, dtype):
    ops = _ops_for_backend(backend_name)

    def body_fun(carry, _):
        new_carry = carry + ops.asarray(2.0, dtype=dtype)
        return new_carry, new_carry

    final, ys = ops.scan(body_fun, ops.asarray(1.0, dtype=dtype), None, length=3)

    np.testing.assert_allclose(to_numpy(final), 7.0)
    np.testing.assert_allclose(to_numpy(ys), [3.0, 5.0, 7.0])


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_cond_selects_expected_branch(backend_name, dtype):
    ops = _ops_for_backend(backend_name)
    x = ops.asarray(3.0, dtype=dtype)

    def true_fun(value):
        return value + ops.asarray(10.0, dtype=dtype)

    def false_fun(value):
        return value - ops.asarray(10.0, dtype=dtype)

    true_out = ops.cond(True, true_fun, false_fun, x)
    false_out = ops.cond(False, true_fun, false_fun, x)

    np.testing.assert_allclose(to_numpy(true_out), 13.0)
    np.testing.assert_allclose(to_numpy(false_out), -7.0)
