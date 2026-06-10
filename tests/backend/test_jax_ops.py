import importlib
import numpy as np
import pytest
from tests._helpers import has_jax, jax_real_dtype

pytestmark = pytest.mark.skipif(not has_jax(), reason="jax is not installed")


def test_jax_ops_basic_array_creation():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()
    dt = jax_real_dtype()
    x = ops.asarray([1, 2, 3], dtype=dt)
    assert ops.is_dense(x)
    assert np.asarray(x).dtype == np.dtype(dt)


def test_jax_ops_linear_algebra_basics():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()
    dt = jax_real_dtype()
    A = ops.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=dt)
    x = ops.asarray([5.0, 6.0], dtype=dt)
    assert np.allclose(np.asarray(ops.matmul(A, x)), np.asarray(A) @ np.asarray(x))


def test_jax_ops_shape_ops():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()
    x = ops.arange(6, dtype=jax_real_dtype())
    y = ops.reshape(x, (2, 3))
    assert y.shape == (2, 3)


def test_jax_ops_swapaxes():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()
    x = ops.reshape(ops.arange(24, dtype=jax_real_dtype()), (2, 3, 4))

    y = ops.swapaxes(x, 0, 2)

    assert y.shape == (4, 3, 2)
    assert np.allclose(np.asarray(y), np.swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2))
