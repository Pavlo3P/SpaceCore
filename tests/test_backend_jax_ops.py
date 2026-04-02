import importlib
import numpy as np
import pytest
from ._helpers import has_jax, jax_real_dtype

pytestmark = pytest.mark.skipif(not has_jax(), reason="jax is not installed")


def test_jax_ops_basic_array_creation():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()
    dt = jax_real_dtype()
    x = ops.asarray([1,2,3], dtype=dt)
    assert ops.is_dense(x)
    assert np.asarray(x).dtype == np.dtype(dt)


def test_jax_ops_linear_algebra_basics():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()
    dt = jax_real_dtype()
    A = ops.asarray([[1.,2.],[3.,4.]], dtype=dt)
    x = ops.asarray([5.,6.], dtype=dt)
    assert np.allclose(np.asarray(ops.matmul(A,x)), np.asarray(A) @ np.asarray(x))


def test_jax_ops_shape_ops():
    sc = importlib.import_module("spacecore")
    ops = sc.JaxOps()
    x = ops.arange(6, dtype=jax_real_dtype())
    y = ops.reshape(x, (2,3))
    assert y.shape == (2,3)
