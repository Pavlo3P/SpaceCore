import importlib
import numpy as np


def test_numpy_ops_basic_array_creation():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    x = ops.asarray([1,2,3], dtype=np.float32)
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.dtype(np.float32)


def test_numpy_ops_linear_algebra_basics():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    A = ops.asarray([[1.,2.],[3.,4.]])
    x = ops.asarray([5.,6.])
    assert np.allclose(ops.matmul(A,x), A @ x)
    assert np.allclose(ops.vdot(x,x), np.vdot(x,x))


def test_numpy_ops_shape_ops():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    x = ops.arange(6)
    y = ops.reshape(x, (2,3))
    assert y.shape == (2,3)
    assert np.allclose(ops.ravel(y), np.arange(6))
