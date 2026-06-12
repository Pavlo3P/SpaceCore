import importlib
import numpy as np
import pytest


def test_numpy_ops_basic_array_creation():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    x = ops.asarray([1, 2, 3], dtype=np.float32)
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.dtype(np.float32)


def test_numpy_ops_linear_algebra_basics():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    A = ops.asarray([[1.0, 2.0], [3.0, 4.0]])
    x = ops.asarray([5.0, 6.0])
    assert np.allclose(ops.matmul(A, x), A @ x)
    assert np.allclose(ops.vdot(x, x), np.vdot(x, x))


def test_numpy_ops_shape_ops():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    x = ops.arange(6)
    y = ops.reshape(x, (2, 3))
    assert y.shape == (2, 3)
    assert np.allclose(ops.ravel(y), np.arange(6))


def test_numpy_ops_swapaxes():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    x = ops.reshape(ops.arange(24), (2, 3, 4))

    y = ops.swapaxes(x, 0, 2)

    assert y.shape == (4, 3, 2)
    assert np.allclose(y, np.swapaxes(np.arange(24).reshape(2, 3, 4), 0, 2))


def test_numpy_ops_reject_complex_to_real_casts():
    sc = importlib.import_module("spacecore")
    ops = sc.NumpyOps()
    x = np.asarray([1.0 + 1.0j], dtype=np.complex64)

    with pytest.raises(TypeError, match="rejected complex-valued input"):
        ops.asarray(x, dtype=np.float32)
    with pytest.raises(TypeError, match="rejected complex-valued input"):
        ops.astype(x, np.float32)
