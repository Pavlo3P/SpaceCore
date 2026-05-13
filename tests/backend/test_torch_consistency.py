import importlib

import numpy as np
import pytest

from tests._helpers import has_torch, to_numpy, torch_real_dtype

pytestmark = pytest.mark.skipif(not has_torch(), reason="torch is not installed")


def test_numpy_and_torch_agree_on_shape_reductions_and_linalg():
    sc = importlib.import_module("spacecore")
    dt = torch_real_dtype()
    np_ops = sc.NumpyOps()
    th_ops = sc.TorchOps()

    x_np = np_ops.asarray([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x_th = th_ops.asarray(x_np, dtype=dt)
    A_np = np_ops.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    A_th = th_ops.asarray(A_np, dtype=dt)

    assert np_ops.shape(x_np) == th_ops.shape(x_th) == (2, 3)
    assert np_ops.ndim(x_np) == th_ops.ndim(x_th) == 2
    assert np_ops.size(x_np) == th_ops.size(x_th) == 6
    assert np.allclose(to_numpy(np_ops.zeros_like(x_np)), to_numpy(th_ops.zeros_like(x_th)))
    assert np.allclose(to_numpy(np_ops.ones_like(x_np)), to_numpy(th_ops.ones_like(x_th)))
    assert np.allclose(to_numpy(np_ops.full_like(x_np, 7.0)), to_numpy(th_ops.full_like(x_th, 7.0)))
    assert np.allclose(to_numpy(np_ops.broadcast_to(np_ops.asarray([1.0, 2.0, 3.0]), (2, 3))), to_numpy(th_ops.broadcast_to(th_ops.asarray([1.0, 2.0, 3.0], dtype=dt), (2, 3))))
    assert np.allclose(to_numpy(np_ops.moveaxis(x_np, 0, 1)), to_numpy(th_ops.moveaxis(x_th, 0, 1)))
    assert np.allclose(to_numpy(np_ops.eigvalsh(A_np)), to_numpy(th_ops.eigvalsh(A_th)))


def test_numpy_and_torch_dense_linop_agree_on_values():
    sc = importlib.import_module("spacecore")
    dt = torch_real_dtype()
    np_ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
    th_ctx = sc.Context(sc.TorchOps(), dtype=dt)
    Xn = sc.VectorSpace((2,), np_ctx)
    Yn = sc.VectorSpace((3,), np_ctx)
    Xt = sc.VectorSpace((2,), th_ctx)
    Yt = sc.VectorSpace((3,), th_ctx)
    data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    opn = sc.DenseLinOp(np_ctx.asarray(data), Xn, Yn, np_ctx)
    opt = sc.DenseLinOp(th_ctx.asarray(data), Xt, Yt, th_ctx)
    x_n = np_ctx.asarray([7.0, 8.0])
    x_t = th_ctx.asarray([7.0, 8.0])

    assert np.allclose(to_numpy(opn.apply(x_n)), to_numpy(opt.apply(x_t)))
