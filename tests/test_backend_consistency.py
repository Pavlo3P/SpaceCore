import importlib
import numpy as np
import pytest
from ._helpers import has_jax, jax_real_dtype, to_numpy

pytestmark = pytest.mark.skipif(not has_jax(), reason="jax is not installed")


def test_numpy_and_jax_agree_on_values_for_supported_dtype():
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ops = sc.NumpyOps()
    jx_ops = sc.JaxOps()
    x_np = np_ops.asarray([1.,2.,3.], dtype=dt)
    x_jx = jx_ops.asarray([1.,2.,3.], dtype=dt)
    assert np.allclose(to_numpy(x_np), to_numpy(x_jx))


def test_numpy_and_jax_dense_linop_agree_on_values():
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    np_ctx = sc.Context(sc.NumpyOps(), dtype=dt)
    jx_ctx = sc.Context(sc.JaxOps(), dtype=dt)
    Xn = sc.VectorSpace((2,), np_ctx)
    Yn = sc.VectorSpace((3,), np_ctx)
    Xj = sc.VectorSpace((2,), jx_ctx)
    Yj = sc.VectorSpace((3,), jx_ctx)
    data = [[1.,2.],[3.,4.],[5.,6.]]
    opn = sc.DenseLinOp(np_ctx.asarray(data), Xn, Yn, np_ctx)
    opj = sc.DenseLinOp(jx_ctx.asarray(data), Xj, Yj, jx_ctx)
    x_n = np_ctx.asarray([7.,8.])
    x_j = jx_ctx.asarray([7.,8.])
    assert np.allclose(to_numpy(opn.apply(x_n)), to_numpy(opj.apply(x_j)))
