import importlib
import numpy as np
from tests._helpers import has_jax, jax_real_dtype, prod


def test_vector_space_construction():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.VectorSpace((3,2), ctx)
    assert X.shape == (3,2)
    assert prod(X.shape) == 6


def test_vector_space_zeros_add_scale_inner():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.VectorSpace((3,), ctx)
    z = X.zeros()
    x = ctx.asarray([1.,2.,3.])
    y = ctx.asarray([4.,5.,6.])
    assert np.allclose(z, 0.)
    assert np.allclose(X.add(x,y), [5.,7.,9.])
    assert np.allclose(X.scale(2.,x), [2.,4.,6.])
    assert np.allclose(X.inner(x,y), 32.)


def test_vector_space_check_member():
    sc = importlib.import_module("spacecore")
    import pytest
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    X = sc.VectorSpace((2,), ctx)
    X.check_member(ctx.asarray([1.,2.]))
    with pytest.raises(Exception):
        X.check_member(np.asarray([1.,2.,3.], dtype=np.float32))


def test_vector_space_convert_changes_backend_not_native_dtype():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.VectorSpace((2,), src)
    Y = X.convert(dst)
    assert Y.dtype == X.dtype
    assert Y.ctx.ops.family == dst.ops.family


def test_vector_space_convert_to_jax_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    X = sc.VectorSpace((2,), sc.Context(sc.NumpyOps(), dtype=dt))
    Y = X.convert(sc.Context(sc.JaxOps(), dtype=dt))
    assert Y.ctx.ops.family == "jax"
