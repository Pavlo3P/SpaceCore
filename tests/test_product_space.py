import importlib
import numpy as np
from ._helpers import has_jax, jax_real_dtype, prod


def test_product_space_construction_and_shape():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    P = sc.ProductSpace((sc.VectorSpace((2,), ctx), sc.VectorSpace((3,), ctx)), ctx)
    assert P.shape == (5,)
    assert prod(P.shape) == 5
    assert len(P.spaces) == 2


def test_product_space_zeros_add_flatten_unflatten():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    P = sc.ProductSpace((sc.VectorSpace((2,), ctx), sc.VectorSpace((3,), ctx)), ctx)
    x = (ctx.asarray([1.,2.]), ctx.asarray([3.,4.,5.]))
    z = P.zeros()
    assert np.allclose(z[0], 0.) and np.allclose(z[1], 0.)
    f = P.flatten(x)
    assert np.allclose(f, [1.,2.,3.,4.,5.])
    xr = P.unflatten(f)
    assert np.allclose(xr[0], x[0]) and np.allclose(xr[1], x[1])
    y = P.add(x, x)
    assert np.allclose(y[0], [2.,4.])


def test_product_space_check_member():
    sc = importlib.import_module("spacecore")
    import pytest
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    P = sc.ProductSpace((sc.VectorSpace((2,), ctx), sc.VectorSpace((3,), ctx)), ctx)
    P.check_member((ctx.asarray([1.,2.]), ctx.asarray([3.,4.,5.])))
    with pytest.raises(Exception):
        P.check_member((ctx.asarray([1.,2.]),))


def test_product_space_preserves_component_native_dtypes():
    sc = importlib.import_module("spacecore")
    c1 = sc.Context(sc.NumpyOps(), dtype=np.float32)
    c2 = sc.Context(sc.NumpyOps(), dtype=np.float64)
    P = sc.ProductSpace((sc.VectorSpace((2,), c1), sc.VectorSpace((3,), c2)), c2)
    assert P.dtype == np.dtype(np.float64)
    assert [sp.dtype for sp in P.spaces] == [np.dtype(np.float32), np.dtype(np.float64)]


def test_product_space_convert_changes_backend_only_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    P = sc.ProductSpace((sc.VectorSpace((2,), sc.Context(sc.NumpyOps(), dtype=dt)), sc.VectorSpace((3,), sc.Context(sc.NumpyOps(), dtype=dt))))
    Q = P.convert(sc.Context(sc.JaxOps(), dtype=dt))
    assert Q.ctx.ops.family == "jax"
