import importlib
import numpy as np
from ._helpers import has_jax, jax_real_dtype


def test_vector_and_hermitian_conversion_preserve_shape_and_native_dtype():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.VectorSpace((2,3), src)
    Y = X.convert(dst)
    assert Y.shape == X.shape and Y.dtype == X.dtype
    H = sc.HermitianSpace(2, ctx=src)
    K = H.convert(dst)
    assert K.shape == H.shape and K.dtype == H.dtype


def test_product_conversion_preserves_component_shapes():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
    P = sc.ProductSpace((sc.VectorSpace((2,2),ctx), sc.VectorSpace((3,),ctx)), ctx)
    Q = P.convert(sc.Context(sc.NumpyOps(), dtype=np.float64))
    assert [sp.shape for sp in Q.spaces] == [(2,2),(3,)]


def test_space_conversion_to_jax_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    X = sc.VectorSpace((3,), sc.Context(sc.NumpyOps(), dtype=dt))
    Y = X.convert(sc.Context(sc.JaxOps(), dtype=dt))
    assert Y.ctx.ops.family == 'jax'
