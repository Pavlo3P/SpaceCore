import importlib
import numpy as np
from tests._helpers import has_jax, jax_real_dtype


def test_vector_space_dtype_follows_context():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
    X = sc.VectorSpace((3,), ctx=ctx)
    assert X.dtype == np.dtype(np.float32)


def test_conversion_preserves_native_dtype_under_keep_native_policy():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.VectorSpace((3,), ctx=src)
    Y = X.convert(dst)
    assert Y.dtype == X.dtype
    assert Y.ctx.ops.family == dst.ops.family


def test_numpy_to_jax_conversion_preserves_source_dtype_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    src_dt = jax_real_dtype()
    X = sc.VectorSpace((3,), ctx=sc.Context(sc.NumpyOps(), dtype=src_dt))
    Y = X.convert(sc.Context(sc.JaxOps(), dtype=src_dt))
    assert Y.dtype == np.dtype(src_dt)
