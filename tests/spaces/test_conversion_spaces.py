import importlib
import numpy as np
from tests._helpers import has_jax, jax_real_dtype


def test_vector_and_hermitian_conversion_use_target_dtype():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.DenseCoordinateSpace((2, 3), src)
    Y = X.convert(dst)
    assert Y.shape == X.shape and Y.dtype == dst.dtype
    H = sc.HermitianSpace(2, ctx=src)
    K = H.convert(dst)
    assert K.shape == H.shape and K.dtype == dst.dtype


def test_product_conversion_preserves_component_shapes():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
    P = sc.ProductSpace(
        (sc.DenseCoordinateSpace((2, 2), ctx), sc.DenseCoordinateSpace((3,), ctx)), ctx
    )
    Q = P.convert(sc.Context(sc.NumpyOps(), dtype=np.float64))
    assert [sp.shape for sp in Q.spaces] == [(2, 2), (3,)]


def test_space_conversion_to_same_effective_context_returns_self():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
    X = sc.DenseCoordinateSpace((3,), ctx)
    P = sc.ProductSpace((X, sc.DenseCoordinateSpace((2,), ctx)), ctx)

    assert X.convert(ctx) is X
    assert P.convert(ctx) is P


def test_space_conversion_to_jax_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    X = sc.DenseCoordinateSpace((3,), sc.Context(sc.NumpyOps(), dtype=dt))
    Y = X.convert(sc.Context(sc.JaxOps(), dtype=dt))
    assert Y.ctx.ops.family == "jax"
