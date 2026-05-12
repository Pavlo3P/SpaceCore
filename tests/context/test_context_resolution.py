import importlib
import numpy as np
from tests._helpers import has_jax, jax_real_dtype


def test_explicit_context_has_priority():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    explicit = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    try:
        sc.set_context(sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True))
        X = sc.VectorSpace((3,), ctx=explicit)
        assert X.ctx == explicit
        assert X.dtype == np.dtype(np.float32)
    finally:
        sc.set_context(original)


def test_default_context_used_when_none_given():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    try:
        sc.set_context(ctx)
        X = sc.VectorSpace((2,))
        assert X.ctx == ctx
    finally:
        sc.set_context(original)


def test_product_space_resolves_common_backend():
    sc = importlib.import_module("spacecore")
    c1 = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    c2 = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    P = sc.ProductSpace((sc.VectorSpace((2,), c1), sc.VectorSpace((3,), c2)))
    assert P.ctx.ops.family == c1.ops.family


def test_cross_backend_product_space_resolution_raises_if_available():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    import pytest
    np_ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    jx_ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=True)
    with pytest.raises(Exception):
        sc.ProductSpace((sc.VectorSpace((2,), np_ctx), sc.VectorSpace((3,), jx_ctx)))
