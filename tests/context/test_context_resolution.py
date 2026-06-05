import importlib
import numpy as np
from tests._helpers import has_jax, jax_real_dtype


def test_explicit_context_has_priority():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    explicit = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    try:
        sc.set_context(sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True))
        X = sc.DenseCoordinateSpace((3,), ctx=explicit)
        assert X.ctx == explicit
        assert X.dtype == np.dtype(np.float32)
    finally:
        sc.set_context(original)


def test_public_resolve_context_priority_wrapper():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    default = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
    inferred = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    explicit = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    try:
        sc.set_context(default)
        X = sc.DenseCoordinateSpace((3,), ctx=inferred)

        assert sc.resolve_context_priority(None, X) == inferred
        assert sc.resolve_context_priority(explicit, X) == explicit
    finally:
        sc.set_context(original)


def test_resolve_context_priority_uses_explicit_ctx_before_inferred_ctx():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    default = sc.Context(sc.NumpyOps(), dtype=np.float16, enable_checks=True)
    inferred = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    explicit = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    try:
        sc.set_context(default)
        X = sc.DenseCoordinateSpace((2,), inferred)

        resolved = sc.resolve_context_priority(explicit, X)

        assert resolved == explicit
        assert resolved.dtype == np.dtype(np.float64)
        assert resolved.enable_checks is False
    finally:
        sc.set_context(original)


def test_resolve_context_priority_uses_inferred_ctx_only_without_explicit_ctx():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    default = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
    inferred = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    try:
        sc.set_context(default)
        X = sc.DenseCoordinateSpace((2,), inferred)

        resolved = sc.resolve_context_priority(None, X)

        assert resolved.ops.family == inferred.ops.family
        assert resolved.dtype == np.dtype(np.float32)
        assert resolved.enable_checks is False
    finally:
        sc.set_context(original)


def test_resolve_context_priority_uses_default_only_without_explicit_or_inferred_ctx():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    default = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    try:
        sc.set_context(default)

        assert sc.resolve_context_priority(None) == default
    finally:
        sc.set_context(original)


def test_default_context_used_when_none_given():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    try:
        sc.set_context(ctx)
        X = sc.DenseCoordinateSpace((2,))
        assert X.ctx == ctx
    finally:
        sc.set_context(original)


def test_product_space_resolves_common_backend():
    sc = importlib.import_module("spacecore")
    c1 = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    c2 = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    P = sc.ProductSpace((sc.DenseCoordinateSpace((2,), c1), sc.DenseCoordinateSpace((3,), c2)))
    assert P.ctx.ops.family == c1.ops.family


def test_cross_backend_product_space_resolution_raises_if_available():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    import pytest
    np_ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    jx_ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=True)
    with pytest.raises(Exception):
        sc.ProductSpace((sc.DenseCoordinateSpace((2,), np_ctx), sc.DenseCoordinateSpace((3,), jx_ctx)))
