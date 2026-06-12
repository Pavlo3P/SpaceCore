import importlib
import numpy as np
import pytest
from tests._helpers import has_jax, jax_real_dtype


def test_numpy_context_creation():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
    assert ctx.dtype == np.dtype(np.float64)
    assert ctx.enable_checks is True


def test_context_equality_depends_on_backend_and_checks():
    sc = importlib.import_module("spacecore")
    a = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=True)
    b = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
    c = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    assert a != b
    assert a != c


def test_default_context_exists():
    sc = importlib.import_module("spacecore")
    ctx = sc.get_context()
    assert isinstance(ctx, sc.Context)


def test_context_dtype_is_a_representation_default():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)

    assert ctx.asarray([1.0, 2.0]).dtype == np.dtype(np.float32)


def test_numpy_context_rejects_complex_to_real_narrowing():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
    value = np.asarray([1.0 + 0.0j], dtype=np.complex64)

    with pytest.raises(TypeError, match="rejected complex-valued input.*x.real"):
        ctx.asarray(value)

    with pytest.raises(TypeError, match="rejected complex-valued input.*x.real"):
        ctx.assparse(value.reshape((1, 1)))


def test_complex_to_real_conversion_succeeds_after_explicit_real_part():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32)
    value = np.asarray([1.0 + 2.0j], dtype=np.complex64)

    converted = ctx.asarray(value.real)

    assert converted.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(converted, [1.0])


def test_jax_context_creation_if_available():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    ctx = sc.Context(sc.JaxOps(), dtype=dt, enable_checks=True)
    assert ctx.dtype == np.dtype(dt)
