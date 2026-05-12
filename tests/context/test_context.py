import importlib
import numpy as np
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
    assert a == b
    assert a != c


def test_default_context_exists():
    sc = importlib.import_module("spacecore")
    ctx = sc.get_context()
    assert isinstance(ctx, sc.Context)


def test_jax_context_creation_if_available():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    ctx = sc.Context(sc.JaxOps(), dtype=dt, enable_checks=True)
    assert ctx.dtype == np.dtype(dt)
