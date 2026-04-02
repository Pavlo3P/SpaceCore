import importlib
import numpy as np
from ._helpers import has_jax, jax_real_dtype


def test_set_and_get_context_with_object():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    try:
        sc.set_context(ctx)
        cur = sc.get_context()
        assert cur == ctx
        assert cur.enable_checks is False
    finally:
        sc.set_context(original)


def test_set_context_with_backend_name():
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    try:
        sc.set_context("numpy")
        assert isinstance(sc.get_context().ops, sc.NumpyOps)
    finally:
        sc.set_context(original)


def test_set_context_invalid_name_raises():
    sc = importlib.import_module("spacecore")
    import pytest
    with pytest.raises(Exception):
        sc.set_context("definitely_not_a_backend")


def test_set_context_jax_if_available():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    original = sc.get_context()
    try:
        sc.set_context(sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=True))
        assert isinstance(sc.get_context().ops, sc.JaxOps)
    finally:
        sc.set_context(original)
