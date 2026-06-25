"""Tests for :class:`spacecore.backend._ops.LazyNamespace`.

``LazyNamespace`` is the deferred-import shim used by
``TorchOps.xp = LazyNamespace("array_api_compat.torch")``. It loads its
target module on the first attribute access and proxies all subsequent
attribute lookups to that module. ``is_loaded`` is the observable flag
tests pin.
"""
from __future__ import annotations

import pytest

from spacecore.backend._ops import LazyNamespace


def test_lazy_namespace_is_not_loaded_at_construction():
    ns = LazyNamespace("numpy")
    assert ns.is_loaded is False


def test_lazy_namespace_loads_on_first_attribute_access():
    ns = LazyNamespace("numpy")
    _ = ns.zeros  # trigger import
    assert ns.is_loaded is True


def test_lazy_namespace_proxies_attribute_access():
    import numpy as np

    ns = LazyNamespace("numpy")
    # ns.zeros and np.zeros should be the same callable.
    assert ns.zeros is np.zeros


def test_lazy_namespace_caches_the_loaded_module():
    """Accessing the namespace twice does not re-import the module."""
    ns = LazyNamespace("numpy")
    _ = ns.array  # first access loads
    assert ns.is_loaded is True
    # Subsequent access returns the same proxied attribute.
    assert ns.array is ns.array


def test_lazy_namespace_missing_module_raises_on_access():
    ns = LazyNamespace("definitely_not_a_real_module_name_xyz")
    assert ns.is_loaded is False
    with pytest.raises(ImportError):
        _ = ns.something


def test_lazy_namespace_preserves_module_name_metadata():
    ns = LazyNamespace("numpy")
    assert ns.__name__ == "numpy"


def test_lazy_namespace_is_not_treated_as_abstract():
    """ABC inheritance must not pick up the namespace as an abstract method."""
    ns = LazyNamespace("numpy")
    assert ns.__isabstractmethod__ is False
