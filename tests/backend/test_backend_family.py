"""Tests for the :class:`spacecore.backend.BackendFamily` enum.

``BackendFamily`` is a ``StrEnum``: members behave as strings and round-trip
through the string form. Per-family behavior lives in the per-backend
specifics files (``test_numpy_ops.py``, ``test_jax_ops.py``, ...); this
module covers only the enum itself plus the family-string alias resolution
done by :func:`spacecore.normalize_ops`.
"""
from __future__ import annotations

import pytest

import spacecore as sc
from spacecore.backend import BackendFamily

from tests._helpers import has_cupy, has_torch


# ---------------------------------------------------------------------------
# Enum members
# ---------------------------------------------------------------------------
def test_backend_family_members_are_lowercase_strings():
    """Every family is a lowercase string value."""
    for member in BackendFamily:
        assert isinstance(member.value, str)
        assert member.value == member.value.lower()


def test_backend_family_expected_members():
    """The enum exposes numpy / jax / torch / cupy."""
    expected = {"numpy", "jax", "torch", "cupy"}
    actual = {m.value for m in BackendFamily}
    assert expected.issubset(actual)


# ---------------------------------------------------------------------------
# String round-trip (StrEnum)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["numpy", "jax", "torch", "cupy"])
def test_backend_family_string_round_trip(name):
    assert BackendFamily(name).value == name


def test_backend_family_string_inequality():
    assert BackendFamily.numpy != BackendFamily.jax


def test_backend_family_unknown_value_raises():
    with pytest.raises(ValueError):
        BackendFamily("madeup")


def test_backend_family_value_is_usable_as_string():
    """StrEnum members can substitute for str in string operations."""
    name = BackendFamily.numpy
    assert "numpy" in f"{name}_ops"
    assert name == "numpy"


# ---------------------------------------------------------------------------
# Alias resolution via ``normalize_ops`` (lives in _contextual but exercised
# here because the public family names that map to BackendFamily come through
# here).
# ---------------------------------------------------------------------------
def test_numpy_string_resolves_via_normalize_ops():
    ops = sc.normalize_ops("numpy")
    assert ops.family == "numpy"


@pytest.mark.skipif(not has_torch(), reason="torch is not installed")
def test_torch_aliases_resolve_via_dense_coordinate_space():
    """``"torch"`` and ``"pytorch"`` both resolve to the torch family."""
    assert sc.DenseCoordinateSpace((1,), "torch").ctx.ops.family == "torch"
    assert sc.DenseCoordinateSpace((1,), "pytorch").ctx.ops.family == "torch"


@pytest.mark.skipif(not has_cupy(), reason="cupy is not installed")
def test_cupy_string_resolves_via_dense_coordinate_space():
    assert sc.DenseCoordinateSpace((1,), "cupy").ctx.ops.family == "cupy"
