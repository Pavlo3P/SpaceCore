"""Backward-compatible re-export of the shared conformance harness.

The canonical definitions now live in :mod:`tests._conformance` so the whole
suite — not just ``tests/backend/`` — can compare a backend result against an
independent reference with one consistent per-op tolerance policy. This module
re-exports them so existing ``from tests.backend._conformance import ...``
imports keep working.

The backend conformance suite parametrizes over the ``backend_ops`` fixture
(``tests/backend/conftest.py``); each test runs once per available backend,
builds a NumPy reference, and asserts equivalence via
:func:`assert_matches_reference`. See :mod:`tests._conformance` for the
tolerance table and helper documentation.
"""
from __future__ import annotations

from tests._conformance import (
    TOLERANCE_TABLE,
    Tolerance,
    assert_eigh_identity,
    assert_matches_reference,
    available_backend_families,
    backend_supports_dtype,
    dtype_kind,
    iter_dtypes,
    numpy_reference,
    tolerance_for,
)

__all__ = [
    "Tolerance",
    "TOLERANCE_TABLE",
    "dtype_kind",
    "tolerance_for",
    "assert_matches_reference",
    "assert_eigh_identity",
    "numpy_reference",
    "available_backend_families",
    "backend_supports_dtype",
    "iter_dtypes",
]
