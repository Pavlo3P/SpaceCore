"""Shared fixtures for the ``tests/context/`` suite.

The fixtures here keep tests in this directory from leaking process-wide
state into one another. The most common leak is :func:`spacecore.set_context`
mutating the default context; without an explicit reset, a later test sees
an unexpected default and fails spuriously.
"""
from __future__ import annotations

import pytest

import spacecore as sc


@pytest.fixture
def preserve_default_context():
    """Snapshot the active default ``Context`` and restore it on teardown.

    Tests that exercise :func:`spacecore.set_context` should depend on this
    fixture so they can mutate the default without side effects on
    siblings. The fixture is intentionally not ``autouse=True`` — most
    tests don't mutate the default and don't need the snapshot.
    """
    original = sc.get_context()
    yield original
    sc.set_context(original)
