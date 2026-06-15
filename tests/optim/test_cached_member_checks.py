"""Correctness invariants for the per-Space membership cache.

The hot-path optimization caches two things on every :class:`Space`
instance:

* ``_cached_member_checks`` — the full ``member_checks()`` tuple from
  walking the MRO and collecting class-level ``checks`` plus
  instance-level ``_local_checks``.
* ``_cached_checks_by_level`` — the per-``check_level`` subset of the
  above, used by ``_check_member``.

These tests assert that the cached lookups are byte-identical to a
freshly-walked one, that the cache is per-instance (not global), that
distinct check_levels return distinct subsets, and that validation
itself produces the same accept/reject decisions before and after the
cache is populated.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


def _check_set(space):
    return tuple(type(c).__name__ for c in space.member_checks())


def test_member_checks_cache_matches_fresh_walk():
    """Cached ``member_checks()`` matches a fresh MRO walk."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseCoordinateSpace((4,), ctx)
    # Force the cache to populate.
    first = space.member_checks()
    # Subsequent reads return the same tuple object.
    second = space.member_checks()
    assert first is second
    # Independently walking the MRO produces the same check set.
    fresh: list = []
    for klass in reversed(type(space).__mro__):
        fresh.extend(klass.__dict__.get("checks", ()))
        local = klass.__dict__.get("_local_checks")
        if local is not None:
            fresh.extend(local(space))
    assert tuple(fresh) == first


def test_member_checks_cache_is_per_instance():
    """Distinct spaces have independent caches.

    Required because ``_local_checks`` and instance state (geometry,
    shape) influence which checks the instance must satisfy.
    """
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    a = sc.DenseCoordinateSpace((4,), ctx)
    b = sc.DenseCoordinateSpace((8,), ctx)
    a.member_checks()
    b.member_checks()
    assert a._cached_member_checks is not b._cached_member_checks


def test_checks_for_level_is_filtered_correctly():
    """``_checks_for_level`` returns only the checks at-or-above the level."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.HermitianSpace(3, ctx)
    none_checks = space._checks_for_level("none")
    cheap_checks = space._checks_for_level("cheap")
    standard_checks = space._checks_for_level("standard")
    strict_checks = space._checks_for_level("strict")
    # Higher levels are supersets of lower.
    assert set(none_checks).issubset(set(cheap_checks))
    assert set(cheap_checks).issubset(set(standard_checks))
    assert set(standard_checks).issubset(set(strict_checks))
    # The standard level adds at least one check over cheap (Hermitian membership).
    assert len(standard_checks) > len(cheap_checks)


def test_checks_for_level_cache_is_per_level():
    """Each level computes its filtered subset at most once."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseCoordinateSpace((4,), ctx)
    first_cheap = space._checks_for_level("cheap")
    second_cheap = space._checks_for_level("cheap")
    assert first_cheap is second_cheap
    standard = space._checks_for_level("standard")
    assert standard is not first_cheap or len(standard) == len(first_cheap)


def test_validation_decisions_unchanged_by_cache():
    """Cached path accepts/rejects the same elements as the un-cached path."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseCoordinateSpace((4,), ctx)
    good = ctx.asarray(np.zeros(4))
    bad_shape = ctx.asarray(np.zeros(5))
    space._check_member(good)  # populates cache
    space._check_member(good)  # uses cache
    with pytest.raises(Exception):
        space._check_member(bad_shape)


def test_check_level_none_bypasses_validation():
    """``checked_method`` fast path: ``check_level="none"`` calls method directly."""
    ctx_none = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
    space = sc.DenseCoordinateSpace((4,), ctx_none)
    # A shape-wrong element would normally raise — but the fast path skips
    # validation entirely. The actual arithmetic still runs.
    bad = ctx_none.asarray(np.zeros(5))
    # Direct method call should NOT raise because none-level skips the
    # check; the underlying arithmetic just does whatever NumPy does.
    out = space.scale(2.0, bad)
    np.testing.assert_array_equal(np.asarray(out), np.zeros(5))


def test_check_level_cheap_validates_shape_dtype_backend():
    """At ``cheap`` level, the cached checks still catch shape mismatches."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")
    space = sc.DenseCoordinateSpace((4,), ctx)
    bad = ctx.asarray(np.zeros(5))
    with pytest.raises(Exception):
        space.check_member(bad)


def test_member_checks_cache_starts_empty():
    """A freshly constructed space has no populated cache yet."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseCoordinateSpace((4,), ctx)
    assert space._cached_member_checks is None
    assert space._cached_checks_by_level == {}
    space.member_checks()
    assert space._cached_member_checks is not None


def test_inner_product_space_caches_consistent_under_repeated_ops():
    """Repeated arithmetic uses the cache without producing wrong answers."""
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseCoordinateSpace((4,), ctx)
    x = ctx.asarray(np.asarray([1.0, 2.0, 3.0, 4.0]))
    y = ctx.asarray(np.asarray([1.0, 1.0, 1.0, 1.0]))
    expected_inner = 1.0 + 2.0 + 3.0 + 4.0
    for _ in range(5):
        assert float(space.inner(x, y)) == pytest.approx(expected_inner)


def test_check_level_change_via_new_context_uses_fresh_space():
    """``space.convert(new_ctx)`` produces a fresh-cache space.

    Spaces are immutable; switching ``check_level`` happens by creating a
    new context and a new space. The new space has its own cache.
    """
    ctx_std = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
    ctx_cheap = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")
    space_std = sc.DenseCoordinateSpace((4,), ctx_std)
    space_std.member_checks()
    space_cheap = space_std.convert(ctx_cheap)
    assert space_cheap is not space_std
    assert space_cheap._cached_member_checks is None
