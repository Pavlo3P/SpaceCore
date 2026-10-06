"""Tests for :mod:`spacecore._checks` pure helper functions.

Book justification: these are pure functions, pinned directly against their
contract rather than only through ``Context`` (Hunt & Thomas, *The Pragmatic
Programmer*, tip 37). The ordering and ``>=`` relations are covered as
exhaustive Parameterized truth tables to avoid Conditional Test Logic and
Assertion Roulette (Meszaros, *xUnit Test Patterns*); normalization that
resolves rather than raises reflects "define errors out of existence"
(Ousterhout, *A Philosophy of Software Design*).

These functions back the public ``check_level`` policy but are otherwise only
exercised transitively through :class:`spacecore.Context`. This module unit
tests them directly.

Checklist section 2:

* ``CHECK_LEVELS`` is the canonical ordering tuple and its order index is
  strictly increasing.
* ``check_level_at_least`` implements the full ``actual`` >= ``required``
  truth table over the four levels.
* ``normalize_check_level`` resolves ``None`` to its default, passes explicit
  levels through, and rejects unknown levels.
* ``minimum_check_level`` returns ``"none"`` for the empty tuple and otherwise
  the least-expensive level of a mixed tuple.
"""
from __future__ import annotations


import numpy as np  # noqa: F401  (kept for convention parity across the suite)
import pytest

import spacecore as sc  # noqa: F401  (convention import; helpers are private)
from spacecore._checks import (
    CHECK_LEVELS,
    check_level_at_least,
    minimum_check_level,
    normalize_check_level,
)


# ===========================================================================
# CHECK_LEVELS ordering
# ===========================================================================
class TestCheckLevelsOrdering:
    def test_canonical_tuple(self):
        assert CHECK_LEVELS == ("none", "cheap", "standard", "strict")

    def test_order_index_is_strictly_increasing(self):
        indices = [CHECK_LEVELS.index(level) for level in CHECK_LEVELS]
        assert indices == sorted(indices)
        assert all(b - a == 1 for a, b in zip(indices, indices[1:]))

    def test_public_alias_matches_module_tuple(self):
        assert sc.CHECK_LEVELS == CHECK_LEVELS


# ===========================================================================
# check_level_at_least
# ===========================================================================
class TestCheckLevelAtLeast:
    @pytest.mark.parametrize(
        "actual, required, expected",
        [
            ("none", "none", True),
            ("none", "cheap", False),
            ("none", "standard", False),
            ("none", "strict", False),
            ("cheap", "none", True),
            ("cheap", "cheap", True),
            ("cheap", "standard", False),
            ("cheap", "strict", False),
            ("standard", "none", True),
            ("standard", "cheap", True),
            ("standard", "standard", True),
            ("standard", "strict", False),
            ("strict", "none", True),
            ("strict", "cheap", True),
            ("strict", "standard", True),
            ("strict", "strict", True),
        ],
    )
    def test_truth_table(self, actual, required, expected):
        assert check_level_at_least(actual, required) is expected


# ===========================================================================
# normalize_check_level
# ===========================================================================
class TestNormalizeCheckLevel:
    def test_none_resolves_to_default_standard(self):
        assert normalize_check_level(None) == "standard"

    def test_none_resolves_to_explicit_default(self):
        assert normalize_check_level(None, default="cheap") == "cheap"

    @pytest.mark.parametrize("level", ["none", "cheap", "standard", "strict"])
    def test_explicit_level_passes_through(self, level):
        assert normalize_check_level(level) == level

    def test_unknown_level_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown check_level"):
            normalize_check_level("fast")

    def test_unknown_level_message_lists_allowed(self):
        with pytest.raises(ValueError) as exc_info:
            normalize_check_level("fast")
        message = str(exc_info.value)
        assert "'none'" in message
        assert "'strict'" in message


# ===========================================================================
# minimum_check_level
# ===========================================================================
class TestMinimumCheckLevel:
    def test_empty_tuple_returns_none(self):
        assert minimum_check_level(()) == "none"

    def test_picks_least_expensive_of_mixed_tuple(self):
        assert minimum_check_level(("strict", "cheap", "standard")) == "cheap"

    def test_single_element_returns_itself(self):
        assert minimum_check_level(("strict",)) == "strict"

    def test_none_in_tuple_dominates(self):
        assert minimum_check_level(("strict", "none", "standard")) == "none"
