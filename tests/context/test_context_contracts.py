"""Contract tests for ``Context`` equality and its coarsening relations.

Covers the value-object contracts of ``spacecore.contextual._context.Context``:
``__eq__``/``__hash__`` and the relations ``same_math`` (ops & dtype) and
``same_backend`` (ops). ``check_level`` is no longer part of the context — it is
a property of the bound object — so ``Context`` identity is purely mathematical
and ``same_math`` now coincides with ``__eq__``; only ``same_backend`` (which
drops the dtype) is a strictly coarser relation.

Book justification: the relations are tested *as contracts* — reflexive /
symmetric / transitive — rather than by poking at fields (Design by Contract;
Hunt & Thomas, *The Pragmatic Programmer*, tip 37). The shared custom assertions
keep each test to one reported concern (Meszaros, *xUnit Test Patterns*: Custom
Assertion, avoid Assertion Roulette).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import spacecore as sc

from ._contracts import assert_equality_contract, assert_equivalence_relation


# A NumpyOps subclass with a distinct family, so cross-family cases need no
# optional backend installed.
class _OtherFamilyOps(sc.NumpyOps):
    _family = "other_family"


def _ctx(dtype=np.float64, ops=None):
    return sc.Context(ops or sc.NumpyOps(), dtype=dtype)


# ---------------------------------------------------------------------------
# __eq__ / __hash__ as a value-object contract
# ---------------------------------------------------------------------------
class TestEqualityContract:
    def test_full_equality_contract(self):
        assert_equality_contract(
            equal=[_ctx(), _ctx()],
            distinct=[
                _ctx(dtype=np.float32),  # dtype differs
                _ctx(ops=_OtherFamilyOps()),  # backend family differs
            ],
        )

    def test_frozen_is_immutable(self):
        ctx = _ctx()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.dtype = np.float32  # type: ignore[misc]

    def test_usable_as_dict_key(self):
        a, b = _ctx(), _ctx()
        assert {a: "v"}[b] == "v"


# ---------------------------------------------------------------------------
# same_math (ops & dtype) — now coincides with __eq__
# ---------------------------------------------------------------------------
class TestSameMathRelation:
    def test_is_equivalence_relation(self):
        assert_equivalence_relation(
            lambda a, b: a.same_math(b),
            equivalent=[_ctx(), _ctx()],
            unrelated=[_ctx(dtype=np.float32), _ctx(ops=_OtherFamilyOps())],
        )

    def test_coincides_with_equality(self):
        a, b = _ctx(dtype=np.float32), _ctx(dtype=np.float64)
        assert a.same_math(b) is False and (a == b) is False
        c, d = _ctx(), _ctx()
        assert c.same_math(d) is True and (c == d) is True

    def test_non_context_is_false(self):
        assert _ctx().same_math(object()) is False
        assert _ctx().same_math(None) is False


# ---------------------------------------------------------------------------
# same_backend (ops only) — strictly coarser than same_math
# ---------------------------------------------------------------------------
class TestSameBackendRelation:
    def test_is_equivalence_relation(self):
        assert_equivalence_relation(
            lambda a, b: a.same_backend(b),
            equivalent=[
                _ctx(dtype=np.float64),
                _ctx(dtype=np.float32),
                _ctx(dtype=np.complex128),
            ],
            unrelated=[_ctx(ops=_OtherFamilyOps())],
        )

    def test_non_context_is_false(self):
        assert _ctx().same_backend(object()) is False
        assert _ctx().same_backend(None) is False


# ---------------------------------------------------------------------------
# The coarsening chain: __eq__ ⊆ same_math ⊆ same_backend
# ---------------------------------------------------------------------------
class TestCoarseningChain:
    CASES = [
        ("identical", _ctx(), _ctx(), True, True, True),
        ("dtype_differs", _ctx(dtype=np.float32), _ctx(dtype=np.float64), False, False, True),
        ("family_differs", _ctx(ops=_OtherFamilyOps()), _ctx(), False, False, False),
    ]

    @pytest.mark.parametrize(
        "a,b,eq,sm,sb",
        [(a, b, eq, sm, sb) for (_n, a, b, eq, sm, sb) in CASES],
        ids=[n for (n, *_r) in CASES],
    )
    def test_relation_values(self, a, b, eq, sm, sb):
        assert (a == b) is eq
        assert a.same_math(b) is sm
        assert a.same_backend(b) is sb

    @pytest.mark.parametrize(
        "a,b",
        [(a, b) for (_n, a, b, *_r) in CASES],
        ids=[n for (n, *_r) in CASES],
    )
    def test_implications_hold(self, a, b):
        # Coarsening: __eq__ ⟹ same_math ⟹ same_backend, for every case.
        if a == b:
            assert a.same_math(b)
        if a.same_math(b):
            assert a.same_backend(b)
