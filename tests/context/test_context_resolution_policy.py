"""Policy tests for context resolution: dtype promotion and surviving checks.

Two properties of how operands are combined:

1. **dtype promotion never narrows.** These operators run on ill-conditioned
   problems (condition number growing like 1/ε), where the smallest scales sit
   below float32's machine epsilon, so silently downcasting a float64 operand
   to float32 would destroy precision the result depends on. Treating dtype /
   numerical precision as part of correctness — not an incidental — is the
   guidance in Irving et al., *Research Software Engineering with Python*
   (numerical-correctness testing).

2. **Surviving check-policy after combination should not depend on operand
   order.**

The dtype cases use a *Parameterized Test* with an independent literal expected
table (Meszaros, *xUnit Test Patterns*, ch. 27 Literal/Derived Value: do not
re-implement the production promotion rule inside the test).
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore.contextual._contextual import Contextual


def _resolver():
    return Contextual()


def _space(dtype, check_level="standard"):
    with sc.use_check_level(check_level):
        return sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), dtype=dtype))


# ---------------------------------------------------------------------------
# dtype promotion never narrows
# ---------------------------------------------------------------------------
class TestDtypePromotionNeverNarrows:
    # (name, dtype_a, dtype_b, expected) — expected is an independent literal,
    # not a call to the production promotion routine.
    PAIRS = [
        ("f32_f32", np.float32, np.float32, np.float32),
        ("f64_f64", np.float64, np.float64, np.float64),
        ("f32_f64", np.float32, np.float64, np.float64),
        ("f64_f32", np.float64, np.float32, np.float64),  # order-independent
        ("f32_c128", np.float32, np.complex128, np.complex128),
    ]

    @pytest.mark.parametrize(
        "da,db,expected",
        [(da, db, exp) for (_n, da, db, exp) in PAIRS],
        ids=[n for (n, *_r) in PAIRS],
    )
    def test_promotes_to_expected(self, da, db, expected):
        st = _resolver()
        ctx = st.resolve_context_priority(None, np.zeros(2, da), np.zeros(2, db))
        assert ctx.dtype == np.dtype(expected)

    @pytest.mark.parametrize(
        "da,db",
        [(da, db) for (_n, da, db, _e) in PAIRS],
        ids=[n for (n, *_r) in PAIRS],
    )
    def test_result_is_never_narrower_than_inputs(self, da, db):
        # The book property, independent of the exact promotion table: the
        # resolved dtype's itemsize is at least each input's (no precision loss).
        st = _resolver()
        ctx = st.resolve_context_priority(None, np.zeros(2, da), np.zeros(2, db))
        widest = max(np.dtype(da).itemsize, np.dtype(db).itemsize)
        assert np.dtype(ctx.dtype).itemsize >= widest

    def test_promotion_is_order_independent(self):
        st = _resolver()
        ab = st.resolve_context_priority(None, np.zeros(2, np.float32), np.zeros(2, np.float64))
        ba = st.resolve_context_priority(None, np.zeros(2, np.float64), np.zeros(2, np.float32))
        assert ab.dtype == ba.dtype


# ---------------------------------------------------------------------------
# Surviving check-policy is operand-order independent (critique §2.6)
# ---------------------------------------------------------------------------
class TestCheckLevelOrderIndependence:
    def test_leaf_objects_take_check_level_from_ambient(self):
        # check_level is a property of the bound object, seeded from the ambient
        # default (or an explicit scope), not from the Context.
        assert _space(np.float64, check_level="none").check_level == "none"
        assert _space(np.float64, check_level="strict").check_level == "strict"

    def test_linop_algebra_surviving_check_level_is_order_independent(self):
        # check_level is combined at the object level via the minimum rule
        # (ContextBound), so A+B and B+A agree and equal the least-strict operand
        # rather than whichever came first.
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        with sc.use_check_level("none"):
            a = sc.DiagonalLinOp(np.ones(2), ctx=ctx)
        with sc.use_check_level("strict"):
            b = sc.DiagonalLinOp(np.ones(2), ctx=ctx)
        assert a.check_level == "none" and b.check_level == "strict"
        assert (a + b).check_level == (b + a).check_level == "none"
