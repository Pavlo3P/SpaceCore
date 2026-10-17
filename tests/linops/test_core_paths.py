"""Check-free cores: the fused algebra paths compute correctly and validate once."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore.linop._algebra import _compose_chain


@pytest.fixture
def ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


def _endo(ctx, X, seed=0):
    rng = np.random.default_rng(seed)
    return sc.DenseLinOp(ctx.asarray(rng.standard_normal((3, 3))), X, X, ctx)


def test_fused_algebra_cores_compute_correctly(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    A = _endo(ctx, X, seed=1)
    B = _endo(ctx, X, seed=2)
    op = 2.0 * (A.H @ B) + sc.IdentityLinOp(X, ctx)
    x = ctx.asarray(np.asarray([1.0, -2.0, 3.0]))
    expected = 2.0 * A.rapply(B.apply(x)) + x
    np.testing.assert_allclose(np.asarray(op.apply(x)), np.asarray(expected))


def test_quadratic_value_validates_input_once(ctx):
    """``LinOpQuadraticForm.value`` checks its input once, not per sub-term.

    Without the cores, ``value`` would call ``Q.apply(x)`` and ``linear.value(x)``, each
    re-validating ``x``. The fused cores reach ``Q._apply_core`` and
    ``linear._value_core`` instead, so only the public boundary validates.
    """
    X = sc.DenseCoordinateSpace((3,), ctx)
    Q = sc.DiagonalLinOp(ctx.asarray([2.0, 3.0, 4.0]), X, ctx)
    c = sc.InnerProductFunctional(ctx.asarray([1.0, 1.0, 1.0]), X, ctx)
    qf = sc.LinOpQuadraticForm(Q, c, 0.5, ctx)
    x = ctx.asarray([1.0, 2.0, 3.0])

    counter = [0]
    original = qf.domain._check_member

    def counting(xx):
        counter[0] += 1
        return original(xx)

    qf.domain._check_member = counting
    value = qf.value(x)
    assert counter[0] == 1, "quadratic value re-validated an intermediate"

    expected = 0.5 * float(np.asarray([1.0, 2.0, 3.0]) @ (np.asarray([2.0, 3.0, 4.0]) * [1.0, 2.0, 3.0]))
    expected += float(np.asarray([1.0, 1.0, 1.0]) @ [1.0, 2.0, 3.0]) + 0.5
    np.testing.assert_allclose(float(np.asarray(value)), expected)


def test_compose_chain_rule_flattens(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    a, b, c = (_endo(ctx, X, s) for s in range(3))
    chain = a @ b @ c
    assert _compose_chain(chain) == chain._apply_chain == (c, b, a)
