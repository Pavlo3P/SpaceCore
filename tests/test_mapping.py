"""``Mapping`` owns the lazy algebra: same-family and cross-family rules, and derived forms."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore.mapping import Mapping, Scalar, register


@pytest.fixture
def ops():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X = sc.DenseCoordinateSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray(np.arange(9.0).reshape(3, 3)), X, X, ctx)
    B = sc.DenseLinOp(ctx.asarray(np.eye(3)), X, X, ctx)
    return ctx, A, B, sc.SquaredL2NormFunctional(X)


def test_both_families_are_mappings(ops):
    _, A, _, F = ops
    assert isinstance(A, Mapping) and isinstance(F, Mapping)


def test_cross_family_rules(ops):
    ctx, A, _, F = ops
    x = ctx.asarray([1.0, -2.0, 0.5])
    assert (F * A) == (A * F)
    assert (F @ A) == F.compose(A)
    np.testing.assert_allclose((F @ A)(x), F(A(x)))
    for undefined in (lambda: A + F, lambda: A @ F, lambda: F - A):
        with pytest.raises(TypeError):
            undefined()


def test_derived_forms(ops):
    ctx, A, B, F = ops
    x = ctx.asarray([1.0, -2.0, 0.5])
    np.testing.assert_allclose((A - B)(x), A(x) - B(x))
    np.testing.assert_allclose((-A)(x), -A(x))
    np.testing.assert_allclose((3.0 - F)(x), 3.0 - F(x))
    assert A + 0 is A and 0 + F is F and sum([A, B]) == A + B


class _Num(Mapping):
    """A toy family: registering one ``add`` and one ``mul`` rule buys the whole algebra."""

    def __init__(self, v):
        super().__init__()
        self.v = v

    domain = codomain = None
    tree_flatten = lambda self: ((self.v,), None)  # noqa: E731
    tree_unflatten = classmethod(lambda cls, aux, children: cls(*children))


register("add", _Num, _Num, lambda a, b: _Num(a.v + b.v))
register("mul", Scalar, _Num, lambda s, a: _Num(s * a.v))


def test_new_family_plugs_in_by_registering_rules():
    a, b = _Num(5.0), _Num(2.0)
    assert (a - b).v == 3.0 and (-a).v == -5.0 and sum([a, b]).v == 7.0
    with pytest.raises(TypeError):
        a * 2.0  # only scalar-on-the-left was registered
