"""Functionals: users write ``_*_core``; every public evaluator is the checked wrapper."""
import numpy as np
import pytest

import spacecore as sc


def _space(level="standard"):
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    return sc.DenseCoordinateSpace((2,), ctx, check_level=level)


class _CoresOnly(sc.Functional):
    """``f(x) = <x, x>``, written with cores only."""

    _fields = ("dom", "ctx")

    def _value_core(self, x):
        return self.ops.sum(x * x)

    def _grad_core(self, x):
        return 2.0 * x


class _PublicGradOverride(_CoresOnly):
    def grad(self, x):  # undecorated: the base must still check it
        return 2.0 * x


@pytest.mark.parametrize("cls", [_CoresOnly, _PublicGradOverride])
def test_public_evaluators_check_their_input(cls):
    X = _space()
    f = cls(X)
    bad = np.zeros(3)
    for evaluate in (f.value, f.grad, f.value_and_grad, f.derivative_at, f.differential):
        with pytest.raises(sc.SpaceValidationError):
            evaluate(bad)
    np.testing.assert_allclose(f.grad(X.ctx.asarray([1.0, 2.0])), [2.0, 4.0])


def test_value_and_grad_checks_the_gradient_output():
    class _WrongGradShape(_CoresOnly):
        def _grad_core(self, x):
            return np.zeros(3)

    with pytest.raises(sc.SpaceValidationError):
        _WrongGradShape(_space()).value_and_grad(np.zeros(2))


def test_unchecked_level_skips_the_wrappers():
    f = _CoresOnly(_space("none"))
    np.testing.assert_allclose(f.grad(np.zeros(3)), np.zeros(3))


def test_a_sum_checks_its_input_once_not_per_term(monkeypatch):
    X = _space()
    calls = []
    original = type(X)._check_member
    monkeypatch.setattr(type(X), "_check_member", lambda self, x: (calls.append(1), original(self, x))[1])
    f = sum((_CoresOnly(X) for _ in range(4)), start=_CoresOnly(X))
    calls.clear()
    f.grad(X.ctx.asarray([1.0, 2.0]))
    assert len(calls) == 2  # the input once, the gradient once


def test_prox_accepts_its_argument_by_keyword_and_checks_it():
    X = _space()
    f = sc.L1NormFunctional(X)
    np.testing.assert_allclose(f.prox(x=X.ctx.asarray([2.0, -0.1]), step=0.5), [1.5, 0.0])
    with pytest.raises(sc.SpaceValidationError):
        f.prox(x=np.zeros(3), step=0.5)
