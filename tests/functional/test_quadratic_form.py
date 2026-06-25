"""Tests for :class:`spacecore.QuadraticForm` — the quadratic-objective base.

Checklist section 7, ``QuadraticForm`` base:

* ``grad`` / ``hess_apply`` raise ``NotImplementedError`` until a subclass
  provides them.
* ``vgrad`` is provided by the base via the ``vmap`` fallback and matches an
  element-wise ``grad`` loop once a subclass defines ``grad``.

The fully-featured operator-backed implementation is covered by
:mod:`tests.functional.test_linop_quadratic_form`.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


class _BareQuadratic(sc.QuadraticForm):
    """Quadratic form that defines only ``value`` (no grad/hess)."""

    def value(self, x):
        return 0.5 * self.ops.sum(x * x)

    def tree_flatten(self):
        return (), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, ctx = aux
        return cls(domain, ctx)

    def _convert(self, new_ctx):
        return _BareQuadratic(self.domain.convert(new_ctx), new_ctx)


class _DiagQuadratic(sc.QuadraticForm):
    """Quadratic form ``f(x) = 1/2 * sum(diag * x**2)`` with an explicit grad."""

    def __init__(self, diag, dom, ctx=None):
        super().__init__(dom, ctx)
        self._diag = self.ctx.asarray(diag)

    def value(self, x):
        return 0.5 * self.ops.sum(x * (self._diag * x))

    def grad(self, x):
        return self._diag * x

    def hess_apply(self, x):
        return self._diag * x

    def tree_flatten(self):
        return (self._diag,), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, ctx = aux
        return cls(children[0], domain, ctx)

    def _convert(self, new_ctx):
        return _DiagQuadratic(self._diag, self.domain.convert(new_ctx), new_ctx)


# ===========================================================================
# Default grad / hess_apply are not implemented
# ===========================================================================
class TestUnimplementedDefaults:
    def test_grad_raises_by_default(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        q = _BareQuadratic(space, numpy_ctx)
        with pytest.raises(NotImplementedError, match="grad"):
            q.grad(numpy_ctx.asarray([1.0, 2.0, 3.0]))

    def test_hess_apply_raises_by_default(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        q = _BareQuadratic(space, numpy_ctx)
        with pytest.raises(NotImplementedError, match="hess_apply"):
            q.hess_apply(numpy_ctx.asarray([1.0, 2.0, 3.0]))


# ===========================================================================
# value / grad / hess_apply on a concrete subclass
# ===========================================================================
class TestConcreteSubclass:
    def test_value_is_half_quadratic_form(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        diag = numpy_ctx.asarray([2.0, 4.0, 0.5])
        q = _DiagQuadratic(diag, space, numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 4.0])
        # 0.5 * (2*1 + 4*4 + 0.5*16) = 0.5 * (2 + 16 + 8) = 13.
        np.testing.assert_allclose(to_numpy(q.value(x)), 13.0)

    def test_grad_and_hess_match_diag_action(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        diag = numpy_ctx.asarray([2.0, 4.0, 0.5])
        q = _DiagQuadratic(diag, space, numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 4.0])
        np.testing.assert_allclose(to_numpy(q.grad(x)), [2.0, 8.0, 2.0])
        np.testing.assert_allclose(to_numpy(q.hess_apply(x)), [2.0, 8.0, 2.0])


# ===========================================================================
# vgrad fallback (base, via ops.vmap over grad)
# ===========================================================================
class TestVGradFallback:
    def test_vgrad_matches_elementwise_grad(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        diag = numpy_ctx.asarray([2.0, 4.0, 0.5])
        q = _DiagQuadratic(diag, space, numpy_ctx)
        xs = numpy_ctx.asarray([[1.0, 2.0, 4.0], [0.0, -1.0, 2.0], [3.0, 0.5, -2.0]])
        expected = q.ops.stack(tuple(q.grad(x) for x in xs), axis=0)
        np.testing.assert_allclose(to_numpy(q.vgrad(xs)), to_numpy(expected))
