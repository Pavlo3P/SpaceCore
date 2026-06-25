"""Tests for :class:`spacecore.LinearFunctional` — the linear-map base.

Checklist section 7, ``LinearFunctional``:

* ``grad(x)`` returns the *constant* Riesz representer, independent of ``x``.
* ``vgrad(xs)`` broadcasts that constant representer across the batch axis.
* A linear functional without a stored representer (matrix-free) inherits
  the ``NotImplementedError`` raised by :attr:`representer`.

The concrete value-level behaviour lives in
:mod:`tests.functional.test_inner_product_functional` and
:mod:`tests.functional.test_matrix_free_linear_functional`; here we exercise
the gradient contract provided by the abstract base, using
``InnerProductFunctional`` as the simplest concrete carrier of a representer.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy


# ===========================================================================
# grad is the constant representer
# ===========================================================================
class TestConstantGradient:
    def test_grad_returns_representer_independent_of_point(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([1.5, -0.25, 0.75])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)

        for point in ([0.0, 0.0, 0.0], [3.0, 4.0, -1.0], [-2.0, 0.5, 7.0]):
            np.testing.assert_allclose(
                to_numpy(f.grad(numpy_ctx.asarray(point))), to_numpy(c)
            )

    def test_grad_equals_representer_attribute(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([2.0, -1.0, 0.5])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)
        x = numpy_ctx.asarray([1.0, 1.0, 1.0])
        np.testing.assert_allclose(to_numpy(f.grad(x)), to_numpy(f.representer))


# ===========================================================================
# vgrad broadcasts the constant representer
# ===========================================================================
class TestBatchedGradient:
    def test_vgrad_broadcasts_representer_over_batch(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([1.5, -0.25, 0.75])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)
        xs = numpy_ctx.asarray([[0.5, -1.0, 2.0], [1.25, 0.75, -0.5], [-2.0, 0.25, 1.5]])

        out = f.vgrad(xs)
        assert out.shape == (3, 3)
        expected = np.broadcast_to(to_numpy(c), (3, 3))
        np.testing.assert_allclose(to_numpy(out), expected)

    def test_vgrad_matches_elementwise_grad(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([1.5, -0.25, 0.75])
        f = sc.InnerProductFunctional(c, space, numpy_ctx)
        xs = numpy_ctx.asarray([[0.5, -1.0, 2.0], [1.25, 0.75, -0.5]])
        expected = f.ops.stack(tuple(f.grad(x) for x in xs), axis=0)
        np.testing.assert_allclose(to_numpy(f.vgrad(xs)), to_numpy(expected))


# ===========================================================================
# Matrix-free linear functionals have no representer
# ===========================================================================
class TestNoRepresenter:
    def test_grad_propagates_not_implemented_without_representer(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        c = numpy_ctx.asarray([1.0, -2.0, 3.0])
        f = sc.MatrixFreeLinearFunctional(lambda y: space.inner(c, y), space, numpy_ctx)

        with pytest.raises(NotImplementedError):
            f.grad(numpy_ctx.asarray([1.0, 1.0, 1.0]))
