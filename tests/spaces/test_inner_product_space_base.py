"""Tests for :class:`spacecore.InnerProductSpace` — the inner-product-bearing base.

Checklist item 7:

* ``norm(x)² = inner(x, x)`` identity.
* Cauchy-Schwarz inequality on samples: ``|<x, y>| ≤ norm(x) · norm(y)``.
* Both invariants tested on Euclidean and weighted geometry,
  on coordinate and non-coordinate inner-product spaces.
"""
from __future__ import annotations

import numpy as np

import spacecore as sc

from tests._helpers import to_numpy


class _NonCoordinateInnerProductSpace(sc.InnerProductSpace):
    """Inner-product space whose elements are plain Python pairs.

    Used to exercise the InnerProductSpace base contract on
    non-coordinate elements (no shape, no flatten).
    """

    def __init__(self, ctx=None) -> None:
        super().__init__(ctx)
        self.geometry = sc.EuclideanInnerProduct()

    def zeros(self):
        return (0.0, 0.0)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def scale(self, a, x):
        return (a * x[0], a * x[1])

    def inner(self, x, y):
        return x[0] * y[0] + x[1] * y[1]

    def _convert(self, new_ctx):
        return _NonCoordinateInnerProductSpace(new_ctx)


# ===========================================================================
# norm² = inner(x, x)
# ===========================================================================
class TestNormFromInner:
    def test_norm_squared_equals_inner_xx_on_dense_real(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        n = float(to_numpy(space.norm(x)))
        ixx = float(to_numpy(space.inner(x, x)))
        np.testing.assert_allclose(n * n, ixx)

    def test_norm_squared_equals_inner_xx_on_dense_complex(self, numpy_complex_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_complex_ctx)
        x = numpy_complex_ctx.asarray([1 + 2j, -3 + 1j, 0.5 - 1j])
        n = float(to_numpy(space.norm(x)))
        ixx = float(to_numpy(space.inner(x, x)).real)
        np.testing.assert_allclose(n * n, ixx)

    def test_norm_squared_equals_inner_xx_on_weighted(self, numpy_ctx):
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0]))
        space = sc.DenseCoordinateSpace((2,), numpy_ctx, geometry=geom)
        x = numpy_ctx.asarray([1.0, 4.0])
        n = float(to_numpy(space.norm(x)))
        ixx = float(to_numpy(space.inner(x, x)))
        np.testing.assert_allclose(n * n, ixx)

    def test_norm_on_non_coordinate_space(self):
        """Non-coordinate inner-product spaces still get ``norm`` from the base."""
        space = _NonCoordinateInnerProductSpace()
        # ||(3, 4)|| = 5.
        assert float(space.norm((3.0, 4.0))) == 5.0


# ===========================================================================
# Cauchy-Schwarz
# ===========================================================================
class TestCauchySchwarz:
    def test_cauchy_schwarz_on_dense_real(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_ctx)
        rng = np.random.default_rng(0)
        for _ in range(5):
            x = numpy_ctx.asarray(rng.standard_normal(4))
            y = numpy_ctx.asarray(rng.standard_normal(4))
            lhs = abs(float(to_numpy(space.inner(x, y))))
            rhs = float(to_numpy(space.norm(x))) * float(to_numpy(space.norm(y)))
            assert lhs <= rhs + 1e-10

    def test_cauchy_schwarz_on_dense_complex(self, numpy_complex_ctx):
        space = sc.DenseCoordinateSpace((4,), numpy_complex_ctx)
        rng = np.random.default_rng(1)
        for _ in range(5):
            x_np = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            y_np = rng.standard_normal(4) + 1j * rng.standard_normal(4)
            x = numpy_complex_ctx.asarray(x_np)
            y = numpy_complex_ctx.asarray(y_np)
            lhs = abs(complex(to_numpy(space.inner(x, y))))
            rhs = float(to_numpy(space.norm(x))) * float(to_numpy(space.norm(y)))
            assert lhs <= rhs + 1e-10

    def test_cauchy_schwarz_on_weighted_real(self, numpy_ctx):
        geom = sc.WeightedInnerProduct(numpy_ctx.asarray([2.0, 3.0, 0.5]))
        space = sc.DenseCoordinateSpace((3,), numpy_ctx, geometry=geom)
        rng = np.random.default_rng(2)
        for _ in range(5):
            x = numpy_ctx.asarray(rng.standard_normal(3))
            y = numpy_ctx.asarray(rng.standard_normal(3))
            lhs = abs(float(to_numpy(space.inner(x, y))))
            rhs = float(to_numpy(space.norm(x))) * float(to_numpy(space.norm(y)))
            assert lhs <= rhs + 1e-10
