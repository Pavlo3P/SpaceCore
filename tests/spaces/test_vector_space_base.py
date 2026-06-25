"""Tests for :class:`spacecore.VectorSpace` — abstract linear-capability base.

Checklist item 2:

* ``VectorSpace`` is abstract — direct instantiation raises.
* Subclasses must implement ``zeros`` / ``add`` / ``scale``.
* ``axpy(a, x, y) = a*x + y`` identity (provided by the base).
* A ``VectorSpace`` that is not a ``CoordinateSpace`` is valid — only
  ``zeros``/``add``/``scale`` are required.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


class _PairVectorSpace(sc.VectorSpace):
    """Non-coordinate concrete subclass: elements are Python ``(a, b)`` tuples."""

    def zeros(self):
        return (0.0, 0.0)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def scale(self, a, x):
        return (a * x[0], a * x[1])

    def _convert(self, new_ctx):
        return _PairVectorSpace(new_ctx)


# ===========================================================================
# Abstract enforcement
# ===========================================================================
class TestAbstract:
    def test_vector_space_is_not_directly_instantiable(self):
        """``VectorSpace()`` raises — the linear-capability base is abstract."""
        with pytest.raises(TypeError):
            sc.VectorSpace()

    def test_pair_vector_space_is_a_vector_space(self):
        """A concrete subclass with zeros/add/scale satisfies the contract."""
        space = _PairVectorSpace()
        assert isinstance(space, sc.VectorSpace)
        assert not isinstance(space, sc.CoordinateSpace)


# ===========================================================================
# Linear operations
# ===========================================================================
class TestLinearOperations:
    def test_zeros_returns_neutral_element(self):
        space = _PairVectorSpace()
        assert space.zeros() == (0.0, 0.0)

    def test_add_is_componentwise(self):
        space = _PairVectorSpace()
        assert space.add((1.0, 2.0), (3.0, 4.0)) == (4.0, 6.0)

    def test_scale_is_componentwise(self):
        space = _PairVectorSpace()
        assert space.scale(3.0, (1.0, 2.0)) == (3.0, 6.0)


# ===========================================================================
# axpy: provided by the base
# ===========================================================================
class TestAxpy:
    def test_axpy_identity_on_pair_space(self):
        """``axpy(a, x, y) = a*x + y``."""
        space = _PairVectorSpace()
        assert space.axpy(2.0, (1.0, 3.0), (-1.0, 4.0)) == (1.0, 10.0)

    def test_axpy_identity_on_dense_coordinate_space(self, numpy_ctx):
        space = sc.DenseCoordinateSpace((3,), numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0, 3.0])
        y = numpy_ctx.asarray([10.0, 20.0, 30.0])
        out = space.axpy(2.0, x, y)
        np.testing.assert_allclose(out, [12.0, 24.0, 36.0])
