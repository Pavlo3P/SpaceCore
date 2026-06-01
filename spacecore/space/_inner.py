from __future__ import annotations

from abc import ABC, abstractmethod


class InnerProduct(ABC):
    """Geometry of a space: inner product plus Riesz maps.

    The Riesz map sends a coordinate vector to its dual representation; its
    inverse sends a dual representation back to coordinates. The true adjoint
    of a coordinate operator ``A`` is ``R_X^{-1} A^dagger R_Y``.
    """

    @abstractmethod
    def inner(self, ops, x, y):
        """Return the inner product of ``x`` and ``y``."""

    def riesz(self, ops, x):
        """Map a coordinate element to its dual. Euclidean default: identity."""
        return x

    def riesz_inverse(self, ops, x):
        """Map a dual element back to coordinates. Euclidean default: identity."""
        return x

    def convert(self, ctx):
        """Return this geometry represented in ``ctx`` when conversion is needed."""
        return self

    @property
    def is_euclidean(self) -> bool:
        """Whether this geometry is the Euclidean coordinate inner product."""
        return False


class EuclideanInnerProduct(InnerProduct):
    """Standard coordinate inner product ``vdot(x, y)`` with identity Riesz maps."""

    def __eq__(self, other):
        """Return whether ``other`` is also Euclidean coordinate geometry."""
        return type(other) is type(self)

    def inner(self, ops, x, y):
        """Return the Euclidean coordinate inner product."""
        return ops.vdot(x, y)

    @property
    def is_euclidean(self) -> bool:
        """Return ``True`` for the standard Euclidean coordinate geometry."""
        return True
