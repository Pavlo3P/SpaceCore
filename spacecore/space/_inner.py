from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as _np


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


class WeightedInnerProduct(InnerProduct):
    """Diagonal-metric geometry ``<x, y> = vdot(x, weights * y)``.

    ``weights`` must be positive elementwise and have the base coordinate shape
    of the space. Riesz maps use elementwise multiplication/division and
    therefore broadcast over leading batch axes.
    """

    def __init__(self, weights):
        self.weights = weights

    def __eq__(self, other):
        """Return whether another weighted geometry has matching weights."""
        if type(other) is not type(self):
            return False
        try:
            return bool(_np.allclose(_np.asarray(self.weights), _np.asarray(other.weights)))
        except Exception:
            return False

    def inner(self, ops, x, y):
        """Return ``vdot(x, weights * y)``."""
        return ops.vdot(x, self.weights * y)

    def riesz(self, ops, x):
        """Map coordinates to dual coordinates by multiplying by weights."""
        return self.weights * x

    def riesz_inverse(self, ops, x):
        """Map dual coordinates to coordinates by dividing by weights."""
        return x / self.weights

    def convert(self, ctx):
        """Convert stored weights to ``ctx``."""
        return WeightedInnerProduct(ctx.asarray(self.weights))

    @property
    def is_euclidean(self) -> bool:
        """Return ``False`` for nontrivial weighted geometry."""
        return False
