from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as _np
from typing import Any

from ._vector import VectorSpace


class InnerProduct(ABC):
    """Geometry of a space: inner product plus Riesz maps."""

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
        return type(other) is type(self)

    def inner(self, ops, x, y):
        return ops.vdot(x, y)

    @property
    def is_euclidean(self) -> bool:
        return True


class WeightedInnerProduct(InnerProduct):
    """Diagonal-metric geometry ``<x, y> = vdot(x, weights * y)``."""

    def __init__(self, weights):
        self.weights = weights

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        try:
            return bool(_np.allclose(_np.asarray(self.weights), _np.asarray(other.weights)))
        except Exception:
            return False

    def inner(self, ops, x, y):
        return ops.vdot(x, self.weights * y)

    def riesz(self, ops, x):
        return self.weights * x

    def riesz_inverse(self, ops, x):
        return x / self.weights

    def convert(self, ctx):
        return WeightedInnerProduct(ctx.asarray(self.weights))

    @property
    def is_euclidean(self) -> bool:
        return False


class InnerProductSpace(VectorSpace):
    """Vector space capability with an inner-product geometry."""

    geometry: InnerProduct

    def inner(self, x: Any, y: Any) -> Any:
        return self.geometry.inner(self.ops, x, y)

    def riesz(self, x: Any) -> Any:
        return self.geometry.riesz(self.ops, x)

    def riesz_inverse(self, x: Any) -> Any:
        return self.geometry.riesz_inverse(self.ops, x)

    def norm(self, x: Any) -> Any:
        value = self.ops.real(self.inner(x, x))
        return self.ops.sqrt(value)

    @property
    def is_euclidean(self) -> bool:
        return self.geometry.is_euclidean
