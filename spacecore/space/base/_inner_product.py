from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as _np
from typing import Any

from ._coordinate import CoordinateSpace
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

    def validate_for(self, space: InnerProductSpace) -> None:
        """Raise if this inner product is invalid for ``space``."""

    @property
    def is_euclidean(self) -> bool:
        """Whether this geometry is the Euclidean coordinate inner product."""
        return False


class EuclideanInnerProduct(InnerProduct):
    """Standard coordinate inner product ``vdot(x, y)`` with identity Riesz maps."""

    def __eq__(self, other):
        return type(other) is type(self)

    def __repr__(self) -> str:
        return "EuclideanInnerProduct()"

    def inner(self, ops, x, y):
        return ops.vdot(x, y)

    @property
    def is_euclidean(self) -> bool:
        return True


class WeightedInnerProduct(InnerProduct):
    """
    Diagonal-metric geometry ``<x, y> = vdot(x, weights * y)``.

    Parameters
    ----------
    weights : array-like
        Positive, finite diagonal weights with exactly the coordinate-space
        shape and context dtype.
    """

    def __init__(self, weights):
        self.weights = weights

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        try:
            lhs = _np.asarray(self.weights)
            rhs = _np.asarray(other.weights)
        except Exception:
            return False
        # Structural before numerical: guard against allclose broadcasting two
        # mismatched-shape weight vectors into a spurious True.
        if lhs.shape != rhs.shape:
            return False
        try:
            return bool(_np.allclose(lhs, rhs))
        except Exception:
            return False

    def __repr__(self) -> str:
        from ..._repr import summarize_value

        return f"WeightedInnerProduct(weights={summarize_value(self.weights)})"

    def inner(self, ops, x, y):
        return ops.vdot(x, self.weights * y)

    def riesz(self, ops, x):
        return self.weights * x

    def riesz_inverse(self, ops, x):
        return x / self.weights

    def convert(self, ctx):
        return WeightedInnerProduct(ctx.asarray(self.weights))

    def validate_for(self, space: InnerProductSpace) -> None:
        """Validate diagonal weights against a coordinate space."""
        if not isinstance(space, CoordinateSpace):
            raise TypeError("WeightedInnerProduct requires a CoordinateSpace.")

        weights = self.weights
        if not space.ops.is_dense(weights):
            raise TypeError(
                "WeightedInnerProduct weights must be dense arrays for the space backend "
                f"{space.ops.family}."
            )
        if tuple(getattr(weights, "shape", ())) != tuple(space.shape):
            raise ValueError(
                "WeightedInnerProduct weights must have exactly the coordinate shape "
                f"{space.shape}; got {tuple(getattr(weights, 'shape', ()))}."
            )
        dtype = space.ops.get_dtype(weights)
        if dtype != space.dtype:
            raise TypeError(
                "WeightedInnerProduct weights must use the same context dtype as the space; "
                f"got {dtype}, expected {space.dtype}."
            )
        try:
            weights_np = _np.asarray(weights)
        except Exception as exc:
            raise TypeError(
                "WeightedInnerProduct weights must be host-readable for validation."
            ) from exc
        if space.ops.is_complex_dtype(dtype):
            if not _np.allclose(_np.imag(weights_np), 0):
                raise ValueError("WeightedInnerProduct weights must be real-valued.")
            weights_np = _np.real(weights_np)
        if not _np.all(_np.isfinite(weights_np)):
            raise ValueError("WeightedInnerProduct weights must be finite.")
        if not _np.all(weights_np > 0):
            raise ValueError("WeightedInnerProduct weights must be strictly positive.")

    @property
    def is_euclidean(self) -> bool:
        return False


class InnerProductSpace(VectorSpace):
    """
    Vector space capability with an inner-product geometry.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    """

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
