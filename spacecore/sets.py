"""Closed sets with native metric projection; no iterative projection solver."""
from __future__ import annotations

from typing import Any

from ._errors import CapabilityError
from ._checks import checked_method
from .backend import PyTreeNode
from .contextual import ContextBound
from .space import DenseCoordinateSpace, EuclideanInnerProduct, WeightedInnerProduct


class Set(ContextBound, PyTreeNode):
    """
    A closed subset of an ambient space, with optional membership and projection.

    Parameters
    ----------
    ambient : Space
        Space owning elements and the projection metric.
    check_level : str or None, optional
        Validation policy; inherited from the ambient space by default.

    Notes
    -----
    A custom set implements ``project``, and optionally ``contains``. An
    indicator's conjugate-prox contract additionally requires convexity;
    custom sets must explicitly declare ``is_convex = True`` to assert it.
    """

    is_convex = False

    def __init__(self, ambient, *, check_level=None):
        (self.ambient,) = self._bind_context(
            ambient.ctx, ambient, check_level=check_level,
        )

    def contains(self, x: Any) -> Any:
        """Return a backend Boolean, or None when membership is unavailable."""
        return None

    def project(self, x: Any) -> Any:
        """Return a nearest element in the ambient metric, if implemented."""
        raise CapabilityError(f"{type(self).__name__} does not implement metric projection.")


class NonnegativeOrthant(Set):
    """
    Nonnegative real dense coordinates in Euclidean or diagonal geometry.

    Parameters
    ----------
    ambient : DenseCoordinateSpace
        Real coordinate space with a supported diagonal metric.
    check_level : str or None, optional
        Validation policy; inherited from the ambient space by default.
    """

    is_convex = True

    def __init__(self, ambient, *, check_level=None):
        if (not isinstance(ambient, DenseCoordinateSpace)
                or ambient.scalar_field != "real" or ambient.field != "real"
                or not isinstance(ambient.geometry, (EuclideanInnerProduct, WeightedInnerProduct))):
            raise CapabilityError(
                "NonnegativeOrthant requires real dense coordinates and Euclidean or diagonal geometry."
            )
        super().__init__(ambient, check_level=check_level)

    @checked_method(in_space="ambient")
    def contains(self, x: Any) -> Any:
        """Return whether every coordinate is nonnegative."""
        return self.ops.xp.all(x >= 0)

    @checked_method(in_space="ambient", out_space="ambient")
    def project(self, x: Any) -> Any:
        """Return the metric projection, coordinatewise ``max(x, 0)``."""
        return self.ops.maximum(x, self.ops.zeros_like(x))

    def tree_flatten(self):
        """Keep the ambient space and geometry as static pytree metadata."""
        return (), (self.ambient, self.check_level)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild from an ambient space and validation policy."""
        ambient, check_level = aux
        return cls(ambient, check_level=check_level)

    def _convert(self, ctx):
        return type(self)(self.ambient.convert(ctx), check_level=self.check_level)
