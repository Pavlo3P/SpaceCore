from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import Any, Tuple

from ...backend import Context
from ...types import DenseArray
from ._vector import VectorSpace


class CoordinateSpace(VectorSpace):
    """
    Finite coordinate vector space capability.

    Parameters
    ----------
    shape : tuple of int
        Canonical coordinate shape for one element of the space.
    ctx : Context, str, or None, optional
        Context specification used for coordinate arrays.
    """

    shape: Tuple[int, ...]

    def __init__(self, shape: Tuple[int, ...], ctx: Context | str | None = None) -> None:
        super().__init__(ctx)
        self.shape = tuple(shape)

    def __eq__(self, other: Any) -> bool:
        if type(self) is type(other):
            return super().__eq__(other) and self.shape == other.shape
        return False

    @property
    def size(self) -> int:
        """Return the flat coordinate dimension of this space."""
        return prod(self.shape)

    @abstractmethod
    def flatten(self, x: Any) -> DenseArray:
        """Return a dense one-dimensional coordinate vector."""

    @abstractmethod
    def unflatten(self, v: DenseArray) -> Any:
        """Inverse of flatten."""

    def flatten_batch(self, xs: Any) -> DenseArray:
        """Flatten a leading-axis batch of space elements to shape ``(N, size)``."""
        n = int(getattr(xs, "shape", (len(xs),))[0])
        rows = tuple(self.flatten(xs[i]) for i in range(n))
        return self.ops.stack(rows, axis=0)

    def unflatten_batch(self, vs: DenseArray) -> Any:
        """Unflatten rows of shape ``(N, size)`` into a leading-axis batch."""
        n = int(getattr(vs, "shape", (len(vs),))[0])
        xs = tuple(self.unflatten(vs[i]) for i in range(n))
        return self.ops.stack(xs, axis=0)

    def add_batch(self, x: Any, y: Any) -> Any:
        """Return the leading-axis batch sum of ``x`` and ``y``."""
        return self.ops.vmap(self.add, in_axes=(0, 0), out_axes=0)(x, y)

    def scale_batch(self, a: Any, x: Any) -> Any:
        """Return the leading-axis batch scalar product ``a * x``."""
        return self.ops.vmap(lambda xi: self.scale(a, xi), in_axes=0, out_axes=0)(x)

    def stacked(self, count: int) -> CoordinateSpace:
        """Return ``count`` leading-axis copies of this leaf space as one space."""
        from ..concrete import StackedSpace

        return StackedSpace(self, count, self.ctx)
