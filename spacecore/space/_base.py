from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import Any, Callable, ClassVar, Tuple

from ..backend import Context
from .._contextual import ContextBound
from ..types import DenseArray
from ._checks import SpaceCheck, _run_checks
from ._inner import InnerProduct


class Space(ContextBound):
    """General space capability: context ownership and membership checks."""

    checks: ClassVar[tuple[SpaceCheck, ...]] = ()

    def __init__(self, ctx: Context | str | None = None) -> None:
        super().__init__(ctx)
        self._enable_checks = self.ctx.enable_checks

    def __eq__(self, other: Any) -> bool:
        if type(self) is type(other):
            return self.ctx == other.ctx
        return False

    def member_checks(self) -> tuple[SpaceCheck, ...]:
        checks: list[SpaceCheck] = []
        for klass in reversed(type(self).__mro__):
            checks.extend(klass.__dict__.get("checks", ()))
            local_checks = klass.__dict__.get("_local_checks")
            if local_checks is not None:
                checks.extend(local_checks(self))
        return tuple(checks)

    def _check_member(self, x: Any) -> None:
        """Raise if ``x`` is not a valid element of this space."""
        _run_checks(self, x, allow_leading=False)

    def check_member(self, x: Any) -> None:
        if self._enable_checks:
            self._check_member(x)

    def _convert(self, new_ctx: Context) -> Space:
        raise NotImplementedError()


class VectorSpace(Space):
    """Abstract vector-space capability: linear operations only."""

    @abstractmethod
    def zeros(self) -> Any:
        """Return the additive identity."""

    @abstractmethod
    def add(self, x: Any, y: Any) -> Any:
        """Return x + y."""

    @abstractmethod
    def scale(self, a: Any, x: Any) -> Any:
        """Return a * x."""

    def axpy(self, a: Any, x: Any, y: Any) -> Any:
        """Return a*x + y."""
        return self.add(self.scale(a, x), y)


class CoordinateSpace(VectorSpace):
    """Finite coordinate vector space capability."""

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
        from ._stacked import StackedSpace

        return StackedSpace(self, count, self.ctx)


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


class StarSpace(Space):
    """Space capability with a canonical involution/star operation."""

    @abstractmethod
    def star(self, x: Any) -> Any:
        """Return the canonical star/involution of ``x``."""


class JordanAlgebraSpace(VectorSpace):
    """Vector space capability with a Jordan product and spectral calculus."""

    @abstractmethod
    def jordan(self, x: Any, y: Any) -> Any:
        """Return the Jordan product of ``x`` and ``y``."""

    @abstractmethod
    def spectrum(self, x: Any) -> Any:
        """Return Jordan-algebraic eigenvalues of ``x``."""

    @abstractmethod
    def spectral_decompose(self, x: Any) -> Any:
        """Return spectral data sufficient to reconstruct ``x``."""

    @abstractmethod
    def from_spectrum(self, eigvals: Any, frame: Any) -> Any:
        """Reconstruct an element from spectral data."""

    def spectral_apply(self, x: Any, f: Callable) -> Any:
        eigvals, frame = self.spectral_decompose(x)
        feigvals = f(eigvals)
        return self.from_spectrum(feigvals, frame)

    def apply(self, x: Any, f: Callable) -> Any:
        """Backward-compatible alias for ``spectral_apply``."""
        return self.spectral_apply(x, f)


class EuclideanJordanAlgebraSpace(JordanAlgebraSpace, InnerProductSpace):
    """Jordan algebra capability with a compatible inner product."""
