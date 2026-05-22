from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, ClassVar, Tuple

from ..backend import Context
from .._contextual import ContextBound
from ..types import DenseArray
from ._checks import SpaceCheck


class Space(ContextBound):
    """
    Abstract Space.

    A Space owns the *geometry* (inner product, norm) and the basic linear
    structure (add/scale/axpy) for its elements.

    Membership validation is exposed through ``check_member``, which respects
    the space's ``enable_checks`` policy. Internal code paths that have already
    checked that policy may call ``_check_member`` to run the concrete checks
    exactly once.

    Solvers should use only this API.
    """

    checks: ClassVar[tuple[SpaceCheck, ...]] = ()

    def __init__(self, shape: Tuple[int, ...], ctx: Context | str | None = None) -> None:
        super().__init__(ctx)
        self.shape = shape
        self._enable_checks = self.ctx.enable_checks

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Space):
            return self.ctx == other.ctx and self.shape == other.shape
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
        """
        Raise if `x` is not a valid element of this space.

        Typical checks:
          - x.space is self (if your elements carry a .space)
          - backend family consistency (via ctx)
          - representation is supported
          - shape/structure constraints (Hermitian, block sizes, etc.)
        """
        for check in self.member_checks():
            check(self, x)

    def check_member(self, x: Any) -> None:
        if self._enable_checks:
            self._check_member(x)

    @abstractmethod
    def zeros(self) -> Any:
        """Return the additive identity in the requested representation."""

    @abstractmethod
    def add(self, x: Any, y: Any) -> Any:
        """Return x + y."""

    @abstractmethod
    def scale(self, a: Any, x: Any) -> Any:
        """Return a * x."""

    def axpy(self, a: Any, x: Any, y: Any) -> Any:
        """Return a*x + y."""
        return self.add(self.scale(a, x), y)

    @abstractmethod
    def inner(self, x: Any, y: Any) -> Any:
        """
        Inner product ⟨x, y⟩ for elements of this space.
        """

    def norm(self, x: Any) -> Any:
        """Induced norm ||x|| = sqrt(real(⟨x,x⟩)). Override if you can do better."""
        v = self.ctx.ops.real(self.inner(x, x))
        return self.ctx.ops.sqrt(v)

    @abstractmethod
    def eigh(self, x: Any, k: int = None) -> Any:
        """Eigendecomposition of x (if applicable).)"""

    @abstractmethod
    def flatten(self, x: Any) -> DenseArray:
        """
        Return a dense 1D coordinate vector (backend-native dense array).

        If a representation forbids materialization, raise a policy/capability error.
        """

    @abstractmethod
    def unflatten(self, v: DenseArray) -> Any:
        """Inverse of flatten; returns an element in the requested representation."""
        raise NotImplementedError

    def batch(
        self,
        batch_shape: Tuple[int, ...],
        batch_axes: Tuple[int, ...] | None = None,
    ) -> Space:
        """Return a wrapper representing a batch/product of this space."""
        from ._batch import BatchSpace

        if batch_axes is None:
            batch_axes = tuple(range(len(batch_shape)))
        return BatchSpace(self, batch_shape, batch_axes)

    def _convert(self, new_ctx: Context) -> Space:
        raise NotImplementedError()

    def apply(self, x: Any, f: Callable) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not define functional application."
        )
