from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, ClassVar, Tuple

from ..backend import Context
from .._contextual import ContextBound
from ..types import DenseArray
from ._checks import SpaceCheck


class Space(ContextBound):
    """
    Define the geometry and linear structure of a vector space.

    A space owns the geometry (inner product, norm) and the basic linear
    structure (add/scale/axpy) for its elements.

    Membership validation is exposed through ``check_member``, which respects
    the space's ``enable_checks`` policy. Internal code paths that have already
    checked that policy may call ``_check_member`` to run the concrete checks
    exactly once.

    Parameters
    ----------
    shape : tuple of int
        Canonical coordinate shape for elements of the space.
    ctx : Context, str, or None, optional
        Backend context specification. Default resolves to the global context.

    Attributes
    ----------
    shape : tuple of int
        Canonical element shape.
    ctx : Context
        Resolved backend context inherited from :class:`ContextBound`.

    Notes
    -----
    Solvers use only this API. Concrete spaces define storage constraints,
    membership checks, and flattening rules.

    Examples
    --------
    Instantiate the concrete :class:`VectorSpace` subclass.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.VectorSpace((2,), ctx)
    >>> X.shape
    (2,)
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
        """Raise if ``x`` is not a valid element of this space."""
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
        r"""Return :math:`\langle x, y \rangle_X` for elements of this space."""

    def norm(self, x: Any) -> Any:
        r"""Return the induced norm :math:`\sqrt{\operatorname{Re}\langle x, x\rangle_X}`."""
        v = self.ctx.ops.real(self.inner(x, x))
        return self.ctx.ops.sqrt(v)

    @abstractmethod
    def eigh(self, x: Any, k: int = None) -> Any:
        """Return an eigendecomposition of ``x`` when the space defines one."""

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

    def _convert(self, new_ctx: Context) -> Space:
        raise NotImplementedError()

    def apply(self, x: Any, f: Callable) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not define functional application."
        )
