from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import Any, Callable, ClassVar, Tuple

from ..backend import Context
from .._contextual import ContextBound
from ..types import DenseArray
from ._checks import SpaceCheck
from ._inner import EuclideanInnerProduct, InnerProduct


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
    geometry : InnerProduct or None, optional
        Inner-product geometry for this space, including Riesz maps used by
        metric-aware adjoints. Defaults to Euclidean coordinate geometry.

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

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ) -> None:
        super().__init__(ctx)
        self.shape = shape
        self.geometry: InnerProduct = geometry if geometry is not None else EuclideanInnerProduct()
        self._enable_checks = self.ctx.enable_checks

    def __eq__(self, other: Any) -> bool:
        if type(self) is type(other):
            return (
                self.ctx == other.ctx
                and self.shape == other.shape
                and type(self.geometry) is type(other.geometry)
                and self.geometry == other.geometry
            )
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

    def add_batch(self, x: Any, y: Any) -> Any:
        """Return the leading-axis batch sum of ``x`` and ``y``."""
        return self.ops.vmap(self.add, in_axes=(0, 0), out_axes=0)(x, y)

    @abstractmethod
    def scale(self, a: Any, x: Any) -> Any:
        """Return a * x."""

    def scale_batch(self, a: Any, x: Any) -> Any:
        """Return the leading-axis batch scalar product ``a * x``."""
        return self.ops.vmap(lambda xi: self.scale(a, xi), in_axes=0, out_axes=0)(x)

    def axpy(self, a: Any, x: Any, y: Any) -> Any:
        """Return a*x + y."""
        return self.add(self.scale(a, x), y)

    @abstractmethod
    def inner(self, x: Any, y: Any) -> Any:
        r"""Return :math:`\langle x, y \rangle_X` for elements of this space."""

    def riesz(self, x: Any) -> Any:
        """Map a coordinate element to its dual representation."""
        return self.geometry.riesz(self.ops, x)

    def riesz_inverse(self, x: Any) -> Any:
        """Map a dual representation back to coordinate elements."""
        return self.geometry.riesz_inverse(self.ops, x)

    @property
    def is_euclidean(self) -> bool:
        """Return whether this space uses Euclidean coordinate geometry."""
        return self.geometry.is_euclidean

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

    @property
    def size(self) -> int:
        """Return the flat coordinate dimension of this space."""
        return prod(self.shape)

    def _convert(self, new_ctx: Context) -> Space:
        raise NotImplementedError()

    def apply(self, x: Any, f: Callable) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not define functional application."
        )
