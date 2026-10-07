from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as _np
from typing import Any

from ._coordinate import CoordinateSpace
from ._vector import VectorSpace


class InnerProduct(ABC):
    """
    Geometry of a space: inner product plus Riesz maps.

    An orthonormal basis gives a representer for each linear functional
    (Horn & Johnson, *Matrix Analysis*, 2nd ed., Sec. 0.6.5).
    In coordinates, ``<x, y> = x^H R y`` with positive-definite Gram matrix ``R``;
    ``riesz(x) = R x`` stores the column in ``<x, y> = (R x)^H y``.
    """

    @abstractmethod
    def inner(self, ops, x, y):
        """Return the inner product of ``x`` and ``y``."""

    def riesz(self, ops, x):
        """Map an element to its dual; non-Euclidean geometry must implement this."""
        from ..._errors import CapabilityError

        raise CapabilityError(f"{type(self).__name__} does not implement riesz.")

    def riesz_inverse(self, ops, x):
        """Invert the Riesz map; non-Euclidean geometry must implement this."""
        from ..._errors import CapabilityError

        raise CapabilityError(f"{type(self).__name__} does not implement riesz_inverse.")

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

    def riesz(self, ops, x):
        """Return the Euclidean representer unchanged."""
        return x

    def riesz_inverse(self, ops, x):
        """Return the Euclidean covector unchanged."""
        return x

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
        # Same class and same weights; any failure to compare means unequal.
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
        # Structure first: dense, exact shape, same dtype as the space.
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
        # Then values, on the host: real, finite, strictly positive.
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


class OperatorInnerProduct(InnerProduct):
    r"""
    Operator-metric geometry :math:`\langle x, y\rangle = \langle x, M y\rangle_E`.

    ``M`` is a self-adjoint, positive-definite linear operator on the space's
    Euclidean coordinates: block, Kronecker, low-rank-plus-diagonal or any
    matrix-free metric. The Riesz map is ``M``; its inverse is ``M_inverse``, which
    may be any operator approximating it, such as one that runs ``sc.cg`` to a
    tolerance. Without ``M_inverse``, ``riesz_inverse`` raises ``CapabilityError``.

    Parameters
    ----------
    M : LinOp
        Self-adjoint positive-definite operator from the Euclidean coordinate space
        of the target space to itself.
    M_inverse : LinOp or None, optional
        Operator applying (an approximation of) the inverse of ``M``.
    """

    def __init__(self, M, M_inverse=None):
        self.M = M
        self.M_inverse = M_inverse

    def __eq__(self, other):
        return (type(other) is type(self) and self.M == other.M
                and self.M_inverse == other.M_inverse)

    def __repr__(self) -> str:
        return f"OperatorInnerProduct(M={self.M._short_repr()})"

    @staticmethod
    def _act(op, x):
        # A single element, or a leading-axis batch folded into one batch axis.
        core = tuple(op.domain.shape)
        lead = tuple(x.shape[: len(x.shape) - len(core)])
        if not lead:
            return op._apply_core(x)
        return op._vapply_core(x.reshape((-1,) + core)).reshape(lead + core)

    def inner(self, ops, x, y):
        return ops.vdot(x, self._act(self.M, y))

    def riesz(self, ops, x):
        return self._act(self.M, x)

    def riesz_inverse(self, ops, x):
        if self.M_inverse is None:
            from ..._errors import CapabilityError

            raise CapabilityError("OperatorInnerProduct needs M_inverse to invert its Riesz map.")
        return self._act(self.M_inverse, x)

    def convert(self, ctx):
        M_inverse = None if self.M_inverse is None else self.M_inverse.convert(ctx)
        return OperatorInnerProduct(self.M.convert(ctx), M_inverse)

    def validate_for(self, space: InnerProductSpace) -> None:
        """Check that ``M`` (and ``M_inverse``) act on the space's Euclidean coordinates."""
        if not isinstance(space, CoordinateSpace):
            raise TypeError("OperatorInnerProduct requires a CoordinateSpace.")
        for name, op in (("M", self.M), ("M_inverse", self.M_inverse)):
            if op is None:
                continue
            for end in (op.domain, op.codomain):
                if tuple(end.shape) != tuple(space.shape) or not end.is_euclidean:
                    raise ValueError(
                        f"OperatorInnerProduct {name} must map the Euclidean coordinate space "
                        f"of shape {space.shape} to itself."
                    )
            if op.ops.family != space.ops.family or op.dtype != space.dtype:
                raise TypeError(
                    f"OperatorInnerProduct {name} must use the space backend and dtype "
                    f"({space.ops.family}, {space.dtype})."
                )
        if space._checks_at_least("standard") and self.M.is_hermitian() is False:
            raise ValueError("OperatorInnerProduct M must be self-adjoint.")


class InnerProductSpace(VectorSpace):
    """
    Vector space capability with an inner-product geometry.

    Parameters
    ----------
    ctx : Context, str, or None, optional
        Context specification used for elements and validation checks.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    #: Optional. A space without one overrides :meth:`inner`, and its elements are
    #: their own Riesz representers (identity Riesz maps).
    geometry: InnerProduct

    def inner(self, x: Any, y: Any) -> Any:
        geometry = getattr(self, "geometry", None)
        if geometry is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no geometry; override inner() or set one."
            )
        return geometry.inner(self.ops, x, y)

    def real_inner(self, x: Any, y: Any) -> Any:
        r"""Return the real inner product :math:`\operatorname{Re}\langle x, y\rangle_X`."""
        return self.ops.real(self.inner(x, y))

    def _real_inner_core(self, x: Any, y: Any) -> Any:
        """Return :meth:`real_inner` without membership checks when the space has a core."""
        return self.ops.real(getattr(self, "_inner_core", self.inner)(x, y))

    def riesz(self, x: Any) -> Any:
        geometry = getattr(self, "geometry", None)
        return x if geometry is None else geometry.riesz(self.ops, x)

    def riesz_inverse(self, x: Any) -> Any:
        geometry = getattr(self, "geometry", None)
        return x if geometry is None else geometry.riesz_inverse(self.ops, x)

    def norm(self, x: Any) -> Any:
        return self.ops.sqrt(self.real_inner(x, x))

    @property
    def is_euclidean(self) -> bool:
        # Without a geometry there is no coordinate inner product to coincide with.
        geometry = getattr(self, "geometry", None)
        return geometry is not None and geometry.is_euclidean
