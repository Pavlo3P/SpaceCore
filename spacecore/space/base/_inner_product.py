from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as _np
from typing import Any

from ._coordinate import CoordinateSpace
from ._vector import VectorSpace


# Strict-level tolerances: positivity relative to the largest eigenvalue, and how
# closely a supplied M_inverse must invert M.
_STRICT_RTOL = 1e-12
_STRICT_RTOL_INVERSE = 1e-6


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

    def norm(self, ops, x):
        """Return the norm ``sqrt(Re <x, x>)`` of this geometry."""
        return ops.sqrt(ops.real(self.inner(ops, x, x)))

    def dual(self) -> InnerProduct:
        """Return the dual geometry on covectors, whose Riesz map is this one's inverse."""
        from ..._errors import CapabilityError

        raise CapabilityError(f"{type(self).__name__} does not define its dual geometry.")

    def dual_norm(self, ops, xi):
        r"""Return the dual norm :math:`\|\xi\|_* = \sqrt{\langle \xi, R^{-1}\xi\rangle}` of a covector."""
        return ops.sqrt(ops.real(ops.vdot(xi, self.riesz_inverse(ops, xi))))

    def factor(self, ops, x):
        """Apply a factor ``C`` with ``R = C^* C``, an isometry onto Euclidean coordinates."""
        from ..._errors import CapabilityError

        raise CapabilityError(f"{type(self).__name__} does not provide a metric factor.")

    def factor_inverse(self, ops, y):
        """Apply the inverse of :meth:`factor`."""
        from ..._errors import CapabilityError

        raise CapabilityError(f"{type(self).__name__} does not provide an inverse metric factor.")

    @property
    def has_riesz_maps(self) -> bool:
        """Whether both Riesz maps are available, so metric adjoints can be formed."""
        cls = type(self)
        return cls.riesz is not InnerProduct.riesz and cls.riesz_inverse is not InnerProduct.riesz_inverse


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

    def dual(self) -> EuclideanInnerProduct:
        return self

    def factor(self, ops, x):
        return x

    def factor_inverse(self, ops, y):
        return y

    @property
    def is_euclidean(self) -> bool:
        return True


class WeightedInnerProduct(InnerProduct):
    """
    Diagonal-metric geometry ``<x, y> = vdot(x, weights * y)``.

    Parameters
    ----------
    weights : array-like
        Positive, finite diagonal weights in the context dtype, with a shape that
        broadcasts to the coordinate-space shape (a 0-d array gives a scaled
        Euclidean metric). Values are checked from ``standard`` checks upward.
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
        # The same metric may be stored at different broadcast shapes; compare the
        # broadcast values, but never let incompatible shapes compare equal.
        try:
            lhs, rhs = _np.broadcast_arrays(lhs, rhs)
        except ValueError:
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

    def dual(self) -> WeightedInnerProduct:
        return WeightedInnerProduct(1 / self.weights)

    def factor(self, ops, x):
        return ops.sqrt(self.weights) * x

    def factor_inverse(self, ops, y):
        return y / ops.sqrt(self.weights)

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
        shape = tuple(getattr(weights, "shape", ()))
        try:
            broadcasts = _np.broadcast_shapes(shape, tuple(space.shape)) == tuple(space.shape)
        except ValueError:
            broadcasts = False
        if not broadcasts:
            raise ValueError(
                "WeightedInnerProduct weights must broadcast to the coordinate shape "
                f"{space.shape}; got {shape}."
            )
        dtype = space.ops.get_dtype(weights)
        if dtype != space.dtype:
            raise TypeError(
                "WeightedInnerProduct weights must use the same context dtype as the space; "
                f"got {dtype}, expected {space.dtype}."
            )
        # Then values, on the host: real, finite, strictly positive.
        if not space._checks_at_least("standard"):
            return
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
    tolerance. Without ``M_inverse``, ``riesz_inverse`` and ``dual`` raise
    ``CapabilityError``.

    From ``standard`` checks upward, an ``M`` known to be non-self-adjoint, or with
    non-positive curvature on a fixed probe vector, is rejected; at ``strict``, its
    Lanczos estimate of the smallest eigenvalue must also be positive.

    ``factor`` (with ``factor_inverse``) declares ``C`` with ``M = C^* C``.

    Parameters
    ----------
    M : LinOp
        Self-adjoint positive-definite operator from the Euclidean coordinate space
        of the target space to itself.
    M_inverse : LinOp or None, optional
        Operator applying (an approximation of) the inverse of ``M``.
    factor, factor_inverse : LinOp or None, optional
        A factor ``C`` with ``M = C^* C`` and its inverse.
    """

    def __init__(self, M, M_inverse=None, *, factor=None, factor_inverse=None):
        self.M = M
        self.M_inverse = M_inverse
        self.factor_op = factor
        self.factor_inverse_op = factor_inverse

    def __eq__(self, other):
        return (type(other) is type(self) and self.M == other.M
                and self.M_inverse == other.M_inverse and self.factor_op == other.factor_op
                and self.factor_inverse_op == other.factor_inverse_op)

    def __repr__(self) -> str:
        return f"OperatorInnerProduct(M={self.M._short_repr()})"

    @staticmethod
    def _missing(what):
        from ..._errors import CapabilityError

        return CapabilityError(f"OperatorInnerProduct needs {what}.")

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

    @property
    def has_riesz_maps(self) -> bool:
        return self.M_inverse is not None

    def riesz_inverse(self, ops, x):
        if self.M_inverse is None:
            raise self._missing("M_inverse to invert its Riesz map")
        return self._act(self.M_inverse, x)

    def dual(self) -> OperatorInnerProduct:
        if self.M_inverse is None:
            raise self._missing("M_inverse to form its dual geometry")
        return OperatorInnerProduct(self.M_inverse, self.M)

    def factor(self, ops, x):
        if self.factor_op is None:
            raise self._missing("factor= to apply a metric factor")
        return self._act(self.factor_op, x)

    def factor_inverse(self, ops, y):
        if self.factor_inverse_op is None:
            raise self._missing("factor_inverse= to invert its metric factor")
        return self._act(self.factor_inverse_op, y)

    def convert(self, ctx):
        def conv(op):
            return None if op is None else op.convert(ctx)

        return OperatorInnerProduct(self.M.convert(ctx), conv(self.M_inverse), factor=conv(self.factor_op),
                                    factor_inverse=conv(self.factor_inverse_op))

    def validate_for(self, space: InnerProductSpace) -> None:
        """Check that ``M`` (and ``M_inverse``) act on the space's Euclidean coordinates."""
        if not isinstance(space, CoordinateSpace):
            raise TypeError("OperatorInnerProduct requires a CoordinateSpace.")
        for name, op in (("M", self.M), ("M_inverse", self.M_inverse), ("factor", self.factor_op),
                         ("factor_inverse", self.factor_inverse_op)):
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
        if not space._checks_at_least("standard"):
            return
        if self.M.is_hermitian() is False:
            raise ValueError("OperatorInnerProduct M must be self-adjoint.")
        # A deterministic probe: positive curvature is necessary for positive definiteness.
        probe = space.ctx.asarray(_np.random.default_rng(0).standard_normal(tuple(space.shape)))
        if not float(space.ops.real(space.ops.vdot(probe, self.M._apply_core(probe)))) > 0:
            raise ValueError("OperatorInnerProduct M must be positive definite; "
                             "its quadratic form is not positive on a probe vector.")
        if not space._checks_at_least("strict"):
            return
        from ...linalg import lanczos_smallest, power_iteration

        iters = min(space.size, 100)
        smallest = float(space.ops.real(lanczos_smallest(self.M, probe, max_iter=iters).eigenvalue))
        largest = float(space.ops.real(power_iteration(self.M, x0=probe, maxiter=iters).eigenvalue))
        if not smallest > _STRICT_RTOL * largest:
            raise ValueError("OperatorInnerProduct M must be positive definite; its eigenvalue estimates "
                             f"are {smallest:.3g} (smallest) and {largest:.3g} (largest).")
        if self.M_inverse is not None:
            # M_inverse may be an inexact solve, but at strict it must invert M to this tolerance.
            defect = self.M_inverse._apply_core(self.M._apply_core(probe)) - probe
            if not float(space.ops.norm(defect)) <= _STRICT_RTOL_INVERSE * float(space.ops.norm(probe)):
                raise ValueError("OperatorInnerProduct M_inverse does not invert M on a probe vector.")
        if self.factor_op is not None:
            # ||C p||^2 must equal <p, M p>.
            cp = self.factor_op._apply_core(probe)
            lhs = float(space.ops.real(space.ops.vdot(cp, cp)))
            rhs = float(space.ops.real(space.ops.vdot(probe, self.M._apply_core(probe))))
            if abs(lhs - rhs) > _STRICT_RTOL_INVERSE * abs(rhs):
                raise ValueError("OperatorInnerProduct factor C does not satisfy M = C^* C on a probe vector.")


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

    def dual_norm(self, xi: Any) -> Any:
        r"""Return the dual norm of a covector, :math:`\|\xi\|_* = \|R^{-1}\xi\|`."""
        return self.norm(self.riesz_inverse(xi))

    @property
    def is_euclidean(self) -> bool:
        # Without a geometry there is no coordinate inner product to coincide with.
        geometry = getattr(self, "geometry", None)
        return geometry is not None and geometry.is_euclidean
