r"""
Real-coordinate view of a complex-domain functional.

For real-valued ``F`` with metric gradient ``g`` and ``v = a + i b``, the real
gradient is ``(Re g_v, Im g_v)`` where ``g_v = flatten(X.riesz(g))``. The view
writes that bookkeeping once, so optimizers that need a real space can use it.
Realification is an isometry, since ``Re<.,.>`` is a real inner product with the
same norm (Kreutz-Delgado, arXiv:0906.4835, section 4).
"""
from __future__ import annotations

from typing import Any, Self, Tuple

from .._checks import checked_method
from ..contextual import Context
from ..space import CoordinateSpace, DenseCoordinateSpace
from ._base import Functional


class RealifiedFunctional(Functional):
    r"""
    View of a complex-domain functional over stacked real coordinates ``(Re v, Im v)``.

    The real gradient is :math:`(\operatorname{Re} g_v, \operatorname{Im} g_v)` with
    :math:`g_v = \operatorname{flatten}(X.\operatorname{riesz}(g))`.

    Parameters
    ----------
    base : Functional
        Functional on a complex :class:`~spacecore.space.CoordinateSpace`.

    Raises
    ------
    TypeError
        If the base domain is not a coordinate space.
    ValueError
        If the base domain is already real.

    Notes
    -----
    :class:`~spacecore.HermitianSpace` stores complex entries but is a real space of
    dimension :math:`n^2`. Realifying it works but doubles the coordinates and leaves
    :math:`n^2` null Hessian directions, so optimize it directly instead.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    >>> X = sc.DenseCoordinateSpace((2,), ctx=ctx)
    >>> F = sc.SquaredL2NormFunctional(X)
    >>> R = sc.realify(F)
    >>> R.domain.shape
    (4,)
    >>> float(R.value(R.to_real(ctx.asarray([3.0 + 4.0j, 0.0]))))
    12.5
    """

    def __init__(self, base: Functional) -> None:
        # Only complex coordinate spaces can be split into real and imaginary parts.
        X = base.domain
        if not isinstance(X, CoordinateSpace):
            raise TypeError(
                "RealifiedFunctional requires a CoordinateSpace domain; "
                f"got {type(X).__name__}."
            )
        if X.field == "real":
            raise ValueError(
                "The base functional already has a real domain; use it directly."
            )
        ops = X.ops
        # check_level is a property of the bound object, not of the Context, so
        # it is threaded through the constructors rather than the context.
        # The view's domain: real coordinates of twice the size.
        real_ctx = Context(ops, dtype=ops.real_dtype(X.dtype))
        dom = DenseCoordinateSpace(
            (2 * X.size,), real_ctx, check_level=X.check_level
        )
        super().__init__(dom, real_ctx, check_level=X.check_level, cod=base.codomain)
        self.base = base
        self.complex_space = X

    def to_real(self, y: Any) -> Any:
        """Flatten a base-domain element into stacked real coordinates."""
        ops = self.ops
        v = self.complex_space.flatten(y)
        return ops.concatenate([ops.real(v), ops.imag(v)])

    def from_real(self, w: Any) -> Any:
        """Rebuild a base-domain element from stacked real coordinates."""
        m = self.complex_space.size
        v = w[:m] + 1j * w[m:]
        return self.complex_space.unflatten(self.complex_space.ctx.asarray(v))

    # Output-only: the input is validated by ``base.value`` against the *complex*
    # domain after ``from_real``, so an ``in_space`` here would check ``w``
    # against the wrong space.
    @checked_method(out_space="codomain")
    def value(self, w: Any, *args: Any, **kwargs: Any) -> Any:
        """Return ``base.value`` at the complex element ``w`` encodes."""
        return self.base.value(self.from_real(w), *args, **kwargs)

    def grad(self, w: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the stacked real gradient ``(Re g_v, Im g_v)``."""
        return self.value_and_grad(w, *args, **kwargs)[1]

    def value_and_grad(self, w: Any, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
        """Return ``(value, stacked real gradient)`` from one fused base evaluation.

        ``riesz`` maps the base's metric gradient back to coordinates before the
        real/imaginary split; skipping it would silently return the wrong vector
        on any non-Euclidean geometry.
        """
        X, ops = self.complex_space, self.ops
        # Evaluate the base at v = a + i b, then stack Re and Im of its coordinate gradient.
        val, g = self.base.value_and_grad(self.from_real(w), *args, **kwargs)
        gv = X.flatten(X.riesz(g))
        return val, ops.concatenate([ops.real(gv), ops.imag(gv)])

    def __eq__(self, other: Any) -> bool:
        """Return whether another realified functional wraps an equal base."""
        if not self.same_math(other):
            return NotImplemented
        return self.base == other.base

    def _convert(self, new_ctx: Context) -> "RealifiedFunctional":
        """Convert the wrapped functional to ``new_ctx`` and re-derive the view.

        ``new_ctx`` describes the *real view*, so the base must be converted to
        the matching **complex** dtype — passing the real context straight
        through would make the base's domain real and the view impossible to
        rebuild. The pairing comes from ``ops.complex_dtype`` because it is not
        portable to derive: NumPy promotes ``float32`` against a Python complex
        to ``complex128``, JAX and Torch to ``complex64``.
        """
        ops = new_ctx.ops
        base_ctx = Context(ops, dtype=ops.complex_dtype(new_ctx.dtype))
        return RealifiedFunctional(self.base.convert(base_ctx))

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """Flatten this functional for pytree registration."""
        return (self.base,), ()

    @classmethod
    def tree_unflatten(cls, aux: Any, children: Any) -> Self:
        """Rebuild this functional from pytree data."""
        (base,) = children
        return cls(base)


def realify(F: Functional) -> Functional:
    """
    Return ``F`` unchanged on a real domain, realified on a complex one.

    The idempotent entry point: calling it on an already-real functional is a
    no-op rather than an error, so a caller preparing an objective for a
    real-only optimizer need not branch on the field.

    Parameters
    ----------
    F : Functional
        Functional on a real or complex coordinate space.

    Returns
    -------
    Functional
        ``F`` itself when its domain is real, else a :class:`RealifiedFunctional`
        over stacked real coordinates.
    """
    if F.domain.field == "real":
        return F
    return RealifiedFunctional(F)


__all__ = ["RealifiedFunctional", "realify"]
