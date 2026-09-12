r"""Real-coordinate view of a complex-domain functional.

Optimizers and line searches that assume a real vector space cannot consume a
functional whose domain is complex — ``spacecore.optimize`` flattens to real
coordinates, and a complex iterate has no ordering, no real inner product to
descend in, and no meaningful ``result_type`` against a real step size. The usual
workaround is to hand-write a real wrapper per objective, re-deriving the
Cauchy-Riemann bookkeeping each time and getting the conjugate wrong.

:class:`RealifiedFunctional` does it once. A real-valued ``F`` on a complex
coordinate space has differential ``dF = Re<g, dv>`` with metric gradient ``g``,
so writing ``v = a + i b`` gives

.. math::

    \partial F/\partial a = \operatorname{Re} g_v, \qquad
    \partial F/\partial b = \operatorname{Im} g_v,

where ``g_v = flatten(X.riesz(g))`` is the *coordinate* gradient. The real
gradient is therefore exactly the stacked pair — no Wirtinger calculus at the
call site, and the metric correction stays where ADR-010 put it.

References
----------
.. [Realification] Treating ``C^m`` as ``R^{2m}`` by ``v = a + i b`` is the
   standard realification of a complex vector space, and ``Re<., .>_C`` is a
   real inner product on ``R^{2m}`` inducing the same norm, so the two views are
   isometric.

   This is **derived**, from the axioms of Conway, *A Course in Functional
   Analysis*, 2nd ed., Springer, 1990, **Definition I.1.1** — stated uniformly
   over ``F = R`` or ``C``, with the note that ``conj(alpha) = alpha`` when
   ``F = R``. Each property of a real inner product needs a different axiom:

   * *real-bilinearity* — (a) and (b) restricted to **real** scalars, where the
     conjugation in (b) is the identity; ``Re`` of a real-linear map is
     real-linear;
   * *symmetry* — (d), since ``Re conj(z) = Re z``, so
     ``Re<x, y> = Re<y, x>``;
   * *positive-definiteness* — (c) for ``<v, v> >= 0`` **and** the extra
     inner-product axiom (f), ``<v, v> = 0 => v = 0``. Without these two,
     ``Re<., .>`` would only be a symmetric real bilinear form;
   * *same norm* — (c) and (d) make ``<v, v>`` real and nonnegative, so
     ``Re<v, v> = <v, v> = ||v||^2``.

   Convention caveat: Conway's I.1.1 is linear in the **first** argument and
   conjugate-linear in the second; SpaceCore's ``Space.inner`` conjugates the
   **first**. The realification is unaffected — ``Re<., .>`` is symmetric either
   way — but do not transcribe the axioms slot-for-slot into this codebase.

.. [Wirtinger] For a **real-valued** ``F`` the Wirtinger derivatives satisfy
   ``dF/d(conj v) = conj(dF/dv)``, so the real gradient is
   ``2 dF/d(conj v)`` — the same stacked pair derived above. See Kreutz-Delgado,
   "The complex gradient operator and the CR-calculus", arXiv:0906.4835, 2009,
   §4 (the real-valued case and the steepest-ascent direction). The conjugate is
   where hand-written wrappers usually go wrong, which is the reason this view
   exists once rather than per objective.
"""
from __future__ import annotations

from typing import Any, Self, Tuple

from .._checks import checked_method
from ..contextual import Context
from ..space import CoordinateSpace, DenseCoordinateSpace
from ._base import Functional


class RealifiedFunctional(Functional):
    r"""
    View of a complex-domain functional over stacked real coordinates.

    For a base functional :math:`F` on a coordinate space :math:`X` with complex
    flattened coordinates :math:`v \in \mathbb{C}^m`, this functional acts on
    :math:`w = (\operatorname{Re} v, \operatorname{Im} v) \in \mathbb{R}^{2m}`.

    Because a real-valued :math:`F` has differential
    :math:`dF = \operatorname{Re}\langle g, dv \rangle` with metric gradient
    :math:`g`, the real coordinate gradient is exactly
    :math:`(\operatorname{Re} g_v, \operatorname{Im} g_v)` where
    :math:`g_v = \operatorname{flatten}(X.\operatorname{riesz}(g))`.

    Parameters
    ----------
    base : Functional
        Functional on a complex :class:`~spacecore.space.CoordinateSpace`.

    Raises
    ------
    TypeError
        If the base domain is not a coordinate space (there are no flattened
        coordinates to split).
    ValueError
        If the base domain is already real — use the functional directly rather
        than paying for a no-op view.

    Notes
    -----
    The domain is classified by ``Space.field``, which is derived from the dtype.
    That is exact for a genuinely complex coordinate space, but **over-reports**
    for a space whose complex storage is constrained — most importantly
    :class:`~spacecore.HermitianSpace`, where Hermitian matrices form a *real*
    vector space of dimension :math:`n^2` yet ``field`` reads ``"complex"``.

    Realifying such a space is *safe but redundant*: ``unflatten`` symmetrizes,
    so ``from_real`` always lands back on the manifold and no step can escape it,
    but the view carries :math:`2n^2` real coordinates for :math:`n^2` real
    dimensions. ``flatten . unflatten`` is then a projection rather than the
    identity, so the realified problem has a rank-deficient Hessian with
    :math:`n^2` null directions — harmless for first-order methods, degenerate
    for Newton-type ones. Prefer optimizing such a space directly.

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
