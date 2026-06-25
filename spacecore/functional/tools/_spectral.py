"""Spectral (Schatten) ``p``-norm functional over a Jordan-algebra spectrum (ADR-019).

Where :class:`~spacecore.functional.LpNormFunctional` is the ``p``-norm of the
*coordinates*, :class:`SpectralLpNormFunctional` is the ``p``-norm of the
*spectrum*: for a Hermitian element ``X = U diag(lambda) U^*`` it is
``(sum_i |lambda_i|^p)^{1/p}`` -- the Schatten-``p`` norm (nuclear norm at
``p = 1``, Frobenius at ``p = 2``).

It is a *spectral function* ``F(X) = f(lambda(X))`` with ``f`` the symmetric
coordinate ``p``-norm. Its gradient is the spectral function gradient
``U diag(grad f(lambda)) U^*`` (Lewis), which is exactly
``from_spectrum(grad f(lambda), frame)`` on the [ADR-012](012_jordan_spectrum.md)
Jordan spectral API. Building it on ``spectrum`` / ``spectral_decompose`` /
``from_spectrum`` (rather than reaching for backend ``eigh``) keeps it correct on
every Jordan space: on an elementwise Jordan space the spectrum *is* the
coordinates, so it coincides with ``LpNormFunctional``.
"""
from __future__ import annotations

import math
from typing import Any, cast

from .._base import Domain
from ...backend import Context, jax_pytree_class
from ...space import JordanAlgebraSpace
from ..._checks import checked_method
from ._coordinate import _CoordinateFunctional, lp_coordinate_grad, lp_value


@jax_pytree_class
class SpectralLpNormFunctional(_CoordinateFunctional[Domain]):
    r"""
    Schatten ``p``-norm ``F(X) = (sum_i |lambda_i(X)|^p)^{1/p}`` for ``p >= 1``.

    ``lambda(X)`` is the Jordan-algebraic spectrum of ``X`` (eigenvalues for a
    Hermitian matrix). The gradient is the spectral function gradient: with
    ``X = U diag(lambda) U^*``, it is ``U diag(g) U^*`` where ``g`` is the
    coordinate ``p``-norm gradient of ``lambda``, reconstructed through the
    space's ``from_spectrum``. ``p = 1`` is the nuclear / trace norm (see
    :func:`NuclearNormFunctional`); ``p = 2`` is the Frobenius norm.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain space with a spectral decomposition (e.g.
        :class:`~spacecore.HermitianSpace`). A space without a Jordan spectrum
        raises ``TypeError``.
    p : float
        Norm order; must be finite and ``>= 1``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.HermitianSpace(2, ctx=ctx)
    >>> A = ctx.asarray([[2.0, 0.0], [0.0, -3.0]])
    >>> f = sc.SpectralLpNormFunctional(X, 1)  # nuclear norm |2| + |-3|
    >>> float(f.value(A))
    5.0
    """

    def __init__(self, dom: Domain, p: Any, ctx: Context | str | None = None) -> None:
        super().__init__(dom, ctx)
        if not isinstance(self.domain, JordanAlgebraSpace):
            raise TypeError(
                "SpectralLpNormFunctional requires a Jordan-algebra domain with a "
                f"spectral decomposition (e.g. HermitianSpace); got "
                f"{type(self.domain).__name__}."
            )
        p = float(p)
        if not math.isfinite(p) or p < 1.0:
            raise ValueError(f"SpectralLpNormFunctional requires a finite p >= 1, got {p}.")
        self.p = p

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return the Schatten-``p`` norm ``(sum_i |lambda_i|^p)^{1/p}``."""
        spectrum = cast(Any, self.domain).spectrum(x)
        return lp_value(self.ops, spectrum, self.p)

    def _coordinate_grad(self, x: Any) -> Any:
        """Spectral gradient ``U diag(grad f(lambda)) U^*`` via ``from_spectrum``."""
        domain = cast(Any, self.domain)
        eigvals, frame = domain.spectral_decompose(x)
        spectral_grad = lp_coordinate_grad(self.ops, eigvals, self.p)
        return domain.from_spectrum(spectral_grad, frame)

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (), (self.domain, self.p, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        domain, p, ctx = aux
        return cls(domain, p, ctx)

    def _convert(self, new_ctx: Context) -> "SpectralLpNormFunctional":
        """Convert this functional to ``new_ctx``."""
        return SpectralLpNormFunctional(self.domain.convert(new_ctx), self.p, new_ctx)


def NuclearNormFunctional(
    dom: Domain, ctx: Context | str | None = None
) -> "SpectralLpNormFunctional[Domain]":
    r"""
    Nuclear (trace) norm, a thin wrapper for ``SpectralLpNormFunctional(X, 1)``.

    Computes ``sum_i |lambda_i(X)|``, the Schatten-1 norm of the Jordan spectrum.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain space with a spectral decomposition.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Returns
    -------
    SpectralLpNormFunctional
        The ``p = 1`` instance of :class:`SpectralLpNormFunctional`.
    """
    return SpectralLpNormFunctional(dom, 1.0, ctx)
