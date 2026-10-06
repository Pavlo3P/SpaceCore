"""
Spectral lift ``F(X) = f(lambda(X))`` of a symmetric coordinate functional (ADR-019).

For real-valued symmetric ``f`` differentiable at the spectrum of a Hermitian ``X``,
the Frobenius gradient is ``from_spectrum(grad f(lambda), frame)``.
Lewis, "Derivatives of Spectral Functions", 1996, Thm 1.1 (convex subgradients: Thm 1.3).
"""
from __future__ import annotations

from typing import Any, cast

from .._base import Domain, Functional
from ...contextual import Context
from ..._checks import CheckLevel
from ...space import JordanAlgebraSpace
from ..._checks import checked_method
from ._coordinate import _CoordinateFunctional
from ._norms import LpNormFunctional


def eigenvalue_space(dom: Any, check_level: CheckLevel | None = None) -> Any:
    r"""
    Return the real Euclidean space of shape ``(rank,)`` that the spectrum of ``dom`` lives in.

    Hermitian eigenvalues are real even with complex entries (Horn & Johnson,
    *Matrix Analysis*, 2nd ed., Thm 4.1.5). Other domains must supply a real
    spectrum for this helper; the rank is read off the zero element.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain whose spectrum is modelled.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; ``dom``'s by default.

    Returns
    -------
    DenseCoordinateSpace
        The eigenvalue space.
    """
    from ...space import DenseCoordinateSpace

    # The rank is the length of the spectrum of the zero element.
    if not isinstance(dom, JordanAlgebraSpace):
        raise TypeError(
            "eigenvalue_space requires a Jordan-algebra domain with a spectral "
            f"decomposition; got {type(dom).__name__}."
        )
    ops = dom.ops
    rank = int(dom.spectrum(dom.zeros()).shape[0])
    real_ctx = Context(ops, dtype=ops.real_dtype(dom.dtype))
    level = dom.check_level if check_level is None else check_level
    return DenseCoordinateSpace((rank,), real_ctx, check_level=level)


class SpectralFunctional(_CoordinateFunctional[Domain]):
    r"""
    Lift a symmetric coordinate functional onto a Jordan spectrum: :math:`F(X) = f(\lambda(X))`.

    ``LpNormFunctional`` lifts to the Schatten norm and ``NegativeEntropyFunctional`` to
    the negative von Neumann entropy on positive trace-one matrices.
    Horn & Johnson, *Matrix Analysis*, 2nd ed., Thm 7.4.7.2; for Hermitian matrices
    the singular values are the absolute eigenvalues, so the gauge is the p-norm.

    .. warning::

        The gradient formula requires permutation-invariant ``f`` and differentiability
        at ``lambda(X)`` (or a convex subgradient). Lewis, "Derivatives of Spectral
        Functions", Thms 1.1 and 1.3. Permutation invariance is not checked; a generic
        :class:`~spacecore.KLDivergenceFunctional` with unequal targets fails it.
        Elementwise spaces reduce to coordinate differentiation; custom Jordan spaces
        must supply a compatible spectral-gradient reconstruction.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain with a spectral decomposition, e.g. :class:`~spacecore.HermitianSpace`.
    base : Functional
        Symmetric functional on the ``(rank,)`` eigenvalue space; see :func:`eigenvalue_space`.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.HermitianSpace(2, ctx=ctx)
    >>> S = sc.eigenvalue_space(X)
    >>> vn = sc.SpectralFunctional(X, sc.NegativeEntropyFunctional(S))
    >>> rho = ctx.asarray([[0.5, 0.0], [0.0, 0.5]])   # maximally mixed state
    >>> float(vn.value(rho))                          # -log 2, i.e. sum p log p
    -0.6931471805599453
    """

    def _derivative_from_gradient(self, gradient):
        if self.codomain.scalar_field == "complex":
            raise NotImplementedError(
                "Complex-valued spectral differentials require an explicit derivative_at implementation."
            )
        return super()._derivative_from_gradient(gradient)

    def __init__(
        self,
        dom: Domain,
        base: Functional,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level, cod=base.codomain)
        # Check the domain has a spectrum and base lives on its eigenvalue space.
        domain = cast(Any, self.domain)
        if not isinstance(domain, JordanAlgebraSpace):
            raise TypeError(
                "SpectralFunctional requires a Jordan-algebra domain with a "
                f"spectral decomposition (e.g. HermitianSpace); got "
                f"{type(domain).__name__}."
            )
        if not isinstance(base, Functional):
            raise TypeError(f"base must be a Functional, got {type(base).__name__}.")
        expected = eigenvalue_space(domain)
        if base.domain.shape != expected.shape:
            raise ValueError(
                f"base domain shape {base.domain.shape} does not match the "
                f"spectrum of {type(domain).__name__} (rank {expected.shape[0]}); "
                f"build it on eigenvalue_space(dom)."
            )
        # Lewis's formula needs the plain coordinate gradient of f.
        if not getattr(base.domain, "is_euclidean", True):
            # Lewis needs the EUCLIDEAN gradient of f. On a Euclidean eigenvalue
            # space base.grad already is that; under any other metric it would be
            # G^-1 df, silently scaling the reconstructed spectral gradient.
            raise ValueError(
                "SpectralFunctional requires a Euclidean eigenvalue space so that "
                "base.grad is the plain coordinate gradient Lewis's formula needs."
            )
        self.base = base

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any) -> Any:
        """Return ``f(lambda(x))``."""
        return self.base.value(cast(Any, self.domain).spectrum(x))

    def _coordinate_grad(self, x: Any) -> Any:
        """Return the spectral gradient (Lewis, "Derivatives of Spectral Functions", Thm 1.1)."""
        domain = cast(Any, self.domain)
        eigvals, frame = domain.spectral_decompose(x)
        return domain.from_spectrum(self.base.grad(eigvals), frame)

    def __eq__(self, other: Any) -> bool:
        """Return whether another spectral functional has the same domain and base."""
        if not self.same_math(other):
            return NotImplemented
        return self.domain == other.domain and self.base == other.base

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        return (self.base,), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        domain, ctx = aux
        (base,) = children
        return cls(domain, base, ctx)

    def _convert(self, new_ctx: Context) -> "SpectralFunctional":
        """Convert this functional and its base to ``new_ctx``."""
        return SpectralFunctional(
            self.domain.convert(new_ctx), self.base.convert(new_ctx), new_ctx
        )


def spectralize(
    dom: Domain,
    make_base: Any,
    ctx: Context | str | None = None,
    check_level: CheckLevel | None = None,
) -> "SpectralFunctional[Domain]":
    r"""
    Build the spectral lift of ``make_base(eigenvalue_space(dom))`` on ``dom``.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain with a spectral decomposition.
    make_base : callable
        ``make_base(eigenvalue_space) -> Functional``, symmetric.
    ctx : Context, str, or None, optional
        Backend context.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level.

    Returns
    -------
    SpectralFunctional
        ``f(lambda(.))`` on ``dom``.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.HermitianSpace(2, ctx=ctx)
    >>> vn = sc.spectralize(X, sc.NegativeEntropyFunctional)   # negative von Neumann entropy
    >>> float(vn.value(ctx.asarray([[0.5, 0.0], [0.0, 0.5]])))
    -0.6931471805599453
    """
    base = make_base(eigenvalue_space(dom, check_level=check_level))
    return SpectralFunctional(dom, base, ctx, check_level=check_level)


def NuclearNormFunctional(
    dom: Domain,
    ctx: Context | str | None = None,
    check_level: CheckLevel | None = None,
) -> "SpectralFunctional[Domain]":
    r"""
    Nuclear norm ``sum_i |lambda_i(X)|``, i.e. the Schatten-1 lift of the 1-norm.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain with a spectral decomposition.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level.

    Returns
    -------
    SpectralFunctional
        The Schatten-1 lift.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.HermitianSpace(2, ctx=ctx)
    >>> f = sc.NuclearNormFunctional(X)
    >>> float(f.value(ctx.asarray([[2.0, 0.0], [0.0, -3.0]])))
    5.0
    """
    return spectralize(
        dom, lambda s: LpNormFunctional(s, 1.0), ctx, check_level=check_level
    )
