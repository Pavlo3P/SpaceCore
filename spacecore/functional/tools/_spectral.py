"""Spectral lift of coordinate functionals over a Jordan-algebra spectrum (ADR-019).

A *spectral function* is ``F(X) = f(lambda(X))`` for a symmetric ``f`` of the
eigenvalues. By Lewis's theorem its gradient is ``U diag(grad f(lambda)) U^*``,
which is exactly ``from_spectrum(grad f(lambda), frame)`` on the
[ADR-012](012_jordan_spectrum.md) Jordan spectral API. Building on
``spectrum`` / ``spectral_decompose`` / ``from_spectrum`` rather than reaching
for a backend ``eigh`` keeps it correct on every Jordan space; on an elementwise
Jordan space the spectrum *is* the coordinates, so a lifted functional coincides
with its coordinate original.

:class:`SpectralFunctional` implements that lift once, for **any** coordinate
functional. There is deliberately no per-formula spectral class: a Schatten
``p``-norm is ``spectralize(X, lambda s: LpNormFunctional(s, p))``, the von
Neumann entropy is ``spectralize(X, NegativeEntropyFunctional)``, and so on. The
alternative — one hand-written class per formula — duplicates the value, the
gradient and the validation of its coordinate twin, which is how the retired
``SpectralLpNormFunctional`` came to re-check ``p >= 1`` that
``LpNormFunctional`` already enforced.

References
----------
.. [Lewis] A. S. Lewis, Theorem 2.3.2 in H. Wolkowicz, R. Saigal, L. Vandenberghe
   (eds.), *Handbook of Semidefinite Programming*, Kluwer, 2000, §2.3.2
   "Smoothness of eigenvalues", pp. 18-19: for a *permutation-invariant* ``f``,
   ``F(X) = f(lambda(X))`` is (Frechet) differentiable at ``X`` **if and only if**
   ``f`` is differentiable at ``lambda(X)``, and then
   ``DF(X) = U^T Diag(f'(lambda(X))) U`` for any orthogonal ``U`` diagonalizing
   ``X``. Permutation-invariance is the hypothesis, not a convenience — it is why
   :class:`SpectralFunctional` states symmetry as a caller contract.
.. [Beck7] A. Beck, *First-Order Methods in Optimization*, MOS-SIAM, 2017,
   Ch. 7: Definition 7.11 (spectral functions over ``S^n``), Definition 7.12
   (symmetric spectral functions), Theorem 7.9 (symmetric conjugate theorem) and
   §7.2.2 (the proximal operator of a symmetric spectral function) — the modern
   optimization treatment, including the conjugate and prox of the lift.
"""
from __future__ import annotations

from typing import Any, cast

from .._base import Domain, Functional
from ...contextual import Context
from ..._check_policy import CheckLevel
from ...space import JordanAlgebraSpace
from ..._checks import checked_method
from ._coordinate import _CoordinateFunctional
from ._norms import LpNormFunctional


def eigenvalue_space(dom: Any, check_level: CheckLevel | bool | None = None) -> Any:
    r"""Return the real coordinate space the spectrum of ``dom`` lives in.

    The spectrum of a Jordan-algebra element is a real vector of length equal to
    the algebra's rank, regardless of whether the element itself is stored with
    complex entries (a Hermitian matrix has real eigenvalues). Rank is read off
    the zero element rather than declared, so this works for any Jordan space
    without a ``rank`` attribute.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain whose spectrum is to be modelled.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation policy for the returned space. Defaults to ``dom``'s.

    Returns
    -------
    DenseCoordinateSpace
        Euclidean real space of shape ``(rank,)``.
    """
    from ...space import DenseCoordinateSpace

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
    r"""Lift a **symmetric** coordinate functional onto a Jordan spectrum.

    Given ``f`` on :math:`\mathbb{R}^r` and a Jordan space of rank :math:`r`,
    this is the spectral function :math:`F(X) = f(\lambda(X))`. By Lewis's
    theorem its gradient is :math:`U \operatorname{diag}(\nabla f(\lambda)) U^*`,
    which is exactly ``from_spectrum(grad f(lambda), frame)`` on the ADR-012
    spectral API.

    Any coordinate functional gains a spectral counterpart by wrapping rather
    than by being re-implemented: ``LpNormFunctional`` lifts to the Schatten
    ``p``-norm (nuclear at ``p = 1``, Frobenius at ``p = 2``),
    ``NegativeEntropyFunctional`` to the **von Neumann entropy**,
    ``SquaredL2NormFunctional`` to the squared Frobenius norm, and
    ``HuberFunctional`` to its spectral analogue.

    .. warning::

        ``f`` **must be symmetric** (permutation-invariant). This is not a
        technicality about gradients: eigenvalues have no canonical order, so for
        a non-symmetric ``f`` the composition :math:`f(\lambda(X))` is not even a
        well-defined function of :math:`X` — its value would depend on the
        backend's sort convention. Symmetry cannot be checked programmatically,
        so it is a contract the caller must honour.
        :class:`~spacecore.KLDivergenceFunctional` is the notable member of the
        toolbox that does **not** qualify, being weighted by a fixed target.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain with a spectral decomposition (e.g. :class:`~spacecore.HermitianSpace`).
    base : Functional
        Symmetric functional on the Euclidean real space of shape ``(rank,)``;
        see :func:`eigenvalue_space`.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation policy for this functional.

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

    def __init__(
        self,
        dom: Domain,
        base: Functional,
        ctx: Context | str | None = None,
        check_level: CheckLevel | bool | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level)
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
        if not getattr(base.domain, "is_euclidean", True):
            # Lewis needs the EUCLIDEAN gradient of f. On a Euclidean eigenvalue
            # space base.grad already is that; under any other metric it would be
            # G^-1 df, silently scaling the reconstructed spectral gradient.
            raise ValueError(
                "SpectralFunctional requires a Euclidean eigenvalue space so that "
                "base.grad is the plain coordinate gradient Lewis's formula needs."
            )
        self.base = base

    @checked_method(in_space="domain", out_scalar=True)
    def value(self, x: Any) -> Any:
        """Return ``f(lambda(x))``."""
        return self.base.value(cast(Any, self.domain).spectrum(x))

    def _coordinate_grad(self, x: Any) -> Any:
        """Return ``U diag(grad f(lambda)) U^*`` via ``from_spectrum`` (Lewis)."""
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
    check_level: CheckLevel | bool | None = None,
) -> "SpectralFunctional[Domain]":
    r"""Build the spectral counterpart of a coordinate functional.

    Constructs the eigenvalue space for ``dom``, hands it to ``make_base``, and
    wraps the result — so a caller never has to derive the rank or build the
    intermediate space by hand.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain with a spectral decomposition.
    make_base : callable
        ``make_base(eigenvalue_space) -> Functional``; the functional must be
        symmetric (see :class:`SpectralFunctional`).
    ctx : Context, str, or None, optional
        Backend context specification.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation policy.

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
    >>> vn = sc.spectralize(X, sc.NegativeEntropyFunctional)   # von Neumann entropy
    >>> float(vn.value(ctx.asarray([[0.5, 0.0], [0.0, 0.5]])))
    -0.6931471805599453
    """
    base = make_base(eigenvalue_space(dom, check_level=check_level))
    return SpectralFunctional(dom, base, ctx, check_level=check_level)


def NuclearNormFunctional(
    dom: Domain,
    ctx: Context | str | None = None,
    check_level: CheckLevel | bool | None = None,
) -> "SpectralFunctional[Domain]":
    r"""
    Nuclear (trace) norm ``sum_i |lambda_i(X)|`` — the Schatten-1 norm.

    Kept as a named constructor because the nuclear norm is a concept in its own
    right (the convex envelope of rank, and the workhorse of low-rank recovery),
    not because the formula needs its own class: it is exactly
    ``spectralize(dom, lambda s: LpNormFunctional(s, 1.0))``, and ``p >= 1`` is
    validated once, by :class:`~spacecore.LpNormFunctional`.

    Parameters
    ----------
    dom : JordanAlgebraSpace
        Domain space with a spectral decomposition.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation policy for the constructed functional.

    Returns
    -------
    SpectralFunctional
        The Schatten-1 lift of the coordinate 1-norm.

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
