"""Leaf operators defined by actions: identity, matrix-free, and the adjoint view."""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, cast

from ._base import LinOp, Domain, Codomain, _require_linop
from ._metric import _requires_euclidean_or_riesz, metric_rapply, metric_rvapply
from .._checks import CheckLevel
from .._checks import checked_method
from ..space.base import CoordinateSpace, InnerProductSpace
from ..space._capabilities import _CAP_BATCH, _CAP_COORDINATE, probe_element
from ..contextual import Context



#: Sentinel for "the caller did not mention ``euclidean_adjoint`` at all".
#: Passing ``False`` explicitly is an assertion that ``rapply`` already carries the
#: geometry, so it silences the advisory; omitting it means the question was never
#: considered, which is the only case worth warning about.
_ADJOINT_UNSPECIFIED = object()


class IdentityLinOp(LinOp[Domain, Domain]):
    r"""
    Lazy identity map on ``space``.

    Parameters
    ----------
    space : Space
        Domain and codomain.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``space`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("space:domain", "ctx")

    def _apply_core(self, x: Any) -> Any:
        return x

    _rapply_core = _apply_core
    _vapply_core = _apply_core

    def __init__(
        self,
        space: Domain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(space, space, ctx, check_level=check_level)

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``xs`` after batched domain validation."""
        self._require(_CAP_BATCH, "IdentityLinOp.vapply")
        return xs

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, xs: Any) -> Any:
        """Return ``xs`` after batched codomain validation."""
        self._require(_CAP_BATCH, "IdentityLinOp.rvapply")
        return xs

    def to_dense(self) -> Any:
        """
        Return the dense tensor representation of this identity map.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        self._require(_CAP_COORDINATE, "IdentityLinOp.to_dense")
        # The identity matrix, reshaped to cod.shape + dom.shape.
        eye = self.ops.eye(math.prod(self.domain.shape), dtype=self.dtype)
        return self.ops.reshape(eye, tuple(self.codomain.shape) + tuple(self.domain.shape))

    def is_hermitian(self) -> bool:
        """
        Return whether this identity operator is Hermitian.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    def _repr_body(self) -> str:
        from .._repr import describe_space

        return describe_space(self.domain)

class MatrixFreeLinOp(LinOp[Domain, Codomain]):
    """
    Linear operator given by forward and adjoint callables; no matrix is stored.

    ``rapply`` must be the adjoint in the declared inner products,
    ``<apply(x), y>_Y = <x, rapply(y)>_X``. It is used as given unless
    ``euclidean_adjoint=True``, and no adjoint test runs at construction.

    Parameters
    ----------
    apply : callable
        Forward map ``apply(x)`` from ``dom`` to ``cod``.
    rapply : callable
        Adjoint map ``rapply(y)`` from ``cod`` to ``dom``.
    dom : Space
        Domain.
    cod : Space
        Codomain.
    ctx : Context, str, or None, optional
        Backend context; inferred from ``dom`` and ``cod`` when omitted.
    vapply : callable or None, optional
        Batched forward map; ``vmap`` of ``apply`` by default.
    rvapply : callable or None, optional
        Batched adjoint; ``vmap`` of ``rapply`` by default.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    euclidean_adjoint : bool, optional
        ``True``: ``rapply`` is the coordinate adjoint and is wrapped into
        ``R_X^-1 A^T R_Y`` (see :meth:`from_coordinate_adjoint`). ``False``: it is the
        metric adjoint. Omitting it on a non-Euclidean space warns, because a
        coordinate adjoint there is silently wrong.
    hermitian : bool or None, optional
        Declared self-adjointness, returned by :meth:`is_hermitian` so solvers need
        no probe; ``None`` (the default) means unknown. At ``strict`` checks a
        declared ``True`` is probed once against ``rapply``.
    """

    _fields = ("apply:apply_fn", "rapply:rapply_fn", "dom:domain", "cod:codomain", "ctx", "vapply:vapply_fn",
               "rvapply:rvapply_fn", "hermitian")

    def _apply_core(self, x: Any) -> Any:
        return self.apply_fn(x)

    # Capability requirements hold at every check level: an adjoint needs inner
    # products, a batch needs batchable spaces.
    def _rapply_core(self, y: Any) -> Any:
        self._require(InnerProductSpace, "MatrixFreeLinOp.rapply")
        return self.rapply_fn(y)

    def _vapply_core(self, xs: Any) -> Any:
        # The supplied batched map, else the generic vmap fallback.
        self._require(_CAP_BATCH, "MatrixFreeLinOp.vapply")
        return super()._vapply_core(xs) if self.vapply_fn is None else self.vapply_fn(xs)

    def _rvapply_core(self, ys: Any) -> Any:
        self._require(_CAP_BATCH, "MatrixFreeLinOp.rvapply")
        return super()._rvapply_core(ys) if self.rvapply_fn is None else self.rvapply_fn(ys)

    def __init__(
        self,
        apply: Callable[[Any], Any],
        rapply: Callable[[Any], Any],
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
        vapply: Callable[[Any], Any] | None = None,
        rvapply: Callable[[Any], Any] | None = None,
        check_level: CheckLevel | None = None,
        *,
        euclidean_adjoint: bool | Any = _ADJOINT_UNSPECIFIED,
        hermitian: bool | None = None,
    ) -> None:
        # Validate the callables, then bind the spaces.
        if not callable(apply):
            raise TypeError(f"apply must be callable, got {type(apply).__name__}.")
        if not callable(rapply):
            raise TypeError(f"rapply must be callable, got {type(rapply).__name__}.")
        if vapply is not None and not callable(vapply):
            raise TypeError(f"vapply must be callable, got {type(vapply).__name__}.")
        if rvapply is not None and not callable(rvapply):
            raise TypeError(f"rvapply must be callable, got {type(rvapply).__name__}.")
        super().__init__(dom, cod, ctx, check_level=check_level)

        # Does the adjoint need Riesz maps? Only on a non-Euclidean inner-product space.
        # Without coordinates there is no coordinate adjoint to confuse with, so a
        # native space never warns.
        has_inner = all(
            isinstance(space, InnerProductSpace) for space in (self.domain, self.codomain)
        )
        non_euclidean = has_inner and any(
            isinstance(space, CoordinateSpace) and not space.is_euclidean
            for space in (self.domain, self.codomain)
        )
        if euclidean_adjoint is True:
            # The caller supplied the *coordinate* adjoint A^dagger; wrap it into
            # the metric adjoint R_X^-1 A^dagger R_Y once, here.
            try:
                _requires_euclidean_or_riesz(
                    self.domain,
                    self.codomain,
                    "MatrixFreeLinOp(euclidean_adjoint=True) / "
                    "MatrixFreeLinOp.from_coordinate_adjoint",
                )
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
            rapply, rvapply = self._wrap_coordinate_adjoint(rapply, rvapply)
        elif non_euclidean and euclidean_adjoint is _ADJOINT_UNSPECIFIED:
            # The callable is stored verbatim and *trusted* as the metric adjoint.
            # On a non-Euclidean geometry a coordinate adjoint is simply wrong here
            # and nothing downstream detects it, so say so once, at construction.
            # Silent on Euclidean spaces (the two coincide) and silent when the
            # caller passed ``euclidean_adjoint=False`` explicitly, which asserts
            # that the supplied adjoint already carries the geometry.
            warnings.warn(
                "MatrixFreeLinOp stores `rapply` verbatim as the metric adjoint, but "
                f"{'domain' if not self.domain.is_euclidean else 'codomain'} geometry is "
                "non-Euclidean, where the metric adjoint R_X^-1 A^dagger R_Y differs from "
                "the coordinate adjoint. Pass euclidean_adjoint=True to have the coordinate "
                "adjoint wrapped for you, or confirm `rapply` already carries the geometry.",
                UserWarning,
                stacklevel=2,
            )

        # Store the callables; strict mode also dot-tests the adjoint.
        self.apply_fn = apply
        self.rapply_fn = rapply
        self.vapply_fn = vapply
        self.rvapply_fn = rvapply
        if self._checks_at_least("strict"):
            self._check_adjoint_consistency()
        self.hermitian = hermitian
        if hermitian and self._checks_at_least("strict"):
            self._check_declared_hermitian()

    def _wrap_coordinate_adjoint(
        self,
        coordinate_rapply: Callable[[Any], Any],
        coordinate_rvapply: Callable[[Any], Any] | None,
    ) -> tuple[Callable[[Any], Any], Callable[[Any], Any] | None]:
        """Return ``(rapply, rvapply)`` wrapping coordinate adjoints in Riesz maps."""
        # Wrap coordinate adjoints into metric ones: R_X^-1 A^T R_Y.
        dom, cod = self.domain, self.codomain

        def wrapped_rapply(y: Any) -> Any:
            return metric_rapply(dom, cod, coordinate_rapply, y)

        wrapped_rvapply: Callable[[Any], Any] | None = None
        if coordinate_rvapply is not None:

            def _wrapped_rvapply(ys: Any) -> Any:
                return metric_rvapply(dom, cod, coordinate_rvapply, ys)

            wrapped_rvapply = _wrapped_rvapply
        return wrapped_rapply, wrapped_rvapply

    def is_hermitian(self) -> bool | None:
        """Return the declared self-adjointness (``None`` when not declared)."""
        return self.hermitian

    def _check_declared_hermitian(self) -> None:
        """Probe a declared self-adjoint operator: ``apply`` and ``rapply`` must agree."""
        x = probe_element(self.domain, self.ops, self.dtype)
        if self.domain != self.codomain or (
            x is not None and not bool(self.ops.allclose(self.apply_fn(x), self.rapply_fn(x)))
        ):
            raise ValueError("MatrixFreeLinOp declared hermitian=True, but apply and rapply differ.")

    def _check_adjoint_consistency(self) -> None:
        """Probe the declared adjoint identity on deterministic space elements."""
        # Dot test <A x, y> = <x, A* y> on deterministic probe vectors.
        if not all(
            isinstance(space, InnerProductSpace)
            for space in (self.domain, self.codomain)
        ):
            return

        x = probe_element(self.domain, self.ops, self.dtype)
        y = probe_element(self.codomain, self.ops, self.dtype)
        if x is None or y is None:
            # Neither coordinates nor ``ones``: the space cannot build a non-zero
            # element, so there is nothing to probe with. Construction-time
            # geometry warnings still apply.
            return

        try:
            ax = self.apply_fn(x)
            ahy = self.rapply_fn(y)
            self.codomain._check_member(ax)
            self.domain._check_member(ahy)
            lhs = cast(Any, self.codomain).inner(ax, y)
            rhs = cast(Any, self.domain).inner(x, ahy)
            consistent = bool(self.ops.allclose(lhs, rhs))
        except Exception as exc:
            raise ValueError(
                "Strict matrix-free adjoint consistency check could not be completed."
            ) from exc
        if not consistent:
            raise ValueError(
                "Strict matrix-free adjoint consistency check failed: "
                "<A x, y> != <x, A* y> for the deterministic probe."
            )

    @classmethod
    def from_coordinate_adjoint(
        cls,
        apply: Callable[[Any], Any],
        coordinate_rapply: Callable[[Any], Any],
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
        vapply: Callable[[Any], Any] | None = None,
        coordinate_rvapply: Callable[[Any], Any] | None = None,
    ) -> MatrixFreeLinOp:
        r"""
        Build a matrix-free operator from a Euclidean coordinate adjoint.

        The stored adjoint is the metric one, ``R_X^-1 A^T R_Y``, which is the given
        callable itself on Euclidean spaces. A non-Euclidean space without usable Riesz
        maps is rejected.

        Parameters
        ----------
        apply : callable
            Forward map from ``dom`` to ``cod``.
        coordinate_rapply : callable
            Euclidean coordinate adjoint of ``apply``.
        dom, cod : Space
            Domain and codomain.
        ctx : Context, str, or None, optional
            Backend context.
        vapply : callable or None, optional
            Batched forward map.
        coordinate_rvapply : callable or None, optional
            Batched coordinate adjoint, wrapped the same way; ``vmap`` of the wrapped
            adjoint by default.
        """
        # Validate the callables, then build via __init__ with euclidean_adjoint=True.
        if not callable(apply):
            raise TypeError(f"apply must be callable, got {type(apply).__name__}.")
        if not callable(coordinate_rapply):
            raise TypeError(
                f"coordinate_rapply must be callable, got {type(coordinate_rapply).__name__}."
            )
        if coordinate_rvapply is not None:
            if not callable(coordinate_rvapply):
                raise TypeError(
                    f"coordinate_rvapply must be callable, got {type(coordinate_rvapply).__name__}."
                )

        # The wrapping now lives in __init__ behind euclidean_adjoint=True, so the
        # two entry points cannot drift apart, and this path raises no warning:
        # passing the coordinate adjoint is exactly what it is documented to take.
        return cls(
            apply,
            coordinate_rapply,
            dom,
            cod,
            ctx,
            vapply,
            coordinate_rvapply,
            euclidean_adjoint=True,
        )

    def fuse(self, *, materialize: bool = False) -> LinOp:
        """Stay matrix-free unless ``materialize=True`` is requested (ADR-021/ADR-008).

        By default a matrix-free operator is never densified, so it returns
        itself. With ``materialize=True`` the caller explicitly opts into
        densification: the operator is probed into a dense tensor (via
        ``to_dense``, the basis sweep) and returned as a :class:`DenseLinOp`,
        letting an enclosing expression collapse to a single dense operator.
        """
        # Matrix-free stays lazy unless densifying was requested.
        if not materialize:
            return self
        from ._dense import DenseLinOp

        return DenseLinOp(self.to_dense(), self.domain, self.codomain, self.ctx)

    def _convert(self, new_ctx: Context) -> MatrixFreeLinOp:
        """
        Convert this matrix-free operator to ``new_ctx``.

        Parameters
        ----------
        new_ctx:
            Concrete target context for converted domain and codomain spaces.

        Returns
        -------
        MatrixFreeLinOp
            Operator with converted spaces and the same user-supplied
            callables.
        """
        # ``self.rapply_fn`` is already the metric adjoint by construction -- either
        # supplied that way, or wrapped once at __init__ under
        # ``euclidean_adjoint=True``. Declaring ``False`` here is therefore both
        # correct (re-wrapping would apply the Riesz maps twice) and quiet: the
        # construction advisory belongs to the original call, not to every convert.
        # The stored rapply is already the metric adjoint, so no re-wrap.
        return MatrixFreeLinOp(
            self.apply_fn,
            self.rapply_fn,
            self.domain.convert(new_ctx),
            self.codomain.convert(new_ctx),
            new_ctx,
            self.vapply_fn,
            self.rvapply_fn,
            euclidean_adjoint=False,
            hermitian=self.hermitian,
        )


class _AdjointViewLinOp(LinOp[Codomain, Domain]):
    """
    Adjoint view ``op.H``: ``apply`` is ``op.rapply`` and ``rapply`` is ``op.apply``; ``op.H.H is op``.

    Parameters
    ----------
    op : LinOp
        Operator whose adjoint is viewed.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _fields = ("op",)
    _dynamic = ("op",)

    def _apply_core(self, y: Any) -> Any:
        return self.op._rapply_core(y)

    def _rapply_core(self, x: Any) -> Any:
        return self.op._apply_core(x)

    def _vapply_core(self, ys: Any) -> Any:
        return self.op.rvapply(ys)

    def __init__(
        self,
        op: LinOp[Domain, Codomain],
        check_level: CheckLevel | None = None,
    ) -> None:
        # The view swaps domain and codomain.
        op = _require_linop(op, "op")
        if check_level is None:
            check_level = op.check_level
        super().__init__(op.codomain, op.domain, op.ctx, check_level=check_level)
        self.op = op

    def vapply(self, ys: Any) -> Any:
        """Return ``op.rvapply(ys)`` over a batch."""
        self._require(_CAP_BATCH, "_AdjointViewLinOp.vapply")
        return self.op.rvapply(ys)

    def rvapply(self, xs: Any) -> Any:
        """Return ``op.vapply(xs)`` over a batch."""
        self._require(_CAP_BATCH, "_AdjointViewLinOp.rvapply")
        return self.op.vapply(xs)

    def fuse(self, *, materialize: bool = False) -> LinOp:
        """Fuse the wrapped operand; return the adjoint of the fused operator (ADR-021).

        ``A.H.fuse()`` is ``A.fuse().H`` — the inner expression is fused (e.g. a
        composition collapses to one dense operator) and the adjoint view wraps
        the result, which keeps the metric adjoint correct on any geometry. A
        matrix-free operand stays matrix-free under its adjoint (or is densified
        when ``materialize=True``).
        """
        return self.op.fuse(materialize=materialize).H

    @property
    def H(self) -> LinOp[Domain, Codomain]:
        """Original operator viewed as the adjoint of this adjoint view."""
        return self.op

    def _repr_class_name(self) -> str:
        """Present a clean public label for the private adjoint-view class."""
        return "AdjointLinOp"

    def _repr_body(self) -> str:
        return f"{self.op._short_repr()}.H"
