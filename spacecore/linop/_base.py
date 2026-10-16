from __future__ import annotations

import warnings
from functools import cached_property
from math import prod
from typing import Any, Generic, TypeVar

from .._batching import _leading_batch_size, _warn_vmap_fallback_once
from .._checks import CheckLevel
from .._checks import checked_method
from ..space._capabilities import _CAP_BATCH, _CAP_COORDINATE, require
from ..space import VectorSpace, RealField
from ..contextual import Context
from ..mapping import Mapping

Domain = TypeVar("Domain", bound=VectorSpace)
Codomain = TypeVar("Codomain", bound=VectorSpace)


class LinOp(Mapping, Generic[Domain, Codomain]):
    r"""
    A linear map from ``dom`` to ``cod``; subclasses supply ``apply`` and ``rapply``.

    ``rapply`` is the Hilbert-space adjoint in each space's own inner product,
    :math:`\langle A x, y\rangle_Y = \langle x, A^* y\rangle_X`, not the coordinate
    transpose. In orthonormal bases it is the conjugate transpose (Trefethen & Bau,
    *Numerical Linear Algebra*, Lecture 2, pp. 11-12); in general coordinates it is
    :math:`R_X^{-1} A^\dagger R_Y`, derived in
    :func:`~spacecore.linop._metric.metric_rapply`.
    Adjoint products reverse order (Trefethen & Bau, *Numerical Linear Algebra*,
    Eq. (2.4)); inserting Gram factors cancels the intermediate metric.

    Parameters
    ----------
    dom : Space
        Domain ``X``.
    cod : Space
        Codomain ``Y``.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` and ``cod`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

    Examples
    --------
    Use a concrete dense operator as a :class:`LinOp`.

    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 2.0]]), X, X, ctx)
    >>> A.apply(ctx.asarray([3.0, 4.0]))
    array([3., 8.])
    """

    def __init__(
        self,
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ):
        # Scalar-valued maps are Functionals, not LinOps (ADR-010).
        if isinstance(cod, RealField):
            raise TypeError(
                "LinOp codomain cannot be a RealField (ADR-010); use Functional "
                "for scalar-valued maps and Riesz gradients."
            )
        self.dom, self.cod = self._bind_context(ctx, dom, cod, check_level=check_level)

    def _init_map(self, domain: Any, codomain: Any, ctx: Any, check_level: Any) -> None:
        """Bind a node of this family (the hook the shared lazy nodes in ``mapping`` call)."""
        LinOp.__init__(self, domain, codomain, ctx, check_level=check_level)

    @property
    def domain(self) -> Domain:
        """Domain space of this linear operator."""
        return self.dom

    @property
    def codomain(self) -> Codomain:
        """Codomain space of this linear operator."""
        return self.cod

    @cached_property
    def A(self) -> Any:
        """
        Native numerical representation of this operator.

        Concrete subclasses may choose the representation that best matches
        their storage model: for example, dense operators return a dense array
        while sparse operators return their sparse matrix. Matrix-free or lazy
        operators generally do not have such a representation and should leave
        this property unimplemented. Use :meth:`to_dense` when a dense tensor
        materialization is explicitly required.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a native numerical representation."
        )

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Apply the forward map to an element of ``self.domain``."""
        return self._apply_core(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        r"""Apply the adjoint map to an element of ``self.codomain``.

        This is the **Hilbert-space adjoint** ``A^#``, defined by
        ``<Ax, y>_Y = <x, A^# y>_X``. Trefethen & Bau, *Numerical Linear Algebra*,
        Lecture 2, pp. 11-12, gives the conjugate transpose in orthonormal coordinates;
        :func:`spacecore.linop._metric.metric_rapply` derives the Gram-matrix form.
        The dual map sends a linear functional ``phi`` to ``phi o A``; writing
        ``phi(y) = b^T y`` gives dual coordinates ``A^T b``, even over the complex field.
        """
        return self._rapply_core(y)

    def _require(self, capability: Any, name: str) -> None:
        """Raise unless both endpoint spaces provide ``capability``."""
        require(self.domain, capability, name)
        require(self.codomain, capability, name)

    # Subclasses implement the check-free cores, which the checked public methods
    # call (an undecorated public override becomes the core; see install_public_checks).
    def _apply_core(self, x: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} defines neither apply nor _apply_core.")

    def _rapply_core(self, y: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} defines neither rapply nor _rapply_core.")

    def _vapply_core(self, xs: Any) -> Any:
        # Generic batching: vmap the single-element apply over the leading axis.
        self._require(_CAP_BATCH, "LinOp.vapply")
        _warn_vmap_fallback_once(self, "vapply", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.apply, in_axes=0, out_axes=0)(xs)

    def _rvapply_core(self, ys: Any) -> Any:
        self._require(_CAP_BATCH, "LinOp.rvapply")
        _warn_vmap_fallback_once(self, "rvapply", _leading_batch_size(self.codomain, ys))
        return self.ops.vmap(self.rapply, in_axes=0, out_axes=0)(ys)

    def __call__(self, x: Any) -> Any:
        """Apply this linear operator to ``x``."""
        return self.apply(x)

    def adjoint_apply(self, y: Any) -> Any:
        """Apply the adjoint of this linear operator to ``y``.

        Spelled-out alias for the inner-product adjoint :meth:`rapply`.
        """
        return self.rapply(y)

    def is_hermitian(self) -> bool | None:
        """
        Return whether this operator is structurally Hermitian when known.

        Returns
        -------
        bool | None
            ``True`` or ``False`` when the subclass can verify the structure
            cheaply, otherwise ``None`` for unknown or matrix-free operators.
        """
        return None

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Apply over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        return self._vapply_core(xs)

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
        return self._rvapply_core(ys)

    @property
    def H(self) -> LinOp:
        r"""Hermitian-adjoint view of this linear operator.

        Returns
        -------
        LinOp
            Adjoint view satisfying
            :math:`\langle A x, y\rangle_Y = \langle x, A^* y\rangle_X`.
        """
        from ._matrix_free import _AdjointViewLinOp

        # Build the adjoint view once and cache it, so A.H is A.H.
        view = getattr(self, "_adjoint_view", None)
        if view is None:
            view = _AdjointViewLinOp(self)
            self._adjoint_view = view
        return view

    def adjoint(self) -> LinOp:
        """Return the Hermitian-adjoint view of this linear operator."""
        return self.H

    def fuse(self, *, materialize: bool = False) -> LinOp:
        r"""
        Return an equivalent operator with dense sub-expressions multiplied out (ADR-021).

        Each maximal run of dense operators becomes one :class:`DenseLinOp`; other leaves
        stay lazy. The result equals ``self`` up to rounding. A leaf returns itself.

        Parameters
        ----------
        materialize : bool, optional
            Also densify matrix-free operands, which may be costly; by default they stay lazy.

        Returns
        -------
        LinOp
            The fused operator.
        """
        return self

    def to_dense(self) -> Any:
        """
        Materialize this operator as a dense backend array.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        The default implementation is intended for small problems, debugging,
        and tests. It materializes the full coordinate matrix, so subclasses
        that already store a dense or sparse matrix should override this method
        for efficiency.
        """
        self._require(_CAP_COORDINATE, "LinOp.to_dense")
        return self.ops.reshape(
            self.to_matrix(), tuple(self.codomain.shape) + tuple(self.domain.shape)
        )

    def to_sparse(self):
        raise NotImplementedError(f"{type(self).__name__} does not define sparse materialization.")

    def to_matrix(self) -> Any:
        """
        Materialize this operator as a 2D dense coordinate matrix.

        The returned array has shape
        ``(prod(self.codomain.shape), prod(self.domain.shape))``. The default
        implementation builds a batch of standard basis vectors and calls
        :meth:`vapply` once. If a space cannot batch-flatten or batch-unflatten
        its representation, it falls back to a safe Python loop. This method is
        for small/testing use; concrete storage-backed subclasses should
        override it when they can expose a matrix directly.
        """
        self._require(_CAP_COORDINATE, "LinOp.to_matrix")
        # Apply the operator to every basis vector: its images are the matrix columns.
        domain_size = prod(self.domain.shape)
        codomain_size = prod(self.codomain.shape)
        eye = self.ops.eye(domain_size, dtype=self.dtype)

        # Fast path: one batched apply over the whole identity.
        try:
            xs = self.domain._unflatten_batch_core(eye)
            ys = self.vapply(xs)
            ys_flat = self.codomain._flatten_batch_core(ys)
            matrix = self.ops.transpose(ys_flat, (1, 0))
            return self.ops.reshape(matrix, (codomain_size, domain_size))
        except (AttributeError, NotImplementedError, TypeError) as exc:
            warnings.warn(
                (
                    f"{type(self).__name__}.to_matrix() could not use the batched "
                    f"materialization path and is falling back to a Python loop. "
                    f"This is slower and not JIT-friendly. Original error: "
                    f"{type(exc).__name__}: {exc}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        # Fallback: one apply per basis vector.
        columns = []
        for i in range(domain_size):
            basis_vector = eye[:, i]
            x = self.domain._unflatten_core(basis_vector)
            y = self.apply(x)
            columns.append(self.codomain._flatten_core(y))
        return self.ops.stack(tuple(columns), axis=1)

    def assert_domain(self, x: Any) -> None:
        """Raise if ``x`` is not in the domain."""
        self.dom.check_member(x)

    def assert_codomain(self, y: Any) -> None:
        """Raise if ``y`` is not in the codomain."""
        self.cod.check_member(y)

    # tree_flatten / tree_unflatten are inherited from PyTreeNode, which owns the
    # flatten contract for every SpaceCore container and auto-registers concrete
    # subclasses with each backend's tree protocol. Every concrete LinOp is a
    # container, so the capability belongs on this base rather than being
    # re-declared (and separately registered) per operator.


LinOp._family = LinOp


def _require_linop(op: Any, name: str) -> LinOp:
    """Return ``op`` as a linear operator or raise a typed error."""
    if not isinstance(op, LinOp):
        raise TypeError(f"{name} must be a LinOp, got {type(op).__name__}.")
    return op


@checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
def _output_checked_rvapply(self: LinOp, ys: Any) -> Any:
    """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
    return self._rvapply_core(ys)


# Assigned as ``rvapply`` by operators that also check the batched output.
_output_checked_rvapply.__name__ = _output_checked_rvapply.__qualname__ = "rvapply"
