from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import cached_property
from math import prod
from typing import Any, Generic, TypeVar

from .._batching import _leading_batch_size, _warn_vmap_fallback_once
from .._check_policy import CheckLevel
from .._checks import checked_method
from ..space._capabilities import _CAP_BATCH, _CAP_COORDINATE, require
from ..space import VectorSpace, Field
from ..contextual import Context
from ..mapping import Mapping

Domain = TypeVar("Domain", bound=VectorSpace)
Codomain = TypeVar("Codomain", bound=VectorSpace)


class LinOp(Mapping, Generic[Domain, Codomain]):
    r"""
    A linear map from ``dom`` to ``cod``; subclasses supply ``apply`` and ``rapply``.

    ``rapply`` is the Hilbert-space adjoint in each space's own inner product,
    :math:`\langle A x, y\rangle_Y = \langle x, A^* y\rangle_X`, not the coordinate
    transpose. On a non-Euclidean space it is :math:`R_X^{-1} A^\dagger R_Y`
    (:func:`~spacecore.linop._metric.metric_rapply`). Conway, *A Course in Functional
    Analysis*, II.2.4.

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
        if isinstance(cod, Field):
            raise TypeError(
                "LinOp codomain cannot be a Field (ADR-010); use Functional "
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

    @abstractmethod
    def apply(self, x: Any) -> Any:
        """Apply the forward map to an element of ``self.domain``."""

    @abstractmethod
    def rapply(self, y: Any) -> Any:
        r"""Apply the adjoint map to an element of ``self.codomain``.

        This is the **Hilbert-space adjoint** ``A^#``, defined by
        ``<Ax, y>_Y = <x, A^# y>_X`` (Dunford-Schwartz, *Linear Operators I*,
        Definition VI.2.9) -- *not* the Banach adjoint of Definition VI.2.1,
        which maps ``Y^* -> X^*`` and is the coordinate transpose. The two
        coincide only when both spaces are Euclidean. See
        :func:`spacecore.linop._metric.metric_rapply` for the derivation and
        the full reference block.
        """

    def _apply_core(self, x: Any) -> Any:
        """Apply without adding validation beyond the concrete implementation."""
        return self.apply(x)

    def _rapply_core(self, y: Any) -> Any:
        """Apply the adjoint without adding validation beyond the implementation."""
        return self.rapply(y)

    def _vapply_core(self, xs: Any) -> Any:
        """Apply to a batch without adding validation beyond the implementation."""
        return self.vapply(xs)

    def _rvapply_core(self, ys: Any) -> Any:
        """Apply the adjoint to a batch without adding validation beyond the implementation."""
        return self.rvapply(ys)

    def __call__(self, x: Any) -> Any:
        """Apply this linear operator to ``x``."""
        return self.apply(x)

    def adjoint_apply(self, y: Any) -> Any:
        """Apply the adjoint of this linear operator to ``y``.

        Spelled-out alias for :meth:`rapply`; the same Definition VI.2.9
        adjoint, not a second notion.
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
        # Generic batching: vmap the single-element apply over the leading axis.
        require(self.domain, _CAP_BATCH, "LinOp.vapply")
        require(self.codomain, _CAP_BATCH, "LinOp.vapply")
        _warn_vmap_fallback_once(self, "vapply", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.apply, in_axes=0, out_axes=0)(xs)

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
        # Same for the adjoint.
        require(self.domain, _CAP_BATCH, "LinOp.rvapply")
        require(self.codomain, _CAP_BATCH, "LinOp.rvapply")
        _warn_vmap_fallback_once(self, "rvapply", _leading_batch_size(self.codomain, ys))
        return self.ops.vmap(self.rapply, in_axes=0, out_axes=0)(ys)

    @property
    def H(self) -> LinOp:
        r"""Hermitian-adjoint view of this linear operator.

        Returns
        -------
        LinOp
            Adjoint view satisfying
            :math:`\langle A x, y\rangle_Y = \langle x, A^* y\rangle_X`.
        """
        from ._algebra import _AdjointViewLinOp

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
        require(self.domain, _CAP_COORDINATE, "LinOp.to_dense")
        require(self.codomain, _CAP_COORDINATE, "LinOp.to_dense")
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
        require(self.domain, _CAP_COORDINATE, "LinOp.to_matrix")
        require(self.codomain, _CAP_COORDINATE, "LinOp.to_matrix")
        # Apply the operator to every basis vector: its images are the matrix columns.
        domain_size = prod(self.domain.shape)
        codomain_size = prod(self.codomain.shape)
        eye = self.ops.eye(domain_size, dtype=self.dtype)

        # Fast path: one batched apply over the whole identity.
        try:
            xs = self.domain.unflatten_batch(eye)
            ys = self.vapply(xs)
            ys_flat = self.codomain.flatten_batch(ys)
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
            x = self.domain.unflatten(basis_vector)
            y = self.apply(x)
            columns.append(self.codomain.flatten(y))
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
