from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import cached_property
from math import prod
from numbers import Number
from typing import Any, Generic, TypeVar

from .._batching import _leading_batch_size, _warn_vmap_fallback_once
from .._checks import checked_method
from ..space import Space
from ..backend import Context
from .._contextual import ContextBound

Domain = TypeVar("Domain", bound=Space)
Codomain = TypeVar("Codomain", bound=Space)


class LinOp(ContextBound, Generic[Domain, Codomain]):
    r"""
    Represent a linear map between two spaces.

    This class is intentionally small. It defines no storage assumptions and
    requires subclasses to provide forward and adjoint actions.

    The adjoint :math:`A^*` satisfies
    :math:`\langle A x, y\rangle_Y = \langle x, A^* y\rangle_X` for
    :math:`x \in X` and :math:`y \in Y`. For complex operators this is the
    conjugate adjoint.

    Parameters
    ----------
    dom : Space
        Domain space ``X``.
    cod : Space
        Codomain space ``Y``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom`` and
        ``cod``.

    Attributes
    ----------
    dom : Space
        Domain space converted to ``ctx``.
    cod : Space
        Codomain space converted to ``ctx``.
    ctx : Context
        Resolved backend context.

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

    def __init__(self, dom: Domain, cod: Codomain, ctx: Context | str | None = None):
        self.dom, self.cod = self._bind_context(ctx, dom, cod)

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
        """Apply the adjoint map to an element of ``self.codomain``."""

    def _apply_core(self, x: Any) -> Any:
        """Apply without adding validation beyond the concrete implementation."""
        return self.apply(x)

    def _rapply_core(self, y: Any) -> Any:
        """Apply the adjoint without adding validation beyond the implementation."""
        return self.rapply(y)

    def _vapply_core(self, xs: Any) -> Any:
        """Apply to a batch without adding validation beyond the implementation."""
        return self.vapply(xs)

    def __call__(self, x: Any) -> Any:
        """Apply this linear operator to ``x``."""
        return self.apply(x)

    def adjoint_apply(self, y: Any) -> Any:
        """Apply the adjoint of this linear operator to ``y``."""
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
        _warn_vmap_fallback_once(self, "vapply", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.apply, in_axes=0, out_axes=0)(xs)

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
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

        view = getattr(self, "_adjoint_view", None)
        if view is None:
            view = _AdjointViewLinOp(self)
            self._adjoint_view = view
        return view

    def __add__(self, other: Any) -> LinOp:
        """Return the lazy sum ``self + other`` of two compatible operators."""
        from ._algebra import make_sum

        if not isinstance(other, LinOp):
            return NotImplemented
        return make_sum((self, other))

    def __radd__(self, other: Any) -> LinOp:
        """Return the lazy sum ``other + self`` of two compatible operators."""
        from ._algebra import make_sum

        if isinstance(other, Number) and other == 0:
            return self
        if not isinstance(other, LinOp):
            return NotImplemented
        return make_sum((other, self))

    def __neg__(self) -> LinOp:
        """Return the lazy negation ``-self``."""
        from ._algebra import make_scaled

        return make_scaled(-1, self)

    def __sub__(self, other: Any) -> LinOp:
        """Return the lazy difference ``self - other`` of two compatible operators."""
        from ._algebra import make_scaled, make_sum

        if not isinstance(other, LinOp):
            return NotImplemented
        return make_sum((self, make_scaled(-1, other)))

    def __rsub__(self, other: Any) -> LinOp:
        """Return the lazy difference ``other - self`` of two compatible operators."""
        from ._algebra import make_scaled, make_sum

        if isinstance(other, Number) and other == 0:
            return make_scaled(-1, self)
        if not isinstance(other, LinOp):
            return NotImplemented
        return make_sum((other, make_scaled(-1, self)))

    def __mul__(self, scalar: Any) -> LinOp:
        """Return the lazy right scalar multiple ``self * scalar``."""
        from ._algebra import is_scalar_like, make_scaled

        if not is_scalar_like(scalar):
            return NotImplemented
        return make_scaled(scalar, self)

    def __rmul__(self, scalar: Any) -> LinOp:
        """Return the lazy left scalar multiple ``scalar * self``."""
        from ._algebra import is_scalar_like, make_scaled

        if not is_scalar_like(scalar):
            return NotImplemented
        return make_scaled(scalar, self)

    def __matmul__(self, other: Any) -> LinOp:
        """Return the lazy composition ``self @ other`` of two compatible operators."""
        from ._algebra import make_composed

        if not isinstance(other, LinOp):
            return NotImplemented
        return make_composed(self, other)

    def adjoint(self) -> LinOp:
        """Return the Hermitian-adjoint view of this linear operator."""
        return self.H

    def to_dense(self) -> Any:
        """
        Materialize this operator as a dense backend array.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        The default implementation is intended for small problems, debugging,
        and tests. It materializes the full coordinate matrix, so subclasses
        that already store a dense or sparse matrix should override this method
        for efficiency.
        """
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
        domain_size = prod(self.domain.shape)
        codomain_size = prod(self.codomain.shape)
        eye = self.ops.eye(domain_size, dtype=self.dtype)

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

    def __eq__(self, other: Any) -> bool:
        """Return structural equality when implemented by a subclass."""
        return NotImplemented

    @abstractmethod
    def tree_flatten(self):
        """Flatten this operator for backend pytree registration."""
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from backend pytree data."""
        ...
