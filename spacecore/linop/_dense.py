from __future__ import annotations

from functools import cached_property
from math import prod
from typing import Any

from ._base import Codomain, Domain, LinOp
from ._metric import (
    _METRIC_BATCH_FALLBACK_ERRORS,
    _metric_is_hermitian_by_basis,
    _requires_euclidean_or_riesz,
    _warn_metric_batch_fallback,
)
from .._batching import _check_batched
from .._checks import checked_method
from ..space import VectorSpace
from ..types import DenseArray
from ..backend import jax_pytree_class, Context
from .._contextual import resolve_context_priority


@jax_pytree_class
class DenseLinOp(LinOp[Domain, Codomain]):
    r"""
    Represent a dense coordinate tensor-backed linear operator.

    ``DenseLinOp(A, dom, cod)`` represents a linear map
    :math:`A \colon X \to Y` where the stored dense array has shape
    ``cod.shape + dom.shape``. Forward application is the raw coordinate
    matrix action. Adjoint application is metric-aware: Euclidean spaces use
    the conjugate transpose fast path, while non-Euclidean spaces use their
    Riesz maps as ``R_X^{-1} A^dagger R_Y``.

    DenseLinOp does not copy or cast the input array. The caller is responsible
    for passing an array compatible with `ctx`. This avoids duplicating large dense
    operators in memory.

    Parameters
    ----------
    A : DenseArray
        Dense backend array with shape ``cod.shape + dom.shape``.
    dom : Space
        Domain space.
    cod : Space or None, optional
        Codomain space. If omitted, it is inferred from the leading axes of
        ``A``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from the spaces.

    Attributes
    ----------
    A : DenseArray
        Stored dense operator tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.VectorSpace((2,), ctx)
    >>> A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, ctx)
    >>> A.apply(ctx.asarray([1.0, 2.0]))
    array([2., 6.])
    """

    def __init__(self,
                 A: DenseArray,
                 dom: Domain,
                 cod: Codomain | None = None,
                 ctx: Context | str | None = None
                 ) -> None:
        ctx = resolve_context_priority(ctx, dom, cod)
        ctx.assert_dense(A)  # Check if A is ndarray of ctx

        if cod is None:
            cod_shape_len = len(A.shape) - len(dom.shape)
            cod = VectorSpace(A.shape[:cod_shape_len], ctx)

        _requires_euclidean_or_riesz(dom, cod, "DenseLinOp")

        super(DenseLinOp, self).__init__(dom, cod, ctx)

        expected = tuple(self.cod.shape) + tuple(self.dom.shape)
        if tuple(A.shape) != expected:
            raise TypeError(f"Expected A.shape == cod.shape + dom.shape == {expected}, got {A.shape}")

        self._A = A  # Intentionally no dtype/backend conversion to avoid extra memory use.
        self._cod_size = prod(self.cod.shape)
        self._dom_size = prod(self.dom.shape)
        self._matrix_shape = (self._cod_size, self._dom_size)
        self._A2 = self.A.reshape(self._matrix_shape)
        dtype = self.ops.get_dtype(self.A)
        is_complex = self.ops.is_complex_dtype(dtype)
        self._A2T = self._A2.T
        self._A2H = self._A2.T.conj() if is_complex else self._A2.T
        self._dom_is_flat = tuple(self.dom.shape) == (self._dom_size,)
        self._cod_is_flat = tuple(self.cod.shape) == (self._cod_size,)
        self._dom_vector_fast_path = type(self.dom) is VectorSpace
        self._cod_vector_fast_path = type(self.cod) is VectorSpace

    @cached_property
    def A(self) -> DenseArray:
        """
        Stored dense tensor representation of this operator.

        The returned array has shape ``self.codomain.shape + self.domain.shape``
        and is the same object supplied at construction.
        """
        return self._A

    @checked_method(in_space="dom", out_space="cod")
    def apply(self, x: DenseArray) -> DenseArray:
        """Apply the dense operator to ``x``."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: DenseArray) -> DenseArray:
        """Apply the flattened dense matrix without membership checks."""
        if self._dom_vector_fast_path:
            x1 = x if self._dom_is_flat else x.reshape((self._dom_size,))
        else:
            x1 = self.dom.flatten(x)
        y1 = self._A2 @ x1
        if self._cod_vector_fast_path:
            return y1 if self._cod_is_flat else y1.reshape(self.cod.shape)
        return self.cod.unflatten(y1)

    @checked_method(in_space="cod", out_space="dom")
    def rapply(self, y: DenseArray) -> DenseArray:
        r"""Apply the adjoint dense operator to ``y``.

        Euclidean spaces use the conjugate transpose of the flattened matrix.
        Non-Euclidean spaces apply the codomain and domain Riesz maps around
        that Euclidean adjoint.
        """
        if self.domain.is_euclidean and self.codomain.is_euclidean:
            return self._euclidean_rapply_unchecked(y)
        yd = self.codomain.riesz(y)
        tmp = self._euclidean_rapply_unchecked(yd)
        return self.domain.riesz_inverse(tmp)

    def _euclidean_rapply_unchecked(self, y: DenseArray) -> DenseArray:
        """Apply the flattened adjoint matrix without membership checks."""
        if self._cod_vector_fast_path:
            y1 = y if self._cod_is_flat else y.reshape((self._cod_size,))
        else:
            y1 = self.cod.flatten(y)
        x1 = self._A2H @ y1
        if self._dom_vector_fast_path:
            return x1 if self._dom_is_flat else x1.reshape(self.dom.shape)
        return self.dom.unflatten(x1)

    def vapply(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        if self._enable_checks:
            _check_batched(self.domain, xs)
        if self._dom_vector_fast_path and self._cod_vector_fast_path:
            lead = tuple(xs.shape[: len(xs.shape) - len(self.dom.shape)])
            xs2 = xs.reshape((-1, self._dom_size))
            ys2 = xs2 @ self._A2T
            return ys2.reshape(lead + tuple(self.cod.shape))
        xs_flat = self.domain.flatten_batch(xs)
        ys_flat = xs_flat @ self._A2T
        return self.codomain.unflatten_batch(ys_flat)

    def rvapply(self, ys: DenseArray) -> DenseArray:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
        if self._enable_checks:
            _check_batched(self.codomain, ys)
        if not (self.domain.is_euclidean and self.codomain.is_euclidean):
            try:
                yd = self.codomain.riesz(ys)
                tmp = self._euclidean_rvapply_unchecked(yd)
                xs = self.domain.riesz_inverse(tmp)
                if self._enable_checks:
                    _check_batched(self.domain, xs)
                return xs
            except _METRIC_BATCH_FALLBACK_ERRORS as err:
                _warn_metric_batch_fallback(type(self).__name__, err)
                return self.ops.vmap(self.rapply, in_axes=0, out_axes=0)(ys)
        return self._euclidean_rvapply_unchecked(ys)

    def _euclidean_rvapply_unchecked(self, ys: DenseArray) -> DenseArray:
        """Apply the Euclidean adjoint over leading batch axes."""
        if self._cod_vector_fast_path and self._dom_vector_fast_path:
            lead = tuple(ys.shape[: len(ys.shape) - len(self.cod.shape)])
            ys2 = ys.reshape((-1, self._cod_size))
            xs2 = ys2 @ self._A2H.T
            return xs2.reshape(lead + tuple(self.dom.shape))
        ys_flat = self.codomain.flatten_batch(ys)
        xs_flat = ys_flat @ self._A2H.T
        return self.domain.unflatten_batch(xs_flat)

    def to_dense(self) -> DenseArray:
        """
        Return the stored dense tensor representation of this operator.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        return self.A

    def to_matrix(self) -> DenseArray:
        """
        Return the flattened dense matrix representation.

        The returned array has shape
        ``(prod(self.codomain.shape), prod(self.domain.shape))``.
        It is a reshape/view of the stored dense tensor when the backend permits.
        """
        return self._A2

    def is_hermitian(self) -> bool | None:
        """
        Return whether this dense operator is structurally self-adjoint.

        Returns
        -------
        bool or None
            ``True`` or ``False`` when the structure can be checked, otherwise
            ``None``.
        """
        if self.dom != self.cod:
            return False
        if not (self.domain.is_euclidean and self.codomain.is_euclidean):
            return _metric_is_hermitian_by_basis(self)
        try:
            return bool(self.ops.allclose(self._A2, self._A2H))
        except Exception:
            return None

    def __eq__(self, x: Any) -> bool:
        """Return whether another dense operator has the same spaces and values."""
        if type(x) is type(self):
            return (self.dom == x.dom
                and self.cod == x.cod
                and self.ops.allclose(self.A, x.A)
            )
        return False

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        aux = (self.dom, self.cod, self.ctx)
        children = (self.A,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        dom, cod, ctx = aux
        A = children[0]
        return cls(A, dom, cod, ctx)

    def _convert(self, new_ctx: Context) -> DenseLinOp:
        """Convert spaces and stored dense tensor to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_A = new_ctx.asarray(self.A)
        return DenseLinOp(new_A, new_dom, new_cod, new_ctx)
