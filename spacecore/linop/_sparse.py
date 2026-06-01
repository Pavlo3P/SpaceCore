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
from ..space import VectorSpace, WeightedInnerProduct
from ..types import DenseArray, SparseArray
from ..backend import jax_pytree_class, Context
from .._contextual import resolve_context_priority


@jax_pytree_class
class SparseLinOp(LinOp[Domain, Codomain]):
    r"""
    Represent a sparse coordinate matrix-backed linear operator.

    ``SparseLinOp(A, dom, cod)`` represents a sparse coordinate matrix between
    spaces. The conceptual operator tensor has shape ``cod.shape + dom.shape``
    while storage uses a two-dimensional sparse matrix with shape
    ``(prod(cod.shape), prod(dom.shape))``.

    Forward application is the raw coordinate matrix action. Adjoint
    application is metric-aware: Euclidean spaces use the conjugate transpose
    fast path, while non-Euclidean spaces use their Riesz maps as
    ``R_X^{-1} A^dagger R_Y``.

    Parameters
    ----------
    A : SparseArray
        Sparse backend matrix with shape ``(prod(cod.shape), prod(dom.shape))``.
    dom : Space
        Domain space.
    cod : Space
        Codomain space.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from the spaces.

    Attributes
    ----------
    A : SparseArray
        Stored sparse matrix representation. The constructor keeps this object
        without sparse conversion or copying; explicit conversion happens only
        through :meth:`_convert`.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sps
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.VectorSpace((2,), ctx)
    >>> A = sc.SparseLinOp(ctx.assparse(sps.eye(2)), X, X, ctx)
    >>> A.apply(ctx.asarray([1.0, 2.0]))
    array([1., 2.])
    """

    def __init__(self,
                 A: SparseArray,
                 dom: Domain,
                 cod: Codomain,
                 ctx: Context | str | None = None
                 ) -> None:
        ctx = resolve_context_priority(ctx, dom, cod)
        ctx.assert_sparse(A)  # Check if A is sparse array of ctx

        _requires_euclidean_or_riesz(dom, cod, "SparseLinOp")

        super(SparseLinOp, self).__init__(dom, cod, ctx)

        expected = (prod(self.cod.shape), prod(self.dom.shape))
        if tuple(A.shape) != expected:
            raise TypeError(f"Expected A.shape == (prod(cod.shape), prod(dom.shape)) == {expected}, got {A.shape}")

        self._A = A  # No dtype conversion
        self._cod_size = expected[0]
        self._dom_size = expected[1]
        dtype = self.ops.get_dtype(self.A)
        self._A_is_complex = self.ops.is_complex_dtype(dtype)
        self._AT = self.A.T
        self._AH = self._sparse_conj(self._AT) if self._A_is_complex else self._AT
        self._dom_is_flat = tuple(self.dom.shape) == (self._dom_size,)
        self._cod_is_flat = tuple(self.cod.shape) == (self._cod_size,)
        self._dom_vector_fast_path = type(self.dom) is VectorSpace
        self._cod_vector_fast_path = type(self.cod) is VectorSpace
        self._weighted_flat_adjoint_fast_path = (
            self._dom_vector_fast_path
            and self._cod_vector_fast_path
            and self._dom_is_flat
            and self._cod_is_flat
            and type(self.dom.geometry) is WeightedInnerProduct
            and type(self.cod.geometry) is WeightedInnerProduct
        )
        if self._weighted_flat_adjoint_fast_path:
            self._dom_weights = self.dom.geometry.weights
            self._cod_weights = self.cod.geometry.weights
        if not self._enable_checks:
            self._install_unchecked_fast_methods()

    def _install_unchecked_fast_methods(self) -> None:
        """Install direct no-check callables for exact vector-space hot paths."""
        if not (self._dom_vector_fast_path and self._cod_vector_fast_path):
            self.apply = self._apply_unchecked
            self.rapply = self._rapply_unchecked
            self.vapply = self._vapply_unchecked
            self.rvapply = self._rvapply_unchecked
            return

        A = self._A
        AH = self._AH
        dom_size = self._dom_size
        cod_size = self._cod_size
        cod_shape = tuple(self.cod.shape)
        dom_shape = tuple(self.dom.shape)
        dom_is_flat = self._dom_is_flat
        cod_is_flat = self._cod_is_flat
        if dom_is_flat and cod_is_flat:
            self.apply = lambda x, A=A: A @ x
            self.vapply = lambda xs, A=A, dom_size=dom_size: (A @ xs.reshape((-1, dom_size)).T).T
        else:
            self.apply = lambda x, A=A, dom_size=dom_size, cod_shape=cod_shape: (A @ x.reshape((dom_size,))).reshape(cod_shape)
            self.vapply = (
                lambda xs, A=A, dom_size=dom_size, cod_shape=cod_shape:
                (A @ xs.reshape((-1, dom_size)).T).T.reshape(tuple(xs.shape[: len(xs.shape) - len(dom_shape)]) + cod_shape)
            )

        if self._weighted_flat_adjoint_fast_path:
            cod_weights = self._cod_weights
            dom_weights = self._dom_weights
            self.rapply = lambda y, AH=AH, cod_weights=cod_weights, dom_weights=dom_weights: (AH @ (cod_weights * y)) / dom_weights
            self.rvapply = lambda ys, AH=AH, cod_weights=cod_weights, dom_weights=dom_weights: (AH @ (ys * cod_weights).T).T / dom_weights
        elif self.domain.is_euclidean and self.codomain.is_euclidean:
            if cod_is_flat and dom_is_flat:
                self.rapply = lambda y, AH=AH: AH @ y
                self.rvapply = lambda ys, AH=AH, cod_size=cod_size: (AH @ ys.reshape((-1, cod_size)).T).T
            else:
                self.rapply = lambda y, AH=AH, cod_size=cod_size, dom_shape=dom_shape: (AH @ y.reshape((cod_size,))).reshape(dom_shape)
                self.rvapply = (
                    lambda ys, AH=AH, cod_size=cod_size, dom_shape=dom_shape:
                    (AH @ ys.reshape((-1, cod_size)).T).T.reshape(tuple(ys.shape[: len(ys.shape) - len(cod_shape)]) + dom_shape)
                )
        else:
            self.rapply = self._rapply_unchecked
            self.rvapply = self._rvapply_unchecked

    def _sparse_conj(self, A: SparseArray) -> SparseArray:
        """Return the complex conjugate of a backend sparse array."""
        if hasattr(A, "conj"):
            return A.conj()
        if hasattr(A, "conjugate"):
            return A.conjugate()
        if hasattr(A, "data") and hasattr(A, "indices"):
            kwargs = {
                "shape": A.shape,
                "indices_sorted": getattr(A, "indices_sorted", False),
                "unique_indices": getattr(A, "unique_indices", False),
            }
            return type(A)((self.ops.conj(A.data), A.indices), **kwargs)
        raise TypeError(f"Cannot conjugate sparse array of type {type(A).__name__}.")

    @cached_property
    def A(self) -> SparseArray:
        """
        Stored sparse matrix representation of this operator.

        The returned sparse matrix has shape
        ``(prod(self.codomain.shape), prod(self.domain.shape))`` and is the
        same object supplied at construction.
        """
        return self._A

    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action ``y = A @ x`` in Euclidean coordinates.

        x must have shape dom.shape (dense).
        """
        if (
            not self._enable_checks
            and self._dom_vector_fast_path
            and self._cod_vector_fast_path
        ):
            if self._dom_is_flat:
                y1 = self._A @ x
            else:
                y1 = self._A @ x.reshape((self._dom_size,))
            return y1 if self._cod_is_flat else y1.reshape(self.cod.shape)
        if self._enable_checks:
            self.dom._check_member(x)
        y = self._apply_unchecked(x)
        if self._enable_checks:
            self.cod._check_member(y)
        return y

    def _apply_unchecked(self, x: DenseArray) -> DenseArray:
        """Apply the stored sparse matrix without membership checks."""
        if self._dom_vector_fast_path:
            x1 = x if self._dom_is_flat else x.reshape((self._dom_size,))
        else:
            x1 = self.dom.flatten(x)
        y1 = self._A @ x1   # (m,)
        if self._cod_vector_fast_path:
            return y1 if self._cod_is_flat else y1.reshape(self.cod.shape)
        return self.cod.unflatten(y1)

    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Metric-aware adjoint action.

        y must have shape cod.shape (dense).
        """
        if self._enable_checks:
            self.cod._check_member(y)
        x = self._rapply_unchecked(y)
        if self._enable_checks:
            self.dom._check_member(x)
        return x

    def _rapply_unchecked(self, y: DenseArray) -> DenseArray:
        """Apply the metric adjoint without membership checks."""
        if self._weighted_flat_adjoint_fast_path:
            return (self._AH @ (self._cod_weights * y)) / self._dom_weights
        if self.domain.is_euclidean and self.codomain.is_euclidean:
            return self._euclidean_rapply_unchecked(y)
        yd = self.codomain.riesz(y)
        tmp = self._euclidean_rapply_unchecked(yd)
        return self.domain.riesz_inverse(tmp)

    def _euclidean_rapply_unchecked(self, y: DenseArray) -> DenseArray:
        """Apply the stored sparse adjoint without membership checks."""
        if self._cod_vector_fast_path:
            y1 = y if self._cod_is_flat else y.reshape((self._cod_size,))
        else:
            y1 = self.cod.flatten(y)
        x1 = self._AH @ y1
        if self._dom_vector_fast_path:
            return x1 if self._dom_is_flat else x1.reshape(self.dom.shape)
        return self.dom.unflatten(x1)

    def vapply(self, xs: DenseArray) -> DenseArray:
        if self._enable_checks:
            _check_batched(self.domain, xs)
        return self._vapply_unchecked(xs)

    def _vapply_unchecked(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis without membership checks."""
        if self._dom_vector_fast_path and self._cod_vector_fast_path:
            lead = tuple(xs.shape[: len(xs.shape) - len(self.dom.shape)])
            xs2 = xs.reshape((-1, self._dom_size))
            ys2 = (self._A @ xs2.T).T
            return ys2.reshape(lead + tuple(self.cod.shape))
        xs_flat = self.domain.flatten_batch(xs)
        ys_flat = (self._A @ xs_flat.T).T
        return self.codomain.unflatten_batch(ys_flat)

    def rvapply(self, ys: DenseArray) -> DenseArray:
        if self._enable_checks:
            _check_batched(self.codomain, ys)
        xs = self._rvapply_unchecked(ys)
        if self._enable_checks:
            _check_batched(self.domain, xs)
        return xs

    def _rvapply_unchecked(self, ys: DenseArray) -> DenseArray:
        """Apply the metric adjoint over leading batch axes without checks."""
        if self._weighted_flat_adjoint_fast_path:
            return (self._AH @ (ys * self._cod_weights).T).T / self._dom_weights
        if not (self.domain.is_euclidean and self.codomain.is_euclidean):
            try:
                yd = self.codomain.riesz(ys)
                tmp = self._euclidean_rvapply_unchecked(yd)
                return self.domain.riesz_inverse(tmp)
            except _METRIC_BATCH_FALLBACK_ERRORS as err:
                _warn_metric_batch_fallback(type(self).__name__, err)
                return self.ops.vmap(self.rapply, in_axes=0, out_axes=0)(ys)
        return self._euclidean_rvapply_unchecked(ys)

    def _euclidean_rvapply_unchecked(self, ys: DenseArray) -> DenseArray:
        """Apply the Euclidean sparse adjoint over leading batch axes."""
        if self._cod_vector_fast_path and self._dom_vector_fast_path:
            lead = tuple(ys.shape[: len(ys.shape) - len(self.cod.shape)])
            ys2 = ys.reshape((-1, self._cod_size))
            xs2 = (self._AH @ ys2.T).T
            return xs2.reshape(lead + tuple(self.dom.shape))
        ys_flat = self.codomain.flatten_batch(ys)
        xs_flat = (self._AH @ ys_flat.T).T
        return self.domain.unflatten_batch(xs_flat)

    def to_sparse(self) -> SparseArray:
        """
        Return the stored sparse matrix representation without copying.

        The returned object is exactly the sparse array supplied at construction.
        """
        return self.A

    def to_matrix(self) -> DenseArray:
        """
        Materialize the stored sparse matrix as a dense 2D coordinate matrix.

        Use :meth:`to_sparse` when sparse storage should be preserved.
        """
        if hasattr(self.A, "toarray"):
            dense = self.A.toarray()
        elif hasattr(self.A, "todense"):
            dense = self.A.todense()
        elif hasattr(self.A, "to_dense"):
            dense = self.A.to_dense()
        else:
            dense = super().to_matrix()
        return self.ops.reshape(self.ctx.asarray(dense), (self._cod_size, self._dom_size))

    def to_dense(self) -> DenseArray:
        """
        Materialize the stored sparse matrix as a dense operator tensor.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        return self.ops.reshape(self.to_matrix(), tuple(self.codomain.shape) + tuple(self.domain.shape))

    def is_hermitian(self) -> bool | None:
        """
        Return whether this sparse operator is structurally self-adjoint.

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
            return bool(self.ops.allclose_sparse(self.A, self._AH))
        except Exception:
            return None

    def __eq__(self, x: Any) -> bool:
        if type(x) is type(self):
            return (self.dom == x.dom
                and self.cod == x.cod
                and self.ops.allclose_sparse(self.A, x.A)
            )
        return False

    def tree_flatten(self):
        aux = (self.dom, self.cod, self.ctx)
        children = (self.A,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod, ctx = aux
        A = children[0]
        return cls(A, dom, cod, ctx)

    def _convert(self, new_ctx: Context) -> SparseLinOp:
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_A = new_ctx.assparse(self.A)
        if new_ctx.ops.get_dtype(new_A) != new_ctx.dtype:
            if hasattr(new_A, "astype"):
                new_A = new_A.astype(new_ctx.dtype)
            elif hasattr(new_A, "to"):
                new_A = new_A.to(dtype=new_ctx.dtype)
            else:
                new_A = new_ctx.assparse(self.to_matrix())
        return SparseLinOp(new_A, new_dom, new_cod, new_ctx)
