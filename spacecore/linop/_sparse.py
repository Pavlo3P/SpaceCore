from __future__ import annotations

from functools import cached_property
from math import prod
from typing import Any

from ._base import LinOp, Domain, Codomain
from .._batching import _check_batched
from .._checks import checked_method
from ..space import VectorSpace
from ..types import DenseArray, SparseArray
from ..backend import jax_pytree_class, Context
from .._contextual import resolve_context_priority


@jax_pytree_class
class SparseLinOp(LinOp):
    r"""
    Represent a sparse matrix-backed linear operator.

    ``SparseLinOp(A, dom, cod)`` represents a tensor map whose conceptual shape
    is ``cod.shape + dom.shape`` while storage uses a two-dimensional sparse
    matrix with shape ``(prod(cod.shape), prod(dom.shape))``.

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
        Stored sparse matrix representation.

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
        self._AH = self._AT.conj() if self._A_is_complex else self._AT
        self._dom_is_flat = tuple(self.dom.shape) == (self._dom_size,)
        self._cod_is_flat = tuple(self.cod.shape) == (self._cod_size,)
        self._dom_vector_fast_path = type(self.dom) is VectorSpace
        self._cod_vector_fast_path = type(self.cod) is VectorSpace

    @cached_property
    def A(self) -> SparseArray:
        """
        Stored sparse matrix representation of this operator.

        The returned sparse matrix has shape
        ``(prod(self.codomain.shape), prod(self.domain.shape))`` and is the
        same object supplied at construction.
        """
        return self._A

    @checked_method(in_space="dom", out_space="cod")
    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action: y = A ⋅ x with y in cod.shape.

        x must have shape dom.shape (dense).
        """
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: DenseArray) -> DenseArray:
        """Apply the stored sparse matrix without membership checks."""
        x1 = x if self._dom_is_flat else x.reshape((self._dom_size,))
        y1 = self.A @ x1   # (m,)
        if self._cod_vector_fast_path:
            return y1 if self._cod_is_flat else y1.reshape(self.cod.shape)
        return self.cod.unflatten(y1)

    @checked_method(in_space="cod", out_space="dom")
    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Adjoint action: x = A^* ⋅ y with x in dom.shape.

        y must have shape cod.shape (dense).
        """
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: DenseArray) -> DenseArray:
        """Apply the stored sparse adjoint without membership checks."""
        y1 = y if self._cod_is_flat else y.reshape((self._cod_size,))
        x1 = self._AH @ y1

        if self._dom_vector_fast_path:
            return x1 if self._dom_is_flat else x1.reshape(self.dom.shape)
        return self.dom.unflatten(x1)

    def vapply(self, xs: DenseArray) -> DenseArray:
        if self._enable_checks:
            _check_batched(self.domain, xs)
        lead = tuple(xs.shape[: len(xs.shape) - len(self.dom.shape)])
        xs2 = xs.reshape((-1, self._dom_size))
        ys2 = (self.A @ xs2.T).T
        return ys2.reshape(lead + tuple(self.cod.shape))

    def rvapply(self, ys: DenseArray) -> DenseArray:
        if self._enable_checks:
            _check_batched(self.codomain, ys)
        lead = tuple(ys.shape[: len(ys.shape) - len(self.cod.shape)])
        ys2 = ys.reshape((-1, self._cod_size))
        xs2 = (self._AH @ ys2.T).T
        return xs2.reshape(lead + tuple(self.dom.shape))

    def to_dense(self) -> DenseArray:
        """
        Materialize the stored sparse matrix as a dense operator tensor.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        if hasattr(self.A, "toarray"):
            dense = self.A.toarray()
        elif hasattr(self.A, "todense"):
            dense = self.A.todense()
        elif hasattr(self.A, "to_dense"):
            dense = self.A.to_dense()
        else:
            dense = super().to_dense().reshape((self._cod_size, self._dom_size))
        return self.ops.reshape(dense, tuple(self.codomain.shape) + tuple(self.domain.shape))

    def is_hermitian(self) -> bool | None:
        """
        Return whether this sparse operator is structurally Hermitian.

        Returns
        -------
        bool or None
            ``True`` or ``False`` for plain :class:`VectorSpace` domains, where
            Hermiticity is checked against the Euclidean sparse matrix.
            ``None`` for custom geometries whose inner product may differ from
            the Euclidean coordinate product.
        """
        if self.dom != self.cod:
            return False
        if type(self.dom) is not VectorSpace:
            return None
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
        return SparseLinOp(new_A, new_dom, new_cod, new_ctx)
