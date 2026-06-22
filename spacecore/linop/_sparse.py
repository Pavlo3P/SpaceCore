from __future__ import annotations

from functools import cached_property
from math import prod
from typing import Any, cast

from ._base import LinOp
from ._metric import _metric_is_hermitian_by_basis, _requires_euclidean_or_riesz
from .._checks import checked_method
from ..space import (
    CoordinateSpace,
    DenseCoordinateSpace,
    DenseVectorSpace,
    ElementwiseJordanSpace,
    WeightedInnerProduct,
)
from ..types import DenseArray, SparseArray
from ..backend import jax_pytree_class, Context
from .._contextual import resolve_context_priority
from ..kernels import core_kernels
from ..kernels.core.sparse import _SparseMode


_VECTOR_SPACE_ONLY = (
    "SparseLinOp is only for coordinate sparse matrices acting between "
    "CoordinateSpace objects. Non-vector or exotic spaces should use "
    "MatrixFreeLinOp with explicit forward and adjoint callbacks."
)


@core_kernels("sparse")
@jax_pytree_class
class SparseLinOp(LinOp[CoordinateSpace, CoordinateSpace]):
    r"""
    Represent a sparse coordinate matrix-backed linear operator.

    ``SparseLinOp(A, dom, cod)`` represents a sparse coordinate matrix between
    vector spaces. Subclasses of :class:`VectorSpace` are supported, but product
    spaces and other non-vector spaces are intentionally rejected. The
    conceptual operator tensor has shape ``cod.shape + dom.shape`` while
    storage uses a two-dimensional sparse matrix with shape
    ``(prod(cod.shape), prod(dom.shape))``.

    Forward application is the raw coordinate matrix action. Adjoint
    application is metric-aware: Euclidean spaces use the conjugate transpose
    fast path, while non-Euclidean spaces use their Riesz maps as
    ``R_X^{-1} A^dagger R_Y``.

    Parameters
    ----------
    A : SparseArray
        Sparse backend matrix with shape ``(prod(cod.shape), prod(dom.shape))``.
    dom : CoordinateSpace
        Domain vector space, or a subclass of ``VectorSpace``.
    cod : CoordinateSpace
        Codomain vector space, or a subclass of ``VectorSpace``.
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
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> A = sc.SparseLinOp(ctx.assparse(sps.eye(2)), X, X, ctx)
    >>> A.apply(ctx.asarray([1.0, 2.0]))
    array([1., 2.])
    """

    def __init__(
        self,
        A: SparseArray,
        dom: CoordinateSpace,
        cod: CoordinateSpace,
        ctx: Context | str | None = None,
    ) -> None:
        ctx = resolve_context_priority(ctx, dom, cod)
        ctx.assert_sparse(A)  # Check if A is sparse array of ctx
        if not isinstance(dom, CoordinateSpace) or not isinstance(cod, CoordinateSpace):
            raise TypeError(_VECTOR_SPACE_ONLY)

        _requires_euclidean_or_riesz(dom, cod, "SparseLinOp")

        super(SparseLinOp, self).__init__(dom, cod, ctx)

        expected = (prod(self.cod.shape), prod(self.dom.shape))
        if tuple(A.shape) != expected:
            raise TypeError(
                f"Expected A.shape == (prod(cod.shape), prod(dom.shape)) == {expected}, got {A.shape}"
            )

        self._A = A  # No dtype conversion
        self._cod_size = expected[0]
        self._dom_size = expected[1]
        dtype = self.ops.get_dtype(self.A)
        self._A_is_complex = self.ops.is_complex_dtype(dtype)
        self._AT = self.A.T
        self._AH = self._sparse_conj(self._AT) if self._A_is_complex else self._AT
        self._dom_dense_array = type(self.dom) in (
            DenseCoordinateSpace,
            DenseVectorSpace,
            ElementwiseJordanSpace,
        )
        self._cod_dense_array = type(self.cod) in (
            DenseCoordinateSpace,
            DenseVectorSpace,
            ElementwiseJordanSpace,
        )
        self._dom_is_flat = self._dom_dense_array and tuple(self.dom.shape) == (self._dom_size,)
        self._cod_is_flat = self._cod_dense_array and tuple(self.cod.shape) == (self._cod_size,)
        self._mode = self._select_mode()
        if self._mode is _SparseMode.WEIGHTED_FUSED:
            self._dom_weights = self.dom.geometry.weights
            self._cod_weights = self.cod.geometry.weights

    def _select_mode(self) -> _SparseMode:
        """Select the sparse computation mode once for this operator."""
        if (
            self._dom_is_flat
            and self._cod_is_flat
            and type(getattr(self.dom, "geometry", None)) is WeightedInnerProduct
            and type(getattr(self.cod, "geometry", None)) is WeightedInnerProduct
        ):
            return _SparseMode.WEIGHTED_FUSED
        if (
            cast(Any, self.domain).is_euclidean
            and cast(Any, self.codomain).is_euclidean
            and self._dom_dense_array
            and self._cod_dense_array
        ):
            if self._dom_is_flat and self._cod_is_flat:
                return _SparseMode.EUCLIDEAN_FLAT
            return _SparseMode.EUCLIDEAN_TENSOR
        return _SparseMode.GENERAL_METRIC

    def _sparse_conj(self, A: SparseArray) -> SparseArray:
        """Return the complex conjugate of a backend sparse array."""
        if hasattr(A, "conj"):
            return A.conj()
        if hasattr(A, "conjugate"):
            return cast(Any, A).conjugate()
        if hasattr(A, "data") and hasattr(A, "indices"):
            A_any = cast(Any, A)
            kwargs = {
                "shape": A.shape,
                "indices_sorted": getattr(A, "indices_sorted", False),
                "unique_indices": getattr(A, "unique_indices", False),
            }
            return cast(Any, type(A))((self.ops.conj(A_any.data), A_any.indices), **kwargs)
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

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: DenseArray) -> DenseArray:
        """
        Forward action ``y = A @ x`` in Euclidean coordinates.

        x must have shape dom.shape (dense).
        """
        return self._apply_core(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: DenseArray) -> DenseArray:
        """
        Metric-aware adjoint action.

        y must have shape cod.shape (dense).
        """
        return self._rapply_core(y)

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: DenseArray) -> DenseArray:
        return self._vapply_core(xs)

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, ys: DenseArray) -> DenseArray:
        return self._rvapply_core(ys)

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
            dense = cast(Any, self.A).toarray()
        elif hasattr(self.A, "todense"):
            dense = cast(Any, self.A).todense()
        elif hasattr(self.A, "to_dense"):
            dense = cast(Any, self.A).to_dense()
        else:
            dense = super().to_matrix()
        return self.ops.reshape(self.ctx.asarray(dense), (self._cod_size, self._dom_size))

    def to_dense(self) -> DenseArray:
        """
        Materialize the stored sparse matrix as a dense operator tensor.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        return self.ops.reshape(
            self.to_matrix(), tuple(self.codomain.shape) + tuple(self.domain.shape)
        )

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
        if not (
            cast(Any, self.domain).is_euclidean
            and cast(Any, self.codomain).is_euclidean
        ):
            return _metric_is_hermitian_by_basis(self)
        try:
            return bool(self.ops.allclose_sparse(self.A, self._AH))
        except Exception:
            return None

    def __eq__(self, x: Any) -> bool:
        if type(x) is type(self):
            return self.dom == x.dom and self.cod == x.cod and self.ops.allclose_sparse(self.A, x.A)
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
                new_A = cast(Any, new_A).astype(new_ctx.dtype)
            elif hasattr(new_A, "to"):
                new_A = cast(Any, new_A).to(dtype=new_ctx.dtype)
            else:
                new_A = new_ctx.assparse(self.to_matrix())
        return SparseLinOp(new_A, new_dom, new_cod, new_ctx)
