from __future__ import annotations

from enum import Enum, auto
from functools import cached_property
from math import prod
from typing import Any, cast

from ._base import LinOp
from ._metric import (
    _metric_is_hermitian_by_basis, _requires_euclidean_or_riesz, metric_rapply, metric_rvapply,
)
from .._checks import CheckLevel
from .._checks import checked_method
from ..space import (
    CoordinateSpace,
    DenseCoordinateSpace,
    DenseVectorSpace,
    ElementwiseJordanSpace,
    WeightedInnerProduct,
)
from ..types import DenseArray, SparseArray
from ..contextual import Context
from ..contextual import resolve_context_priority


_VECTOR_SPACE_ONLY = (
    "SparseLinOp is only for coordinate sparse matrices acting between "
    "CoordinateSpace objects. Non-vector or exotic spaces should use "
    "MatrixFreeLinOp with explicit forward and adjoint callbacks."
)


class _SparseMode(Enum):
    """Private computation modes for sparse coordinate operators."""

    EUCLIDEAN_FLAT = auto()
    EUCLIDEAN_TENSOR = auto()
    WEIGHTED_FUSED = auto()
    GENERAL_METRIC = auto()



class SparseLinOp(LinOp[CoordinateSpace, CoordinateSpace]):
    r"""
    Linear operator stored as a sparse matrix of shape ``(cod.size, dom.size)``.

    Domain and codomain must be vector spaces. The adjoint is the conjugate transpose
    on Euclidean spaces and ``R_X^{-1} A^dagger R_Y`` otherwise. The matrix is used as
    given.

    Parameters
    ----------
    A : SparseArray
        Sparse matrix of shape ``(prod(cod.shape), prod(dom.shape))``.
    dom : CoordinateSpace
        Domain.
    cod : CoordinateSpace
        Codomain.
    ctx : Context, str, or None, optional
        Backend context; resolved from the spaces when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

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

    def _apply_core(self, x: Any) -> Any:
        # Flat or fused: one matvec. Tensor: reshape around it. Else the spaces' flatten/unflatten.
        if self._mode is _SparseMode.EUCLIDEAN_FLAT:
            return self._A @ x
        if self._mode is _SparseMode.EUCLIDEAN_TENSOR:
            return (self._A @ x.reshape((self._dom_size,))).reshape(self.cod.shape)
        if self._mode is _SparseMode.WEIGHTED_FUSED:
            return self._A @ x
        x1 = self.dom.flatten(x)
        y1 = self._A @ x1
        return self.cod.unflatten(y1)

    def _rapply_core(self, y: Any) -> Any:
        # Euclidean: A^H. Diagonal weights: W_X^{-1} A^H W_Y y. Else wrap A^H with the metrics.
        if self._mode is _SparseMode.EUCLIDEAN_FLAT or self._mode is _SparseMode.EUCLIDEAN_TENSOR:
            return self._euclidean_rapply_core(y)
        if self._mode is _SparseMode.WEIGHTED_FUSED:
            return (self._AH @ (self._cod_weights * y)) / self._dom_weights
        return metric_rapply(
            self.domain, self.codomain, self._euclidean_rapply_core, y
        )

    def _vapply_core(self, xs: Any) -> Any:
        # Sparse @ dense wants the batch as columns: (A X^T)^T.
        if self._mode is _SparseMode.EUCLIDEAN_FLAT:
            return (self._A @ xs.reshape((-1, self._dom_size)).T).T
        if self._mode is _SparseMode.EUCLIDEAN_TENSOR:
            lead = tuple(xs.shape[: len(xs.shape) - len(self.dom.shape)])
            xs2 = xs.reshape((-1, self._dom_size))
            ys2 = (self._A @ xs2.T).T
            return ys2.reshape(lead + tuple(self.cod.shape))
        if self._mode is _SparseMode.WEIGHTED_FUSED:
            return (self._A @ xs.reshape((-1, self._dom_size)).T).T
        xs_flat = self.domain.flatten_batch(xs)
        ys_flat = (self._A @ xs_flat.T).T
        return self.codomain.unflatten_batch(ys_flat)

    def _rvapply_core(self, ys: Any) -> Any:
        # Batched form of _rapply_core.
        if self._mode is _SparseMode.EUCLIDEAN_FLAT or self._mode is _SparseMode.EUCLIDEAN_TENSOR:
            return self._euclidean_rvapply_core(ys)
        if self._mode is _SparseMode.WEIGHTED_FUSED:
            return (self._AH @ (ys * self._cod_weights).T).T / self._dom_weights
        return metric_rvapply(
            self.domain,
            self.codomain,
            self._euclidean_rapply_core,
            self._euclidean_rvapply_core,
            ys,
            opname=type(self).__name__,
            ops=self.ops,
        )

    def _euclidean_rapply_core(self, y: Any) -> Any:
        # Plain conjugate transpose A^H, no metric.
        if self._mode is _SparseMode.EUCLIDEAN_FLAT:
            return self._AH @ y
        if self._mode is _SparseMode.EUCLIDEAN_TENSOR:
            return (self._AH @ y.reshape((self._cod_size,))).reshape(self.dom.shape)
        y1 = self.cod.flatten(y)
        x1 = self._AH @ y1
        return self.dom.unflatten(x1)

    def _euclidean_rvapply_core(self, ys: Any) -> Any:
        # (A^H Y^T)^T for a batch of rows.
        if self._mode is _SparseMode.EUCLIDEAN_FLAT:
            return (self._AH @ ys.reshape((-1, self._cod_size)).T).T
        if self._mode is _SparseMode.EUCLIDEAN_TENSOR:
            lead = tuple(ys.shape[: len(ys.shape) - len(self.cod.shape)])
            ys2 = ys.reshape((-1, self._cod_size))
            xs2 = (self._AH @ ys2.T).T
            return xs2.reshape(lead + tuple(self.dom.shape))
        ys_flat = self.codomain.flatten_batch(ys)
        xs_flat = (self._AH @ ys_flat.T).T
        return self.domain.unflatten_batch(xs_flat)

    def __init__(
        self,
        A: SparseArray,
        dom: CoordinateSpace,
        cod: CoordinateSpace,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        # A must be a sparse array of the backend; spaces must be coordinate spaces.
        ctx = resolve_context_priority(ctx, dom, cod)
        ctx.assert_sparse(A)  # Check if A is sparse array of ctx
        if not isinstance(dom, CoordinateSpace) or not isinstance(cod, CoordinateSpace):
            raise TypeError(_VECTOR_SPACE_ONLY)

        _requires_euclidean_or_riesz(dom, cod, "SparseLinOp")

        super(SparseLinOp, self).__init__(dom, cod, ctx, check_level=check_level)

        # Storage is always the flattened 2-D matrix.
        expected = (prod(self.cod.shape), prod(self.dom.shape))
        if tuple(A.shape) != expected:
            raise TypeError(
                f"Expected A.shape == (prod(cod.shape), prod(dom.shape)) == {expected}, got {A.shape}"
            )

        # Cache A^T and A^H once, and note which spaces are plain dense arrays.
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
        # Pick the apply strategy once.
        self._mode = self._select_mode()
        if self._mode is _SparseMode.WEIGHTED_FUSED:
            self._dom_weights = self.dom.geometry.weights
            self._cod_weights = self.cod.geometry.weights

    def _select_mode(self) -> _SparseMode:
        """Select the sparse computation mode once for this operator."""
        # Weighted flat vectors: fused path; Euclidean dense arrays: plain SpMV;
        # anything else: the general Riesz-map path.
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
        # Conjugate whatever sparse type the backend uses.
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
        # Densify with the backend's native method, else by applying to a basis.
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
        # Hermitian needs dom == cod; Euclidean compares A with A^H, else test by basis.
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

    def __eq__(self, other: Any) -> bool:
        if not self.same_math(other):                  # Tier 1: backend
            return NotImplemented
        if self.dom != other.dom or self.cod != other.cod:          # Tier 2: spaces before allclose
            return False
        # Tier 3: sparse values. Note: allclose_sparse has no equal_nan, so a
        # structural NaN compares unequal to itself (rare for sparse storage).
        return bool(self.ops.allclose_sparse(self.A, other.A))

    def _repr_body(self) -> str:
        """Lead with the arrow; add the (cheap) stored non-zero count when known."""
        nnz = self._stored_nnz()
        return self._arrow() if nnz is None else f"{self._arrow()}, nnz={nnz}"

    def _stored_nnz(self) -> int | None:
        """Return the stored non-zero count across backends, or ``None``.

        Tries the common spellings: ``nnz`` (scipy/cupy), ``nse`` (jax BCOO),
        and ``_nnz()`` (torch). Returns ``None`` when none is cheaply available.
        """
        # Stored non-zeros, whatever the backend calls the count.
        A = self._A
        for attr in ("nnz", "nse"):
            value = getattr(A, attr, None)
            if isinstance(value, int):
                return value
        method = getattr(A, "_nnz", None)
        if callable(method):
            try:
                return int(cast(Any, method()))
            except Exception:
                return None
        return None

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
        # Move the spaces and the matrix, fixing the dtype if conversion kept the old one.
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
