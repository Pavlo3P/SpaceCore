from __future__ import annotations

import warnings

from enum import Enum, auto
from functools import cached_property
from math import prod
from typing import Any, cast

from ._base import Codomain, Domain, LinOp, _output_checked_rvapply
from .._checks import CheckLevel
from ._metric import (
    _VECTOR_SPACES, _metric_is_hermitian_by_basis, _requires_euclidean_or_riesz, metric_rapply,
    metric_rvapply,
)
from ..space import (
    DenseCoordinateSpace,
    WeightedInnerProduct,
)
from ..types import DenseArray
from ..contextual import Context
from ..contextual import resolve_context_priority


class _DenseMode(Enum):
    """Private computation modes for dense coordinate operators."""

    EUCLIDEAN_FLAT = auto()
    EUCLIDEAN_TENSOR = auto()
    WEIGHTED_FUSED = auto()
    GENERAL_METRIC = auto()



class DenseLinOp(LinOp[Domain, Codomain]):
    r"""
    Linear operator stored as a dense array of shape ``cod.shape + dom.shape``.

    The adjoint is the conjugate transpose on Euclidean spaces and
    ``R_X^{-1} A^dagger R_Y`` otherwise. The array is used as given, not copied or cast.

    Parameters
    ----------
    A : DenseArray
        Operator tensor of shape ``cod.shape + dom.shape``.
    dom : Space
        Domain.
    cod : Space or None, optional
        Codomain. When omitted it is a *Euclidean* space of ``A``'s leading shape, so for
        an endomorphism of a weighted ``X`` pass ``cod=X`` explicitly.
    ctx : Context, str, or None, optional
        Backend context; resolved from the spaces when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
    >>> A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, ctx)
    >>> A.apply(ctx.asarray([1.0, 2.0]))
    array([2., 6.])
    """

    _fields = ("A", "dom", "cod", "ctx")
    _dynamic = ("A",)

    def _apply_core(self, x: Any) -> Any:
        # Flat or fused: one matvec. Tensor: reshape around it. Else the spaces' flatten/unflatten.
        if self._mode is _DenseMode.EUCLIDEAN_FLAT or self._mode is _DenseMode.WEIGHTED_FUSED:
            return self._A2 @ x
        if self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            return (self._A2 @ x.reshape((self._dom_size,))).reshape(self.cod.shape)
        x1 = self.dom._flatten_core(x)
        y1 = self._A2 @ x1
        return self.cod._unflatten_core(y1)

    def _rapply_core(self, y: Any) -> Any:
        # Euclidean: A^H. Fused: the precomputed weighted adjoint. Else wrap A^H with the metrics.
        if self._mode is _DenseMode.EUCLIDEAN_FLAT or self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            return self._euclidean_rapply_core(y)
        if self._mode is _DenseMode.WEIGHTED_FUSED:
            return self._weighted_A2H @ y
        return metric_rapply(
            self.domain, self.codomain, self._euclidean_rapply_core, y
        )

    def _vapply_core(self, xs: Any) -> Any:
        # A batch of rows: X A^T on flat coordinates, reshaped to the codomain.
        if self._mode is _DenseMode.EUCLIDEAN_FLAT or self._mode is _DenseMode.WEIGHTED_FUSED:
            return xs.reshape((-1, self._dom_size)) @ self._A2T
        if self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            lead = tuple(xs.shape[: len(xs.shape) - len(self.dom.shape)])
            xs2 = xs.reshape((-1, self._dom_size))
            ys2 = xs2 @ self._A2T
            return ys2.reshape(lead + tuple(self.cod.shape))
        xs_flat = self.domain._flatten_batch_core(xs)
        ys_flat = xs_flat @ self._A2T
        return self.codomain._unflatten_batch_core(ys_flat)

    def _rvapply_core(self, ys: Any) -> Any:
        # Batched form of _rapply_core.
        if self._mode is _DenseMode.EUCLIDEAN_FLAT or self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            return self._euclidean_rvapply_core(ys)
        if self._mode is _DenseMode.WEIGHTED_FUSED:
            return ys @ self._weighted_A2H.T
        return metric_rvapply(self.domain, self.codomain, self._euclidean_rvapply_core, ys)

    def _euclidean_rapply_core(self, y: Any) -> Any:
        # Plain conjugate transpose A^H, no metric.
        if self._mode is _DenseMode.EUCLIDEAN_FLAT:
            return self._A2H @ y
        if self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            return (self._A2H @ y.reshape((self._cod_size,))).reshape(self.dom.shape)
        y1 = self.cod._flatten_core(y)
        x1 = self._A2H @ y1
        return self.dom._unflatten_core(x1)

    def _euclidean_rvapply_core(self, ys: Any) -> Any:
        # A batch of rows: Y (A^H)^T.
        if self._mode is _DenseMode.EUCLIDEAN_FLAT:
            return ys.reshape((-1, self._cod_size)) @ self._A2H.T
        if self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            lead = tuple(ys.shape[: len(ys.shape) - len(self.cod.shape)])
            ys2 = ys.reshape((-1, self._cod_size))
            xs2 = ys2 @ self._A2H.T
            return xs2.reshape(lead + tuple(self.dom.shape))
        ys_flat = self.codomain._flatten_batch_core(ys)
        xs_flat = ys_flat @ self._A2H.T
        return self.domain._unflatten_batch_core(xs_flat)

    def __init__(
        self,
        A: DenseArray,
        dom: Domain,
        cod: Codomain | None = None,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        # A must already be a dense array of the resolved backend.
        ctx = resolve_context_priority(ctx, dom, cod)
        ctx.assert_dense(A)  # Check if A is ndarray of ctx

        # Infer a Euclidean codomain from A's leading axes; warn if that is suspect.
        if cod is None:
            cod_shape_len = len(A.shape) - len(dom.shape)
            cod = cast(Codomain, DenseCoordinateSpace(tuple(A.shape[:cod_shape_len]), ctx))
            if not dom.is_euclidean:
                # Only the shape is inferable from ``A``; the geometry is not. On a
                # non-Euclidean domain the Euclidean default is very often not what
                # the caller meant -- ``DenseLinOp(M, X)`` with square ``M`` reads as
                # an endomorphism of ``X`` but is not one, which changes the adjoint
                # and makes ``is_hermitian`` report False for a self-adjoint matrix.
                # Silent when the domain is Euclidean, where the default is right.
                warnings.warn(
                    "DenseLinOp inferred a EUCLIDEAN codomain from the shape of `A`, but "
                    "the domain geometry is non-Euclidean. The operator therefore maps "
                    "X -> (Euclidean space), not X -> X, so its adjoint is "
                    "R_X^-1 A^dagger rather than R_X^-1 A^dagger R_X. Pass `cod=` "
                    "explicitly to state the codomain geometry you intend.",
                    UserWarning,
                    stacklevel=2,
                )

        _requires_euclidean_or_riesz(dom, cod, "DenseLinOp")

        super(DenseLinOp, self).__init__(dom, cod, ctx, check_level=check_level)

        # A's shape must be cod.shape + dom.shape.
        expected = tuple(self.cod.shape) + tuple(self.dom.shape)
        if tuple(A.shape) != expected:
            raise TypeError(
                f"Expected A.shape == cod.shape + dom.shape == {expected}, got {A.shape}"
            )

        # Precompute the 2-D matrix and its (conjugate) transpose once, for the cores.
        self._A = A  # Intentionally no dtype conversion to avoid extra memory use.
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
        # Pick the apply strategy once; weighted spaces get a pre-scaled adjoint.
        self._mode = self._select_mode()
        if self._mode is _DenseMode.WEIGHTED_FUSED:
            # Weights may be stored at any shape that broadcasts to the space's.
            self._dom_weights = self.ops.broadcast_to(self.dom.geometry.weights, tuple(self.dom.shape))
            self._cod_weights = self.ops.broadcast_to(self.cod.geometry.weights, tuple(self.cod.shape))
            self._weighted_A2H = (
                self._A2H * self._cod_weights.reshape((1, self._cod_size))
            ) / self._dom_weights.reshape((self._dom_size, 1))

    def _select_mode(self) -> _DenseMode:
        """Select the dense computation mode once for this operator."""
        # Weighted flat vectors: fused weighted adjoint; Euclidean: plain matmul;
        # anything else: the general Riesz-map path.
        vector_dom = type(self.dom) in _VECTOR_SPACES
        vector_cod = type(self.cod) in _VECTOR_SPACES
        if (
            vector_dom
            and vector_cod
            and self._dom_is_flat
            and self._cod_is_flat
            and type(self.dom.geometry) is WeightedInnerProduct
            and type(self.cod.geometry) is WeightedInnerProduct
        ):
            return _DenseMode.WEIGHTED_FUSED
        if (
            vector_dom
            and vector_cod
            and cast(Any, self.domain).is_euclidean
            and cast(Any, self.codomain).is_euclidean
        ):
            if self._dom_is_flat and self._cod_is_flat:
                return _DenseMode.EUCLIDEAN_FLAT
            return _DenseMode.EUCLIDEAN_TENSOR
        return _DenseMode.GENERAL_METRIC

    @cached_property
    def A(self) -> DenseArray:
        """
        Stored dense tensor representation of this operator.

        The returned array has shape ``self.codomain.shape + self.domain.shape``
        and is the same object supplied at construction.
        """
        return self._A

    rvapply = _output_checked_rvapply

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
        # Hermitian needs dom == cod; Euclidean compares A with A^H, else test by basis.
        if self.dom != self.cod:
            return False
        if not cast(Any, self.domain).is_euclidean:  # dom == cod, so cod is too
            return _metric_is_hermitian_by_basis(self)
        try:
            return bool(self.ops.allclose(self._A2, self._A2H))
        except Exception:
            return None
