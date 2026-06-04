from __future__ import annotations

from enum import Enum, auto
from functools import cached_property
from math import prod
from typing import Any

from ._base import Codomain, Domain, LinOp
from .._checks import checked_method
from ._metric import (
    _metric_is_hermitian_by_basis,
    _requires_euclidean_or_riesz,
    metric_rapply,
    metric_rvapply,
)
from ..space import DenseCoordinateSpace, DenseVectorSpace, WeightedInnerProduct
from ..types import DenseArray
from ..backend import jax_pytree_class, Context
from .._contextual import resolve_context_priority


class _DenseMode(Enum):
    """Private computation modes for dense coordinate operators."""

    EUCLIDEAN_FLAT = auto()
    EUCLIDEAN_TENSOR = auto()
    WEIGHTED_FUSED = auto()
    GENERAL_METRIC = auto()


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
    >>> X = sc.DenseCoordinateSpace((2,), ctx)
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
            cod = DenseCoordinateSpace(A.shape[:cod_shape_len], ctx)

        _requires_euclidean_or_riesz(dom, cod, "DenseLinOp")

        super(DenseLinOp, self).__init__(dom, cod, ctx)

        expected = tuple(self.cod.shape) + tuple(self.dom.shape)
        if tuple(A.shape) != expected:
            raise TypeError(f"Expected A.shape == cod.shape + dom.shape == {expected}, got {A.shape}")

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
        self._mode = self._select_mode()
        if self._mode is _DenseMode.WEIGHTED_FUSED:
            self._dom_weights = self.dom.geometry.weights
            self._cod_weights = self.cod.geometry.weights
            self._weighted_A2H = (
                self._A2H * self._cod_weights.reshape((1, self._cod_size))
            ) / self._dom_weights.reshape((self._dom_size, 1))

    def _select_mode(self) -> _DenseMode:
        """Select the dense computation mode once for this operator."""
        vector_dom = type(self.dom) is DenseCoordinateSpace or type(self.dom) is DenseVectorSpace
        vector_cod = type(self.cod) is DenseCoordinateSpace or type(self.cod) is DenseVectorSpace
        if (
            vector_dom
            and vector_cod
            and self._dom_is_flat
            and self._cod_is_flat
            and type(self.dom.geometry) is WeightedInnerProduct
            and type(self.cod.geometry) is WeightedInnerProduct
        ):
            return _DenseMode.WEIGHTED_FUSED
        if vector_dom and vector_cod and self.domain.is_euclidean and self.codomain.is_euclidean:
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

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: DenseArray) -> DenseArray:
        """Apply the dense operator to ``x``."""
        return self._apply_core(x)

    def _apply_core(self, x: DenseArray) -> DenseArray:
        """Apply the flattened dense matrix without membership checks."""
        if self._mode is _DenseMode.EUCLIDEAN_FLAT:
            return self._A2 @ x
        if self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            return (self._A2 @ x.reshape((self._dom_size,))).reshape(self.cod.shape)
        if self._mode is _DenseMode.WEIGHTED_FUSED:
            return self._A2 @ x
        x1 = self.dom.flatten(x)
        y1 = self._A2 @ x1
        return self.cod.unflatten(y1)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: DenseArray) -> DenseArray:
        r"""Apply the adjoint dense operator to ``y``.

        Euclidean spaces use the conjugate transpose of the flattened matrix.
        Non-Euclidean spaces apply the codomain and domain Riesz maps around
        that Euclidean adjoint.
        """
        return self._rapply_core(y)

    def _rapply_core(self, y: DenseArray) -> DenseArray:
        """Apply the metric adjoint without membership checks."""
        if self._mode is _DenseMode.EUCLIDEAN_FLAT or self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            return self._euclidean_rapply_core(y)
        if self._mode is _DenseMode.WEIGHTED_FUSED:
            return self._weighted_A2H @ y
        return metric_rapply(self.domain, self.codomain, self._euclidean_rapply_core, y)

    def _euclidean_rapply_core(self, y: DenseArray) -> DenseArray:
        """Apply the flattened adjoint matrix without membership checks."""
        if self._mode is _DenseMode.EUCLIDEAN_FLAT:
            return self._A2H @ y
        if self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            return (self._A2H @ y.reshape((self._cod_size,))).reshape(self.dom.shape)
        y1 = self.cod.flatten(y)
        x1 = self._A2H @ y1
        return self.dom.unflatten(x1)

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        return self._vapply_core(xs)

    def _vapply_core(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis without membership checks."""
        if self._mode is _DenseMode.EUCLIDEAN_FLAT:
            return xs.reshape((-1, self._dom_size)) @ self._A2T
        if self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            lead = tuple(xs.shape[: len(xs.shape) - len(self.dom.shape)])
            xs2 = xs.reshape((-1, self._dom_size))
            ys2 = xs2 @ self._A2T
            return ys2.reshape(lead + tuple(self.cod.shape))
        if self._mode is _DenseMode.WEIGHTED_FUSED:
            return xs.reshape((-1, self._dom_size)) @ self._A2T
        xs_flat = self.domain.flatten_batch(xs)
        ys_flat = xs_flat @ self._A2T
        return self.codomain.unflatten_batch(ys_flat)

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, ys: DenseArray) -> DenseArray:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
        return self._rvapply_core(ys)

    def _rvapply_core(self, ys: DenseArray) -> DenseArray:
        """Apply the metric adjoint over leading batch axes without checks."""
        if self._mode is _DenseMode.EUCLIDEAN_FLAT or self._mode is _DenseMode.EUCLIDEAN_TENSOR:
            return self._euclidean_rvapply_core(ys)
        if self._mode is _DenseMode.WEIGHTED_FUSED:
            return ys @ self._weighted_A2H.T
        return metric_rvapply(
            self.domain,
            self.codomain,
            self._euclidean_rapply_core,
            self._euclidean_rvapply_core,
            ys,
            opname=type(self).__name__,
            ops=self.ops,
        )

    def _euclidean_rvapply_core(self, ys: DenseArray) -> DenseArray:
        """Apply the Euclidean adjoint over leading batch axes."""
        if self._mode is _DenseMode.EUCLIDEAN_FLAT:
            return ys.reshape((-1, self._cod_size)) @ self._A2H.T
        if self._mode is _DenseMode.EUCLIDEAN_TENSOR:
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
        aux = (self.dom, self.cod, self.ctx, self._mode)
        children = (self.A,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        if len(aux) == 4:
            dom, cod, ctx, _mode = aux
        else:
            dom, cod, ctx = aux
        A = children[0]
        return cls(A, dom, cod, ctx)

    def _convert(self, new_ctx: Context) -> DenseLinOp:
        """Convert spaces and stored dense tensor to ``new_ctx``."""
        new_dom = self.dom.convert(new_ctx)
        new_cod = self.cod.convert(new_ctx)
        new_A = new_ctx.asarray(self.A)
        return DenseLinOp(new_A, new_dom, new_cod, new_ctx)
