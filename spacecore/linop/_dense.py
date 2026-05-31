from __future__ import annotations

from functools import cached_property
from math import prod
from typing import Any

from ._base import LinOp
from .._batching import _check_batched
from .._checks import checked_method
from ..space import VectorSpace
from ..types import DenseArray
from ..backend import jax_pytree_class, Context
from .._contextual import resolve_context_priority


@jax_pytree_class
class DenseLinOp(LinOp[VectorSpace, VectorSpace]):
    r"""
    Represent a Euclidean dense tensor-backed linear operator.

    ``DenseLinOp(A, dom, cod)`` represents a linear map
    :math:`A \colon X \to Y` between plain :class:`VectorSpace` instances,
    where the stored dense array has shape ``cod.shape + dom.shape``. Forward
    application contracts over the domain axes; adjoint application uses the
    Euclidean conjugate transpose of the flattened matrix representation.

    Custom spaces with non-Euclidean inner products need a metric-aware
    operator class. ``DenseLinOp`` intentionally does not infer Gram/Riesz
    maps from arbitrary spaces.

    DenseLinOp does not copy or cast the input array. The caller is responsible
    for passing an array compatible with `ctx`. This avoids duplicating large dense
    operators in memory.

    Parameters
    ----------
    A : DenseArray
        Dense backend array with shape ``cod.shape + dom.shape``.
    dom : VectorSpace
        Plain Euclidean domain vector space.
    cod : VectorSpace or None, optional
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
                 dom: VectorSpace,
                 cod: VectorSpace | None = None,
                 ctx: Context | str | None = None
                 ) -> None:
        ctx = resolve_context_priority(ctx, dom, cod)
        ctx.assert_dense(A)  # Check if A is ndarray of ctx

        if cod is None:
            cod_shape_len = len(A.shape) - len(dom.shape)
            cod = VectorSpace(A.shape[:cod_shape_len], ctx)

        if type(dom) is not VectorSpace or type(cod) is not VectorSpace:
            raise TypeError(
                "DenseLinOp supports only plain VectorSpace domain and codomain "
                "with Euclidean inner products. Custom inner-product spaces "
                "require a metric-aware dense operator."
            )

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
        x1 = x if self._dom_is_flat else x.reshape((self._dom_size,))
        y1 = self._A2 @ x1
        return y1 if self._cod_is_flat else y1.reshape(self.cod.shape)

    @checked_method(in_space="cod", out_space="dom")
    def rapply(self, y: DenseArray) -> DenseArray:
        r"""Apply the adjoint dense operator to ``y``.

        The adjoint is the Euclidean conjugate transpose of the flattened
        matrix. This is correct for plain :class:`VectorSpace` domains and
        codomains.
        """
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: DenseArray) -> DenseArray:
        """Apply the flattened adjoint matrix without membership checks."""
        y1 = y if self._cod_is_flat else y.reshape((self._cod_size,))
        x1 = self._A2H @ y1
        return x1 if self._dom_is_flat else x1.reshape(self.dom.shape)

    def vapply(self, xs: DenseArray) -> DenseArray:
        """Apply over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        if self._enable_checks:
            _check_batched(self.domain, xs)
        lead = tuple(xs.shape[: len(xs.shape) - len(self.dom.shape)])
        xs2 = xs.reshape((-1, self._dom_size))
        ys2 = xs2 @ self._A2T
        return ys2.reshape(lead + tuple(self.cod.shape))

    def rvapply(self, ys: DenseArray) -> DenseArray:
        """Apply the adjoint over a leading batch axis. Input must have shape ``(N,) + codomain.shape``; use ``moveaxis`` for other layouts."""
        if self._enable_checks:
            _check_batched(self.codomain, ys)
        lead = tuple(ys.shape[: len(ys.shape) - len(self.cod.shape)])
        ys2 = ys.reshape((-1, self._cod_size))
        xs2 = ys2 @ self._A2H.T
        return xs2.reshape(lead + tuple(self.dom.shape))

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
        Return whether this Euclidean dense operator is structurally Hermitian.

        Returns
        -------
        bool or None
            ``True`` or ``False`` from the Euclidean flattened matrix.
        """
        if self.dom != self.cod:
            return False
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
