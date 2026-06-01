from __future__ import annotations

from math import prod
from typing import Any, Tuple, Callable

from ._base import Space
from ._checks import BackendCheck, DTypeCheck, ShapeCheck
from ._inner import InnerProduct
from .._checks import checked_method
from ..types import DenseArray
from ..backend import Context


class VectorSpace(Space):
    r"""
    Represent dense backend arrays with configurable inner-product geometry.

    Elements are backend-native dense arrays with canonical shape ``shape``.
    By default, the geometry is Euclidean:
    :math:`\langle x, y\rangle_X = \operatorname{vdot}(x,y)`, where the
    backend conjugates the first argument for complex arrays. A custom
    :class:`InnerProduct` may be supplied with ``geometry=...`` to define the
    inner product and Riesz maps used by metric-aware adjoints.

    The methods :meth:`inner`, :meth:`riesz`, and :meth:`riesz_inverse`
    delegate to the geometry object.

    Parameters
    ----------
    shape : tuple of int
        Canonical coordinate shape for elements of the space.
    ctx : Context, str, or None, optional
        Backend context specification. Default resolves to the global context.
    geometry : InnerProduct or None, optional
        Inner-product geometry for this coordinate space. Defaults to
        Euclidean coordinate geometry.

    Attributes
    ----------
    shape : tuple of int
        Canonical element shape.
    ctx : Context
        Resolved backend context.

    Examples
    --------
    >>> import numpy as np
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.VectorSpace((2,), ctx)
    >>> x = ctx.asarray([1.0, 2.0])
    >>> X.inner(x, x)
    np.float64(5.0)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ) -> None:
        super(VectorSpace, self).__init__(shape, ctx, geometry=geometry)
        self._size = prod(self.shape)
        self._is_flat_shape = self.shape == (self._size,)

    def _local_checks(self):
        """Return membership checks local to dense vector spaces."""
        return BackendCheck(), ShapeCheck(), DTypeCheck()

    def zeros(self) -> DenseArray:
        """Return the zero vector in this space."""
        return self.ops.zeros(self.shape, dtype=self.dtype)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: Any, y: Any) -> DenseArray:
        """Return the vector-space sum ``x + y``."""
        return x + y

    def add_batch(self, x: Any, y: Any) -> DenseArray:
        """Return the leading-axis batch sum ``x + y``."""
        return x + y

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: Any) -> DenseArray:
        """Return the scalar product ``a * x``."""
        return a * x

    def scale_batch(self, a: Any, x: Any) -> DenseArray:
        """Return the leading-axis batch scalar product ``a * x``."""
        return a * x

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: Any, y: Any) -> Any:
        r"""Return :math:`\langle x, y\rangle_X` using this space's geometry."""
        return self.geometry.inner(self.ops, x, y)

    def eigh(self, x: Any, k: int = None) -> Any:
        """Raise because vector elements do not have a canonical eigendecomposition."""
        raise TypeError(
            f"{type(self).__name__}.eigh is not defined for vector spaces."
        )

    @checked_method(in_space="self")
    def flatten(self, X: DenseArray) -> DenseArray:
        """Return ``X`` as a dense one-dimensional coordinate vector."""
        return X if self._is_flat_shape else X.reshape((-1,))

    def unflatten(self, v: DenseArray) -> DenseArray:
        """Reshape a flat coordinate vector into this space's canonical shape."""
        V = self.ctx.assert_dense(v) if self._enable_checks else v
        return V if self._is_flat_shape else V.reshape(self.shape)

    def flatten_batch(self, xs: DenseArray) -> DenseArray:
        """Flatten a leading-axis batch of dense elements to ``(N, size)``."""
        xs = self.ctx.assert_dense(xs) if self._enable_checks else xs
        return xs if self._is_flat_shape else xs.reshape((xs.shape[0], -1))

    def unflatten_batch(self, vs: DenseArray) -> DenseArray:
        """Unflatten rows of shape ``(N, size)`` into dense space elements."""
        vs = self.ctx.assert_dense(vs) if self._enable_checks else vs
        return vs if self._is_flat_shape else vs.reshape((vs.shape[0],) + self.shape)

    def _convert(self, new_ctx: Context) -> VectorSpace:
        """Convert this vector space to ``new_ctx`` without changing shape."""
        return VectorSpace(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))

    def _apply_entrywise(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Apply ``f`` entrywise and verify that shape is preserved."""
        try:
            y = f(x)
        except Exception:
            # optional fallback if backend has vectorize/map
            y = self.ops.vectorize(f)(x)
        if self._enable_checks:
            if y.shape != x.shape:
                raise ValueError("Function application changed shape.")
        return y

    @checked_method(in_space="self", out_space="self")
    def apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        r"""
        Apply a scalar function to a vector-space element entrywise.

        For a space element
        $$
        x \in \mathbb{K}^{n_1 \times \cdots \times n_k},
        $$
        this method returns the element
        $$
        y = f(x)
        $$
        obtained by applying ``f`` coordinatewise to the entries of ``x``.

        Parameters
        ----------
        x:
            Element of this vector space. Must have shape ``self.shape`` and
            dtype compatible with this space.
        f:
            Callable representing an entrywise transformation. It is expected
            to act elementwise on backend arrays, or to be compatible with the
            backend vectorization fallback.

        Returns
        -------
        DenseArray
            The transformed element, with the same shape as ``x``.

        Raises
        ------
        TypeError
            If ``x`` is not a valid member of this space.
        ValueError
            If the result of the application does not preserve the shape of the
            space element.

        Notes
        -----
        This is the canonical functional calculus for ``VectorSpace``:
        application is performed entrywise in the distinguished coordinate
        representation.
        """
        y = self._apply_entrywise(x, f)
        return y
