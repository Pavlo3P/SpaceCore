from __future__ import annotations

from math import prod
from typing import Any, Tuple, Callable

from ._base import Space
from ._checks import BackendCheck, DTypeCheck, ShapeCheck
from .._checks import checked_method
from ..types import DenseArray
from ..backend import Context


class VectorSpace(Space):
    """
    Dense vector space R^{n1, ..., nK} or C^{n1, ..., nK}.

    Elements:
      - backend-native dense arrays;
      - canonical shape is (n1, ..., nK).

    Geometry:
      - Euclidean / ℓ2 inner product
            ⟨x, y⟩ = vdot(x, y).
    """

    def __init__(self, shape: Tuple[int, ...], ctx: Context | str | None = None) -> None:
        super(VectorSpace, self).__init__(shape, ctx)
        self._size = prod(self.shape)
        self._is_flat_shape = self.shape == (self._size,)

    def _local_checks(self):
        return BackendCheck(), ShapeCheck(), DTypeCheck()

    def zeros(self) -> DenseArray:
        return self.ops.zeros(self.shape, dtype=self.dtype)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def add(self, x: Any, y: Any) -> DenseArray:
        return x + y

    @checked_method(in_space="self", arg_positions=(1,))
    def scale(self, a: Any, x: Any) -> DenseArray:
        return a * x

    @checked_method(in_space="self", arg_positions=(0, 1))
    def inner(self, x: Any, y: Any) -> Any:
        return self.ops.vdot(x, y)

    def eigh(self, x: Any, k: int = None) -> Any:
        raise TypeError(
            f"{type(self).__name__}.eigh is not defined for vector spaces."
        )

    @checked_method(in_space="self")
    def flatten(self, X: DenseArray) -> DenseArray:
        return X if self._is_flat_shape else X.reshape((-1,))

    def unflatten(self, v: DenseArray) -> DenseArray:
        V = self.ctx.assert_dense(v) if self._enable_checks else v
        return V if self._is_flat_shape else V.reshape(self.shape)

    def _convert(self, new_ctx: Context) -> VectorSpace:
        return VectorSpace(self.shape, new_ctx)

    def _apply_entrywise(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
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
