from __future__ import annotations

from typing import Any, Callable, Literal, Tuple

from ..base import (
    EuclideanInnerProduct,
    EuclideanJordanAlgebraSpace,
    InnerProduct,
    JordanAlgebraSpace,
    StarSpace,
)
from ..._check_policy import require_mutually_exclusive
from ..._checks import checked_method
from ..._contextual import normalize_context
from ...backend import Context, jax_pytree_class
from ...types import DenseArray
from ._dense_coordinate import DenseCoordinateSpace


def _resolve_elementwise_geometry(
    geometry: InnerProduct | None,
    inner_product: InnerProduct | None = None,
) -> InnerProduct:
    """Resolve the public geometry aliases for elementwise coordinate spaces."""
    require_mutually_exclusive(
        "geometry", geometry, "inner_product", inner_product, verb="Specify"
    )
    if inner_product is not None:
        return inner_product
    if geometry is not None:
        return geometry
    return EuclideanInnerProduct()


def _euclidean_elementwise_jordan_incompatibility(
    field: Literal["real", "complex"],
    geometry: InnerProduct,
) -> str | None:
    """Return the violated Euclidean-elementwise invariant, if any."""
    if field != "real":
        return "field"
    if type(geometry) is not EuclideanInnerProduct:
        return "geometry"
    return None


def _is_euclidean_elementwise_jordan_compatible(ctx: Context, geometry: InnerProduct) -> bool:
    """Return whether elementwise coordinates have Euclidean-Jordan geometry."""
    field = "complex" if ctx.ops.is_complex_dtype(ctx.dtype) else "real"
    return _euclidean_elementwise_jordan_incompatibility(field, geometry) is None


def _validate_euclidean_elementwise_jordan(
    space: DenseCoordinateSpace,
    geometry: InnerProduct,
) -> None:
    """Raise if a Euclidean elementwise Jordan space would be untruthful."""
    reason = _euclidean_elementwise_jordan_incompatibility(space.field, geometry)
    if reason == "field":
        raise ValueError("EuclideanElementwiseJordanSpace requires a real scalar field.")
    if reason == "geometry":
        raise TypeError("EuclideanElementwiseJordanSpace requires EuclideanInnerProduct.")


class DenseVectorSpace(DenseCoordinateSpace, StarSpace):
    """
    Plain one-dimensional dense vectors with no Jordan capability by default.

    Parameters
    ----------
    shape : tuple of int
        One-dimensional dense vector shape.
    ctx : Context, str, or None, optional
        Context specification used for dense arrays.
    geometry : InnerProduct or None, optional
        Inner-product geometry. If omitted, Euclidean coordinate geometry is
        used.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
    ) -> None:
        shape = tuple(shape)
        if len(shape) != 1:
            raise ValueError(f"DenseVectorSpace requires one-dimensional shape, got {shape}.")
        super().__init__(shape, ctx, geometry=geometry)

    @checked_method(in_space="self")
    def star(self, x: DenseArray) -> DenseArray:
        """Return elementwise conjugation."""
        return self.ops.conj(x)

    def _convert(self, new_ctx: Context) -> DenseVectorSpace:
        return DenseVectorSpace(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))


@jax_pytree_class
class ElementwiseJordanSpace(JordanAlgebraSpace, DenseCoordinateSpace, StarSpace):
    """
    Elementwise Jordan algebra for real or complex dense coordinates.

    Parameters
    ----------
    shape : tuple of int
        Canonical dense array shape for one element.
    ctx : Context, str, or None, optional
        Context specification used for dense arrays.
    geometry : InnerProduct or None, optional
        Inner-product geometry. If omitted, Euclidean coordinate geometry is
        used.
    inner_product : InnerProduct or None, optional
        Alias for ``geometry``.
    """

    def __new__(
        cls,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
        *,
        inner_product: InnerProduct | None = None,
    ):
        geometry = _resolve_elementwise_geometry(geometry, inner_product)
        if cls is ElementwiseJordanSpace:
            resolved_ctx = normalize_context(ctx)
            if _is_euclidean_elementwise_jordan_compatible(resolved_ctx, geometry):
                cls = EuclideanElementwiseJordanSpace
        return super(ElementwiseJordanSpace, cls).__new__(cls)

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
        *,
        inner_product: InnerProduct | None = None,
    ) -> None:
        geometry = _resolve_elementwise_geometry(geometry, inner_product)
        DenseCoordinateSpace.__init__(self, shape, ctx, geometry=geometry)

    @checked_method(in_space="self")
    def star(self, x: DenseArray) -> DenseArray:
        """Return elementwise conjugation."""
        return self.ops.conj(x)

    @checked_method(in_space="self", arg_positions=(0, 1))
    def jordan(self, x: DenseArray, y: DenseArray) -> DenseArray:
        """Return the elementwise Jordan product."""
        return x * y

    def spectrum(self, x: DenseArray) -> DenseArray:
        """Return ``x`` as its Jordan spectrum under elementwise product."""
        self._check_unbatched_member(x)
        return x

    def spectral_decompose(self, x: DenseArray) -> tuple[DenseArray, None]:
        """Return the trivial spectral decomposition ``(x, None)``."""
        self._check_unbatched_member(x)
        return x, None

    def from_spectrum(self, eigvals: DenseArray, frame: Any = None) -> DenseArray:
        """Reconstruct an elementwise Jordan element from its spectrum."""
        if frame is not None:
            raise ValueError(f"{type(self).__name__}.from_spectrum expects frame=None.")
        self._check_unbatched_member(eigvals)
        return eigvals

    def trace(self, x: DenseArray) -> DenseArray:
        """Return the Jordan trace as the sum over the element's own coordinates."""
        self._check_unbatched_member(x)
        return self.ops.sum(x, axis=tuple(range(-len(self.shape), 0)))

    def determinant(self, x: DenseArray) -> DenseArray:
        """Return the Jordan determinant as the product over the coordinates."""
        self._check_unbatched_member(x)
        return self.ops.prod(x, axis=tuple(range(-len(self.shape), 0)))

    def unit(self) -> DenseArray:
        """Return the Jordan identity: the all-ones element in the space dtype."""
        return self.ops.ones(self.shape, dtype=self.dtype)

    def _apply_entrywise(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Apply ``f`` entrywise and verify that shape is preserved."""
        try:
            y = f(x)
        except Exception:
            y = self.ops.vectorize(f)(x)
        if self._checks_at_least("cheap") and y.shape != x.shape:
            raise ValueError("Function application changed shape.")
        return y

    @checked_method(in_space="self", out_space="self")
    def spectral_apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        """Apply a scalar function coordinatewise."""
        return self._apply_entrywise(x, f)

    def _convert(self, new_ctx: Context) -> ElementwiseJordanSpace:
        return ElementwiseJordanSpace(self.shape, new_ctx, geometry=self.geometry.convert(new_ctx))

    def tree_flatten(self):
        """Flatten this space for JAX pytree registration."""
        return (), (self.shape, self.ctx, self.geometry)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this space from pytree aux data."""
        shape, ctx, geometry = aux
        return cls(shape, ctx, geometry=geometry)


@jax_pytree_class
class EuclideanElementwiseJordanSpace(ElementwiseJordanSpace, EuclideanJordanAlgebraSpace):
    """
    Real elementwise Euclidean Jordan algebra.

    Parameters
    ----------
    shape : tuple of int
        Canonical dense array shape for one element.
    ctx : Context, str, or None, optional
        Context specification used for dense arrays.
    geometry : InnerProduct or None, optional
        Inner-product geometry. This class is selected only for real contexts
        with Euclidean coordinate geometry.
    inner_product : InnerProduct or None, optional
        Alias for ``geometry``.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        geometry: InnerProduct | None = None,
        *,
        inner_product: InnerProduct | None = None,
    ) -> None:
        resolved_ctx = normalize_context(ctx)
        geometry = _resolve_elementwise_geometry(geometry, inner_product)
        DenseCoordinateSpace.__init__(self, shape, resolved_ctx, geometry=geometry)
        _validate_euclidean_elementwise_jordan(self, self.geometry)
