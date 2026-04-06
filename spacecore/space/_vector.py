from __future__ import annotations

from typing import Any, Tuple, Callable

from ._base import Space
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

    def _check_member(self, x: Any) -> None:
        X = self.ctx.assert_dense(x)
        maybe_shape = tuple(getattr(X, "shape", ()))
        if maybe_shape != self.shape:
            raise TypeError(f"Expected shape {self.shape}, got {maybe_shape}")
        maybe_dtype = getattr(X, "dtype", None)
        if maybe_dtype != self.dtype:
            raise TypeError(f"Expected dtype {self.dtype}, got {maybe_dtype}")

    def zeros(self) -> DenseArray:
        return self.ops.zeros(self.shape, dtype=self.dtype)

    def add(self, x: Any, y: Any) -> DenseArray:
        self.check_member(x)
        self.check_member(y)
        return x + y

    def scale(self, a: Any, x: Any) -> DenseArray:
        self.check_member(x)
        return a * x

    def inner(self, x: Any, y: Any) -> Any:
        self.check_member(x)
        self.check_member(y)
        return self.ops.vdot(x, y)

    def eigh(self, x: Any, k: int = None) -> Any:
        raise TypeError(
            f"{type(self).__name__}.eigh is not defined for vector spaces."
        )

    def flatten(self, X: DenseArray) -> DenseArray:
        self.check_member(X)
        return self.ops.ravel(X)

    def unflatten(self, v: DenseArray) -> DenseArray:
        V = self.ctx.assert_dense(v)
        return self.ops.reshape(V, self.shape)

    def _convert(self, new_ctx: Context) -> VectorSpace:
        return VectorSpace(self.shape, new_ctx)

    def _apply_entrywise(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        try:
            y = f(x)
        except Exception:
            # optional fallback if backend has vectorize/map
            y = self.ops.vectorize(f)(x)
        return y

    def apply(self, x: DenseArray, f: Callable[[DenseArray], DenseArray]) -> DenseArray:
        self.check_member(x)
        y = self._apply_entrywise(x, f)
        if self.ctx.enable_checks:
            if y.shape != self.shape:
                raise ValueError("Function application changed shape.")
        return y
