from __future__ import annotations

from typing import Any, Tuple

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
        if tuple(getattr(X, "shape", ())) != self.shape:
            raise TypeError(f"Expected shape {self.shape}, got {getattr(X, 'shape', None)}")

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

    def flatten(self, x: Any) -> DenseArray:
        self.check_member(x)
        return x

    def unflatten(self, v: DenseArray) -> DenseArray:
        V = self.ctx.assert_dense(v)
        return self.ops.reshape(V, self.shape)
