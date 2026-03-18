from dataclasses import dataclass
from typing import Any

from ._ops import BackendOps
from ..types import DenseArray, SparseArray, DType, ArrayLike


@dataclass(frozen=True, slots=True)
class Context:
    ops: BackendOps
    dtype: DType | None = None
    enable_checks: bool = True

    def __post_init__(self):
        if not isinstance(self.ops, BackendOps):
            raise TypeError("ops must be a BackendOps")

        sanitized = self.ops.sanitize_dtype(self.dtype)
        object.__setattr__(self, "dtype", sanitized)

    def assert_dense(self, x: Any) -> DenseArray:
        if self.enable_checks:
            if not self.ops.is_dense(x):
                raise TypeError(f"Expected dense array for {self.ops.family}, got {type(x).__name__}")
            return x
        else:
            return x

    def assert_sparse(self, x: Any) -> SparseArray:
        if self.enable_checks:
            if not self.ops.allow_sparse:
                raise TypeError("Sparse objects are disallowed by this backend.")
            if not self.ops.is_sparse(x):
                raise TypeError(f"Expected sparse array for {self.ops.family}, got {type(x).__name__}")
            return x
        else:
            return x

    def asarray(self, x: Any) -> DenseArray:
        return self.ops.asarray(x, dtype=self.dtype)

    def assparse(self, x: Any) -> SparseArray:
        return self.ops.assparse(x, dtype=self.dtype)

    def convert(self, x: Any) -> ArrayLike:
        if self.ops.is_dense(x):
            return self.asarray(x)
        elif self.ops.is_sparse(x):
            return self.assparse(x)
        else:
            raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Context):
            return self.ops == other.ops
        return False
