from __future__ import annotations

from ._array_like import ArrayLike
from typing import Protocol, runtime_checkable, Any, Tuple


@runtime_checkable
class SparseArray(ArrayLike):
    def reshape(self, shape: Tuple[int, ...]) -> SparseArray: ...
