from __future__ import annotations

from typing import Any, Protocol, Sequence, Self, TypeAlias, runtime_checkable

from ._dtype import DType

Shape: TypeAlias = Sequence[int]


@runtime_checkable
class ArrayLike(Protocol):
    """
    Minimal array-like protocol: common metadata only, satisfied by dense, sparse and scalar arrays.

    Parameters
    ----------
    *args : Any
        Construction arguments of the concrete type.
    **kwargs : Any
        Keyword construction arguments of the concrete type.
    """

    @property
    def shape(self) -> Shape: ...

    @property
    def dtype(self) -> DType: ...


class SparseArray(ArrayLike, Protocol):
    """
    Portable sparse-array protocol; backend APIs such as ``tocsr()`` are deliberately excluded.

    Parameters
    ----------
    *args : Any
        Construction arguments of the concrete type.
    **kwargs : Any
        Keyword construction arguments of the concrete type.
    """

    @property
    def T(self) -> Self: ...

    def conj(self) -> Self: ...
    def reshape(self, *shape: Any, **kwargs: Any) -> Self: ...
    def __matmul__(self, other: Any) -> Any: ...


class DenseArray(ArrayLike, Protocol):
    """
    Portable dense-array protocol: the operations SpaceCore uses; devices, layout and gradients excluded.

    Parameters
    ----------
    *args : Any
        Construction arguments of the concrete type.
    **kwargs : Any
        Keyword construction arguments of the concrete type.
    """

    @property
    def ndim(self) -> int: ...

    @property
    def T(self) -> Self: ...

    def conj(self) -> Self: ...
    def reshape(self, *shape: Any, **kwargs: Any) -> Self: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any: ...
    def __truediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any: ...
    def __neg__(self) -> Any: ...
    def __matmul__(self, other: Any) -> Any: ...
    def __rmatmul__(self, other: Any) -> Any: ...
