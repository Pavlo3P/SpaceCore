from __future__ import annotations

from typing import Any, Protocol, Sequence, Self, TypeAlias, runtime_checkable

from ._dtype import DType

Shape: TypeAlias = Sequence[int]


@runtime_checkable
class ArrayLike(Protocol):
    """
    Define the minimal array-like object accepted by backend helpers.

    This intentionally only models common metadata. NumPy arrays, JAX arrays,
    PyTorch tensors, sparse arrays, scalar-like backend arrays, and array
    wrappers can satisfy this without implementing every dense-array method.

    Parameters
    ----------
    *args : Any
        Construction arguments accepted by concrete array implementations.
    **kwargs : Any
        Keyword construction arguments accepted by concrete array
        implementations.
    """

    @property
    def shape(self) -> Shape: ...

    @property
    def dtype(self) -> DType: ...


class SparseArray(ArrayLike, Protocol):
    """
    Define the portable sparse-array surface used by sparse operators.

    Backend-specific sparse APIs such as SciPy ``tocsr()``, JAX sparse
    ``indices``/``data``, and Torch ``to_dense()`` are intentionally not part
    of this protocol. Concrete backends may use those after checking that the
    object belongs to their sparse family.

    Parameters
    ----------
    *args : Any
        Construction arguments accepted by concrete sparse array
        implementations.
    **kwargs : Any
        Keyword construction arguments accepted by concrete sparse array
        implementations.
    """

    @property
    def T(self) -> Self: ...

    def conj(self) -> Self: ...
    def reshape(self, *shape: Any, **kwargs: Any) -> Self: ...
    def __matmul__(self, other: Any) -> Any: ...


class DenseArray(ArrayLike, Protocol):
    """
    Define the portable dense-array surface used by core abstractions.

    The protocol includes only operations that SpaceCore core abstractions use
    directly on dense arrays. Backend-specific metadata such as device,
    sharding, layout, strides, and gradient state belongs to concrete backend
    implementations, not to this portable type.

    Parameters
    ----------
    *args : Any
        Construction arguments accepted by concrete dense array implementations.
    **kwargs : Any
        Keyword construction arguments accepted by concrete dense array
        implementations.
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
