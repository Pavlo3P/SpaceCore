"""Consumer typing contracts; checked by Pyright and mypy, never executed."""
from typing import Any, Callable, TypeVar, assert_type

import numpy as np
from numpy.typing import NDArray

import spacecore as sc
from spacecore.types import DenseArray


def portable(x: DenseArray) -> DenseArray:
    return x.reshape((3,))


portable(np.ones(3))
ops = sc.NumpyOps()
x = ops.asarray([1., 2., 3.])
assert_type(x, NDArray[Any])
assert_type(ops.zeros((3,)), NDArray[Any])
assert_type(ops.stack([x, x]), NDArray[Any])
assert_type(ops.concatenate([x, x]), NDArray[Any])
assert_type(ops.sqrt(x), NDArray[Any])
assert_type(ops.conj(x), NDArray[Any])
assert_type(ops.clip(x, 0, 1), NDArray[Any])
assert_type(ops.einsum('i,i->', x, x), NDArray[Any])
assert_type(ops.sum(x), NDArray[Any])
assert_type(ops.index_set(x, 0, 1.), NDArray[Any])
assert_type(ops.where(ops.isfinite(x), x, 0.), NDArray[Any])
assert_type(ops.maximum(0., x), NDArray[Any])
assert_type(ops.eigh(ops.eye(3)), tuple[NDArray[Any], NDArray[Any]])
x.astype(np.float32)

ctx = sc.Context(ops)
assert_type(ctx, sc.Context[NDArray[Any]])
assert_type(ctx.asarray([1., 2.]), NDArray[Any])
assert_type(ctx.assert_dense(x), NDArray[Any])
assert_type(ctx.ops.zeros((3,)), NDArray[Any])
assert_type(sc.Context(sc.NumpyOps).asarray([1.]), NDArray[Any])
assert_type(sc.Context(ctx).asarray([1.]), NDArray[Any])
assert_type(sc.Context('numpy').asarray([1.]), DenseArray)

# Dynamic selection promises the portable surface, with no invented backend.
backends: list[sc.BackendOps] = [ops]
for backend in backends:
    assert_type(backend.zeros((3,)), DenseArray)

space = sc.DenseVectorSpace((3,), ctx)
assert_type(space.flatten(x), DenseArray)
assert_type(space.add(x, x), DenseArray)

# The decorator must preserve generic callable identity, not just return types.
T = TypeVar('T')


class Checked:
    @sc.checked_method()
    def identity(self, value: T) -> T:
        return value

    @sc.checked_method()
    def named(self, value: int, *, factor: int = 1) -> int:
        return value * factor


assert_type(Checked().identity(x), NDArray[Any])
assert_type(Checked().named(1, factor=2), int)
callback: Callable[[int], int] = Checked().named

# Intentional negative cases: unused-ignore checks fail if these degrade to Any.
ops.reshape(x)  # type: ignore
ops.stack([object()])  # type: ignore
portable([1., 2.])  # type: ignore
x.nonexistent_array_method()  # type: ignore
Checked().named('bad')  # type: ignore
Checked().named(1, 2)  # type: ignore
