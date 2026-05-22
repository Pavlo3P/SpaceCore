from __future__ import annotations

from math import prod
from typing import Any, Callable, Tuple

from ._base import Space
from ._checks import BackendCheck, DTypeCheck, ShapeCheck
from ._product import ProductSpace
from ..backend import Context
from ..types import DenseArray


def _batched_shape(
    base_shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
    batch_axes: tuple[int, ...],
) -> tuple[int, ...]:
    total_ndim = len(base_shape) + len(batch_shape)
    axes = tuple(axis + total_ndim if axis < 0 else axis for axis in batch_axes)
    if len(batch_shape) != len(axes):
        raise ValueError("batch_shape and batch_axes must have the same length.")
    if len(set(axes)) != len(axes):
        raise ValueError("batch_axes must be unique.")
    if any(axis < 0 or axis >= total_ndim for axis in axes):
        raise ValueError(
            f"batch_axes must be valid axes for batched ndim {total_ndim}, got {batch_axes}."
        )

    out: list[int | None] = [None] * total_ndim
    for axis, dim in zip(axes, batch_shape):
        out[axis] = int(dim)

    base_iter = iter(int(dim) for dim in base_shape)
    for i, dim in enumerate(out):
        if dim is None:
            out[i] = next(base_iter)
    return tuple(dim for dim in out if dim is not None)


class BatchSpace(Space):
    """
    Wrapper space representing a batch of elements from a base space.

    ``BatchSpace(X, batch_shape, batch_axes)`` represents ``X`` repeated over
    the given batch dimensions. It deliberately wraps the original space rather
    than folding batch dimensions into the base ``Space`` instance.
    """

    def __init__(
        self,
        base: Space,
        batch_shape: Tuple[int, ...],
        batch_axes: Tuple[int, ...],
        ctx: Context | str | None = None,
    ) -> None:
        ctx = base.ctx if ctx is None else ctx
        super().__init__(
            _batched_shape(tuple(base.shape), tuple(batch_shape), tuple(batch_axes)),
            ctx,
        )
        self.base = base.convert(self.ctx)
        self.batch_shape = tuple(int(dim) for dim in batch_shape)
        total_ndim = len(self.base.shape) + len(self.batch_shape)
        self.batch_axes = tuple(axis + total_ndim if axis < 0 else axis for axis in batch_axes)
        self._batch_size = prod(self.batch_shape)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BatchSpace):
            return (
                self.ctx == other.ctx
                and self.base == other.base
                and self.batch_shape == other.batch_shape
                and self.batch_axes == other.batch_axes
            )
        return False

    @property
    def _is_product(self) -> bool:
        return isinstance(self.base, ProductSpace)

    def _component_spaces(self) -> tuple[BatchSpace, ...]:
        if not isinstance(self.base, ProductSpace):
            raise TypeError("BatchSpace component spaces are available only for ProductSpace bases.")
        return tuple(sp.batch(self.batch_shape, self.batch_axes) for sp in self.base.spaces)

    def _check_member(self, x: Any) -> None:
        if isinstance(self.base, ProductSpace):
            if not isinstance(x, tuple) or len(x) != self.base.arity:
                raise TypeError(
                    f"BatchSpace over ProductSpace expects tuple length {self.base.arity}."
                )
            for space, component in zip(self._component_spaces(), x):
                space.check_member(component)
            return

        BackendCheck()(self, x)
        ShapeCheck()(self, x)
        DTypeCheck()(self, x)
        for check in self.base.member_checks():
            if isinstance(check, (BackendCheck, ShapeCheck, DTypeCheck)):
                continue
            check(self, x)

    def zeros(self) -> Any:
        if isinstance(self.base, ProductSpace):
            return tuple(space.zeros() for space in self._component_spaces())
        return self.ops.zeros(self.shape, dtype=self.dtype)

    def add(self, x: Any, y: Any) -> Any:
        if self._enable_checks:
            self._check_member(x)
            self._check_member(y)
        if isinstance(self.base, ProductSpace):
            return tuple(space.add(xi, yi) for space, xi, yi in zip(self._component_spaces(), x, y))
        return x + y

    def scale(self, a: Any, x: Any) -> Any:
        if self._enable_checks:
            self._check_member(x)
        if isinstance(self.base, ProductSpace):
            return tuple(space.scale(a, xi) for space, xi in zip(self._component_spaces(), x))
        return a * x

    def inner(self, x: Any, y: Any) -> Any:
        if self._enable_checks:
            self._check_member(x)
            self._check_member(y)
        if isinstance(self.base, ProductSpace):
            acc = None
            for space, xi, yi in zip(self._component_spaces(), x, y):
                v = space.inner(xi, yi)
                acc = v if acc is None else acc + v
            return acc
        return self.ops.vdot(x, y)

    def eigh(self, x: Any, k: int = None) -> Any:
        raise TypeError(f"{type(self).__name__}.eigh is not defined for batched spaces.")

    def flatten(self, x: Any) -> DenseArray:
        if self._enable_checks:
            self._check_member(x)
        if isinstance(self.base, ProductSpace):
            parts = tuple(space.flatten(xi) for space, xi in zip(self._component_spaces(), x))
            return parts[0] if len(parts) == 1 else self.ops.concatenate(parts, axis=0)
        return self.ops.reshape(x, (-1,))

    def unflatten(self, v: DenseArray) -> Any:
        vv = self.ctx.assert_dense(v) if self._enable_checks else v
        if isinstance(self.base, ProductSpace):
            if (
                tuple(getattr(vv, "shape", ())) == tuple(self.shape)
                and self.batch_axes == tuple(range(len(self.batch_shape)))
            ):
                xs = []
                offset = 0
                for component, space in zip(self.base.spaces, self._component_spaces()):
                    size = prod(component.shape)
                    flat_component = vv[(..., slice(offset, offset + size))]
                    xs.append(space.unflatten(flat_component))
                    offset += size
                return tuple(xs)
            xs = []
            offset = 0
            for space in self._component_spaces():
                size = prod(space.shape)
                xs.append(space.unflatten(vv[offset : offset + size]))
                offset += size
            return tuple(xs)
        return self.ops.reshape(vv, self.shape)

    def apply(self, x: Any, f: Callable) -> Any:
        if self._enable_checks:
            self._check_member(x)
        if isinstance(self.base, ProductSpace):
            return tuple(space.apply(xi, f) for space, xi in zip(self._component_spaces(), x))
        try:
            y = f(x)
        except Exception:
            y = self.ops.vmap(lambda xi: self.base.apply(xi, f))(x)
        if self._enable_checks:
            self._check_member(y)
        return y

    def _convert(self, new_ctx: Context) -> BatchSpace:
        return BatchSpace(self.base.convert(new_ctx), self.batch_shape, self.batch_axes, new_ctx)
