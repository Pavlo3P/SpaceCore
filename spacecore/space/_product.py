from __future__ import annotations

from typing import Any, Tuple, List, Sequence, Callable

from ._base import Space
from ..types import DenseArray
from ..backend import Context

from .._contextual.manager import ctx_manager


def _prod_int(shape: Tuple[int, ...]) -> int:
    p = 1
    for d in shape:
        p *= int(d)
    return int(p)


class ProductSpace(Space):
    """
    Cartesian product space X = X1 × ... × Xk.

    Elements are tuples:
        x = (x1, ..., xk) with xi ∈ Xi

    Canonical dense coordinates:
        flatten(x) = concat(flatten_i(xi))

    Notes:
      - `shape` for this space is the *1D coordinate length* of the concatenated flattening.
      - `eigh` has no canonical meaning here and raises by default.
    """

    def _convert(self, new_ctx: Context) -> Space:
        new_spaces = []
        for sp in self.spaces:
            new_spaces.append(sp.convert(new_ctx))
        return ProductSpace(tuple(new_spaces), new_ctx)

    def __init__(self, spaces: Tuple[Space, ...], ctx: Context | str | None = None) -> None:
        if len(spaces) == 0:
            raise ValueError("ProductSpace requires at least one subspace.")

        spaces = self._validate_spaces(spaces)
        ctx = ctx_manager.resolve_context_priority(ctx, *spaces)

        dims = tuple(_prod_int(s.shape) for s in spaces)
        offsets: List[int] = [0]
        for d in dims:
            offsets.append(offsets[-1] + d)

        self._dims = dims
        self._offsets = tuple(offsets)
        shape = (offsets[-1],)

        super(ProductSpace, self).__init__(shape, ctx)
        uniform_spaces = tuple(sp.convert(self.ctx) for sp in spaces)
        self.spaces = uniform_spaces

    def _validate_spaces(self, spaces: Any) -> Tuple[Space, ...]:
        if isinstance(spaces, Sequence):
            spaces = tuple(spaces)
            for i, sp in enumerate(spaces):
                if isinstance(sp, Space):
                    continue
                else:
                    raise TypeError(f"ProductSpace requires a sequence of spaces, got {type(sp)!r} at index {i}.")
            return spaces
        else:
            raise TypeError(f"ProductSpace requires a sequence of spaces, got {type(spaces)!r}.")

    @property
    def arity(self) -> int:
        return len(self.spaces)

    def _check_member(self, x: Sequence[Any]) -> None:
        if not isinstance(x, tuple):
            raise TypeError(f"ProductSpace element must be a tuple, got {type(x).__name__}")
        if len(x) != self.arity:
            raise ValueError(f"Expected tuple of length {self.arity}, got {len(x)}")

        for i, (si, xi) in enumerate(zip(self.spaces, x)):
            try:
                si.check_member(xi)
            except Exception as e:
                raise type(e)(f"Invalid component {i} for spaces[{i}] ({type(si).__name__}): {e}") from e

    def zeros(self) -> Tuple[Any, ...]:
        return tuple(s.zeros() for s in self.spaces)

    def add(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Tuple[Any, ...]:
        self.check_member(x)
        self.check_member(y)
        return tuple(s.add(xi, yi) for s, xi, yi in zip(self.spaces, x, y))

    def scale(self, a: Any, x: Tuple[Any, ...]) -> Tuple[Any, ...]:
        self.check_member(x)
        return tuple(s.scale(a, xi) for s, xi in zip(self.spaces, x))

    def inner(self, x: Tuple[Any, ...], y: Tuple[Any, ...]) -> Any:
        self.check_member(x)
        self.check_member(y)

        # Accumulate via backend ops (vdot works for scalars too, but sum is enough)
        acc = None
        for s, xi, yi in zip(self.spaces, x, y):
            v = s.inner(xi, yi)
            acc = v if acc is None else (acc + v)
        return acc

    def eigh(self, x: Any, k: int = None) -> Any:
        raise NotImplementedError(
            "ProductSpace.eigh is not defined. "
            "Call eigh on a specific component space, or define a custom convention."
        )

    def flatten(self, x: Tuple[Any, ...]) -> DenseArray:
        self.check_member(x)

        parts = []
        for s, xi in zip(self.spaces, x):
            vi = s.flatten(xi)
            vi = self.ctx.assert_dense(vi)
            parts.append(self.ctx.ops.ravel(vi))

        if len(parts) == 1:
            return parts[0]

        return self.ctx.ops.concatenate(parts, axis=0)

    def unflatten(self, v: DenseArray) -> Tuple[Any, ...]:
        v = self.ctx.assert_dense(v)
        v1 = self.ctx.ops.ravel(v)

        xs: List[Any] = []
        for i, s in enumerate(self.spaces):
            a = self._offsets[i]
            b = self._offsets[i + 1]
            vi = v1[a:b]
            xs.append(s.unflatten(vi))

        return tuple(xs)

    def apply(self, x: Tuple[Any, ...], f: Callable[[Any], Any]) -> Tuple[Any, ...]:
        self.check_member(x)
        return tuple(s.apply(xi, f) for s, xi in zip(self.spaces, x))
