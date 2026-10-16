from __future__ import annotations

from math import prod
from typing import Any, Tuple

from ..._checks import CheckLevel, checked_method
from ...contextual import Context
from ..._repr import shape_descriptor
from ...types import DenseArray
from ._vector import VectorSpace


class CoordinateSpace(VectorSpace):
    """
    Finite coordinate vector space capability.

    Parameters
    ----------
    shape : tuple of int
        Canonical coordinate shape for one element of the space.
    ctx : Context, str, or None, optional
        Context specification used for coordinate arrays.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    shape: Tuple[int, ...]

    def __init__(
        self,
        shape: Tuple[int, ...],
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(ctx, check_level=check_level)
        self.shape = tuple(shape)

    def _eq_algebra(self, other: Any) -> bool:
        # Tier 2: exact shape (after field, before any numerical compare).
        return super()._eq_algebra(other) and self.shape == other.shape

    def _space_descriptor(self) -> str:
        """Return a shape-aware descriptor like ``ℝ^5`` or ``ℝ^(2, 3)``."""
        # repr must never raise: describe the space without a field if it cannot be read.
        try:
            field = self.field
        except Exception:
            field = None
        return shape_descriptor(field, self.shape)

    @property
    def size(self) -> int:
        """Return the flat coordinate dimension of this space."""
        return prod(self.shape)

    # Public coordinate operations: check, then call the core. Families implement
    # ``_flatten_core`` and ``_unflatten_core``; the batched cores default to loops.

    @checked_method(in_space="self")
    def flatten(self, x: Any) -> DenseArray:
        """Return a dense one-dimensional coordinate vector, in **C order**.

        C order (row-major, last axis varying fastest) is SpaceCore's declared
        vec convention on every backend -- see the flattening-order note in
        :mod:`spacecore.backend._ops`. It is what makes ``to_matrix`` and
        ``flatten`` compose: ``to_matrix() @ flatten(x) == flatten(A.apply(x))``
        holds only for a consistent choice, and the Kronecker identities assume
        this one.
        """
        return self._flatten_core(x)

    @checked_method(out_space="self")
    def unflatten(self, v: DenseArray) -> Any:
        """Inverse of flatten, in the same C order."""
        return self._unflatten_core(v)

    @checked_method(in_space="self", in_batched=True)
    def flatten_batch(self, xs: Any) -> DenseArray:
        """Flatten a leading-axis batch of space elements to shape ``(N, size)``."""
        return self._flatten_batch_core(xs)

    @checked_method(out_space="self", out_batched=True)
    def unflatten_batch(self, vs: DenseArray) -> Any:
        """Unflatten rows of shape ``(N, size)`` into a leading-axis batch."""
        return self._unflatten_batch_core(vs)

    @checked_method(in_space="self", arg_positions=(1,))
    def pair(self, xi: Any, x: Any) -> Any:
        r"""Return the metric-free pairing :math:`\langle \xi, x\rangle = \operatorname{Re}\,\mathrm{vdot}(\xi, x)` of a covector with an element.

        A covector is an array of this space's coordinate shape; no inner product is
        involved, so the pairing is the same under every geometry.
        """
        return self._pair_core(xi, x)

    @checked_method(in_space="self", arg_positions=(1,), in_batched=True)
    def vpair(self, xi: Any, xs: Any) -> Any:
        """Return ``pair(xi, xs[i])`` for every element of a leading-axis batch."""
        return self._vpair_core(xi, xs)

    @checked_method(in_space="self", out_space="self", arg_positions=(0, 1), in_batched=True, out_batched=True)
    def add_batch(self, x: Any, y: Any) -> Any:
        """Return the leading-axis batch sum of ``x`` and ``y``."""
        return self._add_batch_core(x, y)

    @checked_method(in_space="self", out_space="self", arg_positions=(1,), in_batched=True, out_batched=True)
    def scale_batch(self, a: Any, x: Any) -> Any:
        """Return the leading-axis batch scalar product ``a * x``."""
        self.check_scalar(a)
        return self._scale_batch_core(a, x)

    # Default cores. A family that overrode a public operation (older style) keeps it.

    def _flatten_core(self, x: Any) -> DenseArray:
        if type(self).flatten is CoordinateSpace.flatten:
            raise NotImplementedError(f"{type(self).__name__} defines neither flatten nor _flatten_core.")
        return self.flatten(x)

    def _unflatten_core(self, v: DenseArray) -> Any:
        if type(self).unflatten is CoordinateSpace.unflatten:
            raise NotImplementedError(f"{type(self).__name__} defines neither unflatten nor _unflatten_core.")
        return self.unflatten(v)

    def _flatten_batch_core(self, xs: Any) -> DenseArray:
        n = int(getattr(xs, "shape", (len(xs),))[0])
        return self.ops.stack(tuple(self._flatten_core(xs[i]) for i in range(n)), axis=0)

    def _unflatten_batch_core(self, vs: DenseArray) -> Any:
        n = int(getattr(vs, "shape", (len(vs),))[0])
        return self.ops.stack(tuple(self._unflatten_core(vs[i]) for i in range(n)), axis=0)

    def _pair_core(self, xi: Any, x: Any) -> Any:
        return self.ops.real(self.ops.vdot(self._flatten_core(xi), self._flatten_core(x)))

    def _vpair_core(self, xi: Any, xs: Any) -> Any:
        return self.ops.real(self._flatten_batch_core(xs) @ self.ops.conj(self._flatten_core(xi)))

    def _add_batch_core(self, x: Any, y: Any) -> Any:
        return self.ops.vmap(self._add_core, in_axes=(0, 0), out_axes=0)(x, y)

    def _scale_batch_core(self, a: Any, x: Any) -> Any:
        return self.ops.vmap(lambda xi: self._scale_core(a, xi), in_axes=0, out_axes=0)(x)

    def stacked(self, count: int) -> CoordinateSpace:
        """Return ``count`` leading-axis copies of this leaf space as one space."""
        from ..concrete import StackedSpace

        return StackedSpace(self, count, self.ctx)
