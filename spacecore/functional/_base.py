from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from .._contextual import ContextBound, resolve_context_priority
from ..backend import Context
from ..space import Space


Domain = TypeVar("Domain", bound=Space)


class Functional(ContextBound, Generic[Domain]):
    """
    Scalar-valued map on a space.

    ``Functional`` represents a map ``F : X -> K`` without assuming any storage
    model. It mirrors the minimal ``LinOp`` contract: the domain is converted
    into the resolved context, value checks follow ``ctx.enable_checks``, and
    batched evaluation is implemented by a backend ``vmap`` fallback.
    """

    def __init__(self, dom: Domain, ctx: Context | str | None = None) -> None:
        ctx = resolve_context_priority(ctx, dom)
        super().__init__(ctx)
        self.dom = dom.convert(self.ctx)
        self._enable_checks = self.ctx.enable_checks

    @property
    def domain(self) -> Domain:
        """Domain space of this scalar-valued map."""
        return self.dom

    @abstractmethod
    def value(self, x: Any) -> Any:
        """
        Evaluate this functional at ``x``.

        Contract:
          - x is an element of ``self.domain``;
          - the return value is scalar-like in the functional context.
        """

    def __call__(self, x: Any) -> Any:
        """Evaluate this functional at ``x``."""
        return self.value(x)

    def vvalue(self, xs: Any, batch_space: Space | None = None) -> Any:
        """Evaluate this functional independently over leading batch axes."""
        return self._fallback_vvalue(xs, batch_space)

    def _infer_batch_shape(self, space: Space, value: Any) -> tuple[int, ...]:
        if hasattr(space, "spaces") and isinstance(value, tuple) and value:
            return self._infer_batch_shape(space.spaces[0], value[0])
        shape = tuple(getattr(value, "shape", ()))
        base_shape = tuple(space.shape)
        if not base_shape:
            return shape
        if len(shape) < len(base_shape) or shape[-len(base_shape):] != base_shape:
            raise ValueError(
                f"Cannot infer leading batch shape for value shape {shape} "
                f"and base space shape {base_shape}."
            )
        return shape[: len(shape) - len(base_shape)]

    def _input_batch_space(
        self,
        space: Space,
        value: Any,
        batch_space: Space | None,
    ) -> Space:
        if batch_space is not None:
            return batch_space
        batch_shape = self._infer_batch_shape(space, value)
        return space.batch(batch_shape, tuple(range(len(batch_shape))))

    def _output_batch_space(self, space: Space, input_batch_space: Space) -> Space:
        batch_shape = getattr(input_batch_space, "batch_shape", None)
        batch_axes = getattr(input_batch_space, "batch_axes", None)
        if batch_shape is None or batch_axes is None:
            raise TypeError("batch_space must be a BatchSpace-compatible object.")
        return space.batch(tuple(batch_shape), tuple(batch_axes))

    def _require_leading_batch_axes(self, batch_space: Space) -> tuple[int, ...]:
        batch_shape = tuple(getattr(batch_space, "batch_shape", ()))
        batch_axes = tuple(getattr(batch_space, "batch_axes", ()))
        expected_axes = tuple(range(len(batch_shape)))
        if batch_axes != expected_axes:
            raise ValueError(
                "Functional batching currently expects leading batch axes; "
                f"got batch_axes={batch_axes}, expected {expected_axes}."
            )
        return batch_shape

    def _vmap_leading(self, fn: Any, batch_ndim: int) -> Any:
        mapped = fn
        for _ in range(batch_ndim):
            mapped = self.ops.vmap(mapped, in_axes=0, out_axes=0)
        return mapped

    def _check_scalar_batch(self, values: Any, batch_shape: tuple[int, ...]) -> None:
        shape = tuple(getattr(values, "shape", ()))
        if shape != batch_shape:
            raise ValueError(
                f"Expected scalar batch output with shape {batch_shape}, got {shape}."
            )

    def _fallback_vvalue(self, xs: Any, batch_space: Space | None = None) -> Any:
        in_space = self._input_batch_space(self.domain, xs, batch_space)
        batch_shape = self._require_leading_batch_axes(in_space)
        if self._enable_checks:
            in_space._check_member(xs)
        values = self._vmap_leading(self.value, len(batch_shape))(xs)
        if self._enable_checks:
            self._check_scalar_batch(values, batch_shape)
        return values

    def assert_domain(self, x: Any) -> None:
        self.dom.check_member(x)

    @abstractmethod
    def tree_flatten(self):
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux, children):
        ...
