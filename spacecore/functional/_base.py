from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .._contextual import ContextBound, resolve_context_priority
from ..backend import Context
from ..space import Space

if TYPE_CHECKING:
    from ..linop import LinOp


Domain = TypeVar("Domain", bound=Space)


class Functional(ContextBound, Generic[Domain]):
    r"""
    Scalar-valued map on a space.

    ``Functional`` represents a map ``F : X -> K`` without assuming any storage
    model. It mirrors the minimal ``LinOp`` contract: the domain is converted
    into the resolved context, value checks follow ``ctx.enable_checks``, and
    batched evaluation is implemented by a backend ``vmap`` fallback.

    Parameters
    ----------
    dom : Space
        Domain space ``X``.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Attributes
    ----------
    dom : Space
        Domain space converted to ``ctx``.
    ctx : Context
        Resolved backend context.
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
        """Evaluate this functional at an element of ``self.domain``."""

    def __call__(self, x: Any) -> Any:
        """Evaluate this functional at ``x``."""
        return self.value(x)

    def compose(self, A: "LinOp") -> "Functional":
        """
        Return the pull-back ``self o A``.

        Parameters
        ----------
        A : LinOp
            Linear operator whose codomain matches this functional's domain.

        Returns
        -------
        Functional
            Functional on ``A.domain`` evaluating ``self.value(A.apply(x))``.
        """
        from ._composed import make_functional_composed

        return make_functional_composed(self, A)

    def vvalue(self, xs: Any, batch_space: Space | None = None) -> Any:
        """Evaluate this functional independently over leading batch axes."""
        return self._fallback_vvalue(xs, batch_space)

    def _infer_batch_shape(self, space: Space, value: Any) -> tuple[int, ...]:
        """Infer leading batch dimensions from a value and base space."""
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
        """Return the batch space used to validate batched inputs."""
        if batch_space is not None:
            return batch_space
        batch_shape = self._infer_batch_shape(space, value)
        return space.batch(batch_shape, tuple(range(len(batch_shape))))

    def _output_batch_space(self, space: Space, input_batch_space: Space) -> Space:
        """Return the batch space corresponding to a batched output."""
        batch_shape = getattr(input_batch_space, "batch_shape", None)
        batch_axes = getattr(input_batch_space, "batch_axes", None)
        if batch_shape is None or batch_axes is None:
            raise TypeError("batch_space must be a BatchSpace-compatible object.")
        return space.batch(tuple(batch_shape), tuple(batch_axes))

    def _require_leading_batch_axes(self, batch_space: Space) -> tuple[int, ...]:
        """Return batch shape or raise when batch axes are not leading."""
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
        """Vectorize ``fn`` over ``batch_ndim`` leading axes."""
        mapped = fn
        for _ in range(batch_ndim):
            mapped = self.ops.vmap(mapped, in_axes=0, out_axes=0)
        return mapped

    def _check_scalar_batch(self, values: Any, batch_shape: tuple[int, ...]) -> None:
        """Raise if scalar batch output does not have ``batch_shape``."""
        shape = tuple(getattr(values, "shape", ()))
        if shape != batch_shape:
            raise ValueError(
                f"Expected scalar batch output with shape {batch_shape}, got {shape}."
            )

    def _fallback_vvalue(self, xs: Any, batch_space: Space | None = None) -> Any:
        """Evaluate this functional over a leading batch with backend ``vmap``."""
        in_space = self._input_batch_space(self.domain, xs, batch_space)
        batch_shape = self._require_leading_batch_axes(in_space)
        if self._enable_checks:
            in_space._check_member(xs)
        values = self._vmap_leading(self.value, len(batch_shape))(xs)
        if self._enable_checks:
            self._check_scalar_batch(values, batch_shape)
        return values

    def assert_domain(self, x: Any) -> None:
        """Raise if ``x`` is not in the domain."""
        self.dom.check_member(x)

    @abstractmethod
    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        ...

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        ...
