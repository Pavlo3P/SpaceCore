from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar
import warnings

from .._batching import _check_batched
from .._contextual import ContextBound, resolve_context_priority
from ..backend import Context
from ..space import Space

if TYPE_CHECKING:
    from ..linop import LinOp


Domain = TypeVar("Domain", bound=Space)
_VMAP_FALLBACK_WARNED: set[tuple[type, str]] = set()
_VMAP_FALLBACK_WARN_BATCH = 32


def _check_scalar_shape(values: Any, shape: tuple[int, ...]) -> None:
    """Raise if scalar output does not have ``shape``."""
    value_shape = tuple(getattr(values, "shape", ()))
    if value_shape != shape:
        raise ValueError(f"Expected scalar batch output with shape {shape}, got {value_shape}.")


def _leading_batch_size(space: Space, xs: Any) -> int:
    """Return the leading batch size for dense-array batches."""
    if isinstance(xs, tuple) and xs:
        return _leading_batch_size(getattr(space, "spaces", (space,))[0], xs[0])
    shape = tuple(getattr(xs, "shape", ()))
    base = tuple(space.shape)
    if not shape:
        return 0
    if base:
        return int(shape[0])
    return int(shape[0])


def _warn_vmap_fallback_once(obj: Any, method: str, batch_size: int) -> None:
    """Warn once per class/method for NumPy-style Python-loop batched fallback."""
    if batch_size <= _VMAP_FALLBACK_WARN_BATCH or obj.ops.has_native_vmap:
        return
    key = (type(obj), method)
    if key in _VMAP_FALLBACK_WARNED:
        return
    _VMAP_FALLBACK_WARNED.add(key)
    warnings.warn(
        f"{type(obj).__name__}.{method} falls back to a Python loop on this backend "
        "(no native vmap); this is O(batch). Provide a vectorized batched override, "
        "or use JAX/Torch.",
        RuntimeWarning,
        stacklevel=3,
    )


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

    def vvalue(self, xs: Any) -> Any:
        """Evaluate over a leading batch axis. Input must have shape ``(N,) + domain.shape``; use ``moveaxis`` for other layouts."""
        if self._enable_checks:
            _check_batched(self.domain, xs)
        _warn_vmap_fallback_once(self, "vvalue", _leading_batch_size(self.domain, xs))
        return self.ops.vmap(self.value, in_axes=0, out_axes=0)(xs)

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
