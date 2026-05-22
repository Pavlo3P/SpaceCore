from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ._base import Domain, Functional
from ..backend import Context, jax_pytree_class
from ..space import Space


def _convert_space_element(space: Space, value: Any) -> Any:
    if hasattr(space, "spaces") and isinstance(value, tuple):
        if len(value) != len(space.spaces):
            raise ValueError(
                f"Expected tuple of length {len(space.spaces)}, got {len(value)}."
            )
        return tuple(
            _convert_space_element(component_space, component)
            for component_space, component in zip(space.spaces, value)
        )
    return space.ctx.asarray(value)


class LinearFunctional(Functional[Domain]):
    """Linear scalar-valued map ``ell : X -> K``."""

    @property
    @abstractmethod
    def representer(self) -> Any:
        """
        Riesz representer of this functional when one is explicitly available.

        Matrix-free functionals may not have a stored representer and should
        raise ``NotImplementedError``.
        """


@jax_pytree_class
class InnerProductFunctional(LinearFunctional[Domain]):
    """
    Linear functional represented by a domain element.

    ``InnerProductFunctional(c, X)`` evaluates ``ell_c(x) = <c, x>_X``.
    """

    def __init__(
        self,
        c: Any,
        dom: Domain,
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, ctx)
        self._c = _convert_space_element(self.domain, c)
        if self._enable_checks:
            self.domain._check_member(self._c)

    @property
    def representer(self) -> Any:
        """Stored domain element ``c`` defining ``ell_c(x) = <c, x>``."""
        return self._c

    def value(self, x: Any) -> Any:
        """Return ``domain.inner(representer, x)``."""
        if self._enable_checks:
            self.domain._check_member(x)
        return self.domain.inner(self._c, x)

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.domain == other.domain and self.ops.allclose(
                self.domain.flatten(self._c),
                other.domain.flatten(other._c),
            )
        return False

    def tree_flatten(self):
        children = (self._c,)
        aux = (self.domain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, ctx = aux
        c = children[0]
        return cls(c, domain, ctx)

    def _convert(self, new_ctx: Context) -> InnerProductFunctional:
        return InnerProductFunctional(self._c, self.domain.convert(new_ctx), new_ctx)


@jax_pytree_class
class MatrixFreeLinearFunctional(LinearFunctional[Domain]):
    """
    Linear functional defined by user-supplied evaluation callables.

    No representer is stored or materialized.
    """

    def __init__(
        self,
        value: Any,
        dom: Domain,
        ctx: Context | str | None = None,
        vvalue: Any | None = None,
    ) -> None:
        if not callable(value):
            raise TypeError(f"value must be callable, got {type(value).__name__}.")
        if vvalue is not None and not callable(vvalue):
            raise TypeError(f"vvalue must be callable, got {type(vvalue).__name__}.")
        super().__init__(dom, ctx)
        self.value_fn = value
        self.vvalue_fn = vvalue

    @property
    def representer(self) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not store a Riesz representer."
        )

    def value(self, x: Any) -> Any:
        """Return ``value_fn(x)``."""
        if self._enable_checks:
            self.domain._check_member(x)
        y = self.value_fn(x)
        if self._enable_checks:
            self._check_scalar_batch(y, ())
        return y

    def vvalue(self, xs: Any, batch_space: Space | None = None) -> Any:
        """Return ``vvalue_fn(xs)`` when supplied, otherwise use fallback batching."""
        if self.vvalue_fn is None:
            return super().vvalue(xs, batch_space)
        in_space = self._input_batch_space(self.domain, xs, batch_space)
        batch_shape = self._require_leading_batch_axes(in_space)
        if self._enable_checks:
            in_space._check_member(xs)
        values = self.vvalue_fn(xs)
        if self._enable_checks:
            self._check_scalar_batch(values, batch_shape)
        return values

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return (
                self.domain == other.domain
                and self.value_fn is other.value_fn
                and self.vvalue_fn is other.vvalue_fn
            )
        return False

    def tree_flatten(self):
        children = ()
        aux = (self.value_fn, self.domain, self.ctx, self.vvalue_fn)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        value_fn, domain, ctx, vvalue_fn = aux
        return cls(value_fn, domain, ctx, vvalue_fn)

    def _convert(self, new_ctx: Context) -> MatrixFreeLinearFunctional:
        return MatrixFreeLinearFunctional(
            self.value_fn,
            self.domain.convert(new_ctx),
            new_ctx,
            self.vvalue_fn,
        )
