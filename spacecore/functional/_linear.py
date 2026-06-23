from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from ._base import Domain, Functional
from .._batching import _check_scalar_shape, _leading_batch_size
from .._checks import checked_method
from ..backend import Context, jax_pytree_class
from ..kernels import core_kernels
from ..space import Space, TreeElement, TreeSpace


def _convert_space_element(space: Space, value: Any) -> Any:
    """Convert a value recursively into a possibly tree-valued space."""
    if isinstance(space, TreeSpace):
        if isinstance(value, TreeElement):
            source_spaces = value.space.leaf_spaces
            leaves = value.leaves
            converted = tuple(
                target.unflatten(space.ctx.asarray(source.flatten(leaf)))
                for source, target, leaf in zip(source_spaces, space.leaf_spaces, leaves)
            )
        else:
            leaves = space.flatten_tree(value)
            converted = tuple(
                leaf_space.ctx.asarray(leaf)
                for leaf_space, leaf in zip(space.leaf_spaces, leaves)
            )
        return space.unflatten_tree(converted)
    return space.ctx.asarray(value)


@core_kernels("functional-linear")
class LinearFunctional(Functional[Domain]):
    r"""
    Represent a linear scalar-valued map.

    Parameters
    ----------
    dom : Space
        Domain space.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.
    """

    @property
    @abstractmethod
    def representer(self) -> Any:
        """
        Riesz representer of this functional when one is explicitly available.

        Matrix-free functionals may not have a stored representer and should
        raise ``NotImplementedError``.
        """

    @checked_method(in_space="domain", out_space="domain")
    def grad(self, x: Any) -> Any:
        """
        Return the constant Riesz gradient of this linear functional.

        For ``ell(x) = <c, x>_X``, the gradient is the space element ``c``.
        Matrix-free functionals without a stored representer inherit the
        ``NotImplementedError`` raised by :attr:`representer`.
        """
        return self._grad_core(x)

    @checked_method(in_space="domain", out_space="domain", in_batched=True, out_batched=True)
    def vgrad(self, xs: Any) -> Any:
        """Return the constant Riesz gradient over a leading batch axis."""
        return self._vgrad_core(xs)


@core_kernels("inner-product-functional")
@jax_pytree_class
class InnerProductFunctional(LinearFunctional[Domain]):
    r"""
    Linear functional represented by a domain element.

    ``InnerProductFunctional(c, X)`` evaluates
    :math:`\ell_c(x) = \langle c, x\rangle_X`.

    Parameters
    ----------
    c : array-like
        Riesz representer in ``dom``.
    dom : Space
        Domain space.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``dom``.

    Attributes
    ----------
    representer : array-like
        Stored domain element ``c``.
    """

    def __init__(
        self,
        c: Any,
        dom: Domain,
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, ctx)
        self._c = _convert_space_element(self.domain, c)
        if self._checks_at_least("standard"):
            self.domain._check_member(self._c)

    @property
    def representer(self) -> Any:
        """Stored domain element ``c`` defining ``ell_c(x) = <c, x>``."""
        return self._c

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """Return ``domain.inner(representer, x)``."""
        return self._value_core(x)

    @checked_method(in_space="domain", in_batched=True)
    def vvalue(self, xs: Any) -> Any:
        """Evaluate ``domain.inner(representer, xs[i])`` without a Python loop."""
        values = self._vvalue_core(xs)
        if self._checks_at_least("standard"):
            _check_scalar_shape(values, (_leading_batch_size(self.domain, xs),))
        return values

    def __eq__(self, other: Any) -> bool:
        """Return whether another inner-product functional has the same representer."""
        if not self._eq_backend_compatible(other):              # Tier 1: backend
            return NotImplemented
        if self.domain != other.domain:                         # Tier 2: domain before allclose
            return False
        return bool(self.ops.allclose(                          # Tier 3: representer
            self.domain.flatten(self._c),
            other.domain.flatten(other._c),
            equal_nan=True,
        ))

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        children = (self._c,)
        aux = (self.domain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        domain, ctx = aux
        c = children[0]
        return cls(c, domain, ctx)

    def _convert(self, new_ctx: Context) -> InnerProductFunctional:
        """Convert the domain and representer to ``new_ctx``."""
        return InnerProductFunctional(self._c, self.domain.convert(new_ctx), new_ctx)


@core_kernels("matrixfree-linear-functional")
@jax_pytree_class
class MatrixFreeLinearFunctional(LinearFunctional[Domain]):
    """
    Linear functional defined by user-supplied evaluation callables.

    ``MatrixFreeLinearFunctional(value, X)`` represents a linear scalar-valued
    map on ``X`` without storing or materializing a Riesz representer.

    Parameters
    ----------
    value : callable
        Callable with signature ``value(x: Any) -> Any`` accepting an element of
        ``dom`` and returning a scalar-like backend value.
    dom : Space
        Domain space of the functional.
    ctx : Context, str, or None, optional
        Optional context specification. An explicit context wins over inferred
        and default contexts.
    vvalue : callable or None, optional
        Optional callable with signature ``vvalue(xs: Any) -> Any`` for batched
        evaluation. If omitted, backend ``vmap`` fallback is used.

    Returns
    -------
    MatrixFreeLinearFunctional
        Functional using the supplied callable for scalar evaluation and,
        optionally, batched scalar evaluation.
    """

    def __init__(
        self,
        value: Callable[[Any], Any],
        dom: Domain,
        ctx: Context | str | None = None,
        vvalue: Callable[[Any], Any] | None = None,
    ) -> None:
        """
        Initialize a matrix-free linear functional.

        Parameters
        ----------
        value:
            Callable ``value(x)`` accepting an element of ``dom`` and returning
            a scalar-like value.
        dom:
            Domain space of the functional.
        ctx:
            Optional context specification for the functional and converted
            domain.
        vvalue:
            Optional callable ``vvalue(xs)`` accepting a batch of domain
            elements and returning a batch of scalar-like values.

        Returns
        -------
        None
            The initializer stores the callables and converted domain on
            ``self``.
        """
        if not callable(value):
            raise TypeError(f"value must be callable, got {type(value).__name__}.")
        if vvalue is not None and not callable(vvalue):
            raise TypeError(f"vvalue must be callable, got {type(vvalue).__name__}.")
        super().__init__(dom, ctx)
        self.value_fn = value
        self.vvalue_fn = vvalue

    @property
    def representer(self) -> Any:
        """
        Raise because matrix-free functionals do not store a representer.

        Parameters
        ----------
        None

        Returns
        -------
        Any
            This property never returns; it raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not store a Riesz representer.")

    @checked_method(in_space="domain")
    def value(self, x: Any) -> Any:
        """
        Evaluate the scalar functional.

        Parameters
        ----------
        x:
            Element of ``self.domain`` passed to ``value_fn``.

        Returns
        -------
        Any
            Scalar-like backend value returned by ``value_fn``.
        """
        y = self._value_core(x)
        if self._checks_at_least("standard"):
            _check_scalar_shape(y, ())
        return y

    @checked_method(in_space="domain", in_batched=True)
    def vvalue(self, xs: Any) -> Any:
        """
        Evaluate the scalar functional over a batch of domain elements.

        Parameters
        ----------
        xs:
            Batched element of ``self.domain``.

        Returns
        -------
        Any
            Backend array of scalar-like values with shape matching the leading
            batch shape.
        """
        if self.vvalue_fn is None:
            return super().vvalue(xs)
        values = self.vvalue_fn(xs)
        if self._checks_at_least("standard"):
            shape = tuple(getattr(xs, "shape", ()))
            base = tuple(self.domain.shape)
            leading = shape if not base else shape[: len(shape) - len(base)]
            _check_scalar_shape(values, leading)
        return values

    def __eq__(self, other: Any) -> bool:
        """Return whether another matrix-free functional uses the same callables."""
        if not self._eq_backend_compatible(other):              # Tier 1: backend
            return NotImplemented
        if self.domain != other.domain:                         # Tier 2: domain
            return False
        # Callable identity: extensional equality of callables is undecidable.
        return self.value_fn is other.value_fn and self.vvalue_fn is other.vvalue_fn

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        children = ()
        aux = (self.value_fn, self.domain, self.ctx, self.vvalue_fn)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        value_fn, domain, ctx, vvalue_fn = aux
        return cls(value_fn, domain, ctx, vvalue_fn)

    def _convert(self, new_ctx: Context) -> MatrixFreeLinearFunctional:
        """
        Convert this functional to ``new_ctx``.

        Parameters
        ----------
        new_ctx:
            Concrete target context for the converted domain.

        Returns
        -------
        MatrixFreeLinearFunctional
            Functional with converted domain and the same user-supplied
            callables.
        """
        return MatrixFreeLinearFunctional(
            self.value_fn,
            self.domain.convert(new_ctx),
            new_ctx,
            self.vvalue_fn,
        )
