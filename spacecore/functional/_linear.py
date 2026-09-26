from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from ..space import Field
from ._base import Domain, Functional
from .._batching import _check_scalar_shape
from .._checks import checked_method
from .._check_policy import CheckLevel
from ..contextual import Context
from ..kernels.core.functional import (
    inner_product_value_core,
    inner_product_vvalue_core,
    linear_grad_core,
    linear_vgrad_core,
    matrixfree_linear_value_core,
    matrixfree_linear_vvalue_core,
)
from ..space import Space, TreeElement, TreeSpace


def _convert_space_element(space: Space, value: Any) -> Any:
    """Convert a value recursively into a possibly tree-valued space."""
    # Tree spaces convert leaf by leaf; other spaces convert the array directly.
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


class LinearFunctional(Functional[Domain]):
    """
    A linear scalar-valued map on ``dom``.

    Parameters
    ----------
    dom : Space
        Domain.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    cod : Field or None, optional
        Scalar codomain; the domain's ``scalars`` by default.
    """

    _grad_core = linear_grad_core
    _vgrad_core = linear_vgrad_core

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


class InnerProductFunctional(LinearFunctional[Domain]):
    r"""
    The linear functional :math:`x \mapsto \langle c, x\rangle_X` of a representer ``c``.

    Parameters
    ----------
    c : array-like
        Riesz representer in ``dom``.
    dom : Space
        Domain.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``dom`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    _value_core = inner_product_value_core
    _vvalue_core = inner_product_vvalue_core

    def __init__(
        self,
        c: Any,
        dom: Domain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level)
        # Store the representer on the functional's backend, and check it belongs.
        self._c = _convert_space_element(self.domain, c)
        if self._checks_at_least("standard"):
            self.domain._check_member(self._c)

    @property
    def representer(self) -> Any:
        """Stored domain element ``c`` defining ``ell_c(x) = <c, x>``."""
        return self._c

    @checked_method(in_space="domain", out_space="codomain")
    def value(self, x: Any) -> Any:
        """Return ``domain.inner(representer, x)``."""
        return self._value_core(x)

    @checked_method(in_space="domain", in_batched=True, out_batched_scalar=True)
    def vvalue(self, xs: Any) -> Any:
        """Evaluate ``domain.inner(representer, xs[i])`` without a Python loop."""
        return self._vvalue_core(xs)

    def __eq__(self, other: Any) -> bool:
        """Return whether another inner-product functional has the same representer."""
        if not self.same_math(other):              # Tier 1: backend
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


class MatrixFreeLinearFunctional(LinearFunctional[Domain]):
    """
    Linear functional given by an evaluation callable; no Riesz representer is stored.

    Parameters
    ----------
    value : callable
        ``value(x)`` returning a scalar for ``x`` in ``dom``.
    dom : Space
        Domain.
    ctx : Context, str, or None, optional
        Backend context.
    vvalue : callable or None, optional
        Batched evaluation; ``vmap`` of ``value`` by default.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    cod : Field or None, optional
        Scalar codomain; ``dom.scalars`` by default.
    """

    _value_core = matrixfree_linear_value_core
    _vvalue_core = matrixfree_linear_vvalue_core

    def __init__(
        self,
        value: Callable[[Any], Any],
        dom: Domain,
        ctx: Context | str | None = None,
        vvalue: Callable[[Any], Any] | None = None,
        check_level: CheckLevel | None = None,
        *,
        cod: Field | None = None,
    ) -> None:
        # Validate the callables, then bind the domain.
        if not callable(value):
            raise TypeError(f"value must be callable, got {type(value).__name__}.")
        if vvalue is not None and not callable(vvalue):
            raise TypeError(f"vvalue must be callable, got {type(vvalue).__name__}.")
        super().__init__(dom, ctx, check_level=check_level, cod=cod)
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

    @checked_method(in_space="domain", out_space="codomain")
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
        return self._value_core(x)

    # NOT ``out_batched_scalar=True``: the decorator derives the expected batch
    # shape as ``(_leading_batch_size(domain, xs),)`` — one leading axis, the
    # contract ``Functional.vvalue`` documents. The check below is deliberately
    # more permissive, stripping the domain shape so a user-supplied
    # ``vvalue_fn`` may take *several* leading axes. Narrowing that to the
    # documented contract is a real behavior change and is left alone here.
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
        # Use the supplied batched evaluator, else vmap; check one scalar per input.
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
        if not self.same_math(other):              # Tier 1: backend
            return NotImplemented
        if self.domain != other.domain or self.codomain != other.codomain:  # Tier 2: endpoints
            return False
        # Callable identity: extensional equality of callables is undecidable.
        return self.value_fn is other.value_fn and self.vvalue_fn is other.vvalue_fn

    def tree_flatten(self):
        """Flatten this functional for pytree registration."""
        children = ()
        aux = (self.value_fn, self.domain, self.ctx, self.vvalue_fn, self.codomain, self.check_level)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this functional from pytree data."""
        value_fn, domain, ctx, vvalue_fn, cod, check_level = aux
        return cls(value_fn, domain, ctx, vvalue_fn, check_level=check_level, cod=cod)

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
            check_level=self.check_level,
            cod=self.codomain,
        )
