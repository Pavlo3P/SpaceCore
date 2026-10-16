from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from ._base import Domain, Functional
from .._batching import _batched_real_pairing, _check_batched, _check_scalar_shape
from .._checks import checked_method
from .._checks import CheckLevel
from ..contextual import Context
from ..space import Space, CoordinateSpace, TreeElement, TreeSpace


def _broadcast_space_element(space: Any, value: Any, n: int) -> Any:
    """Broadcast a single space element to a leading-axis batch of size ``n``."""
    # Tree spaces broadcast leaf by leaf; array spaces to (n, *shape).
    if isinstance(space, TreeSpace):
        return space.unflatten_tree(
            tuple(
                _broadcast_space_element(part, component, n)
                for part, component in zip(space.leaf_spaces, space.flatten_tree(value))
            )
        )
    return space.ops.broadcast_to(value, (n,) + tuple(space.shape))



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
    if not isinstance(space, CoordinateSpace):
        return value
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
    """

    def _grad_core(self, x: Any) -> Any:
        return self.representer

    def _vgrad_core(self, xs: Any) -> Any:
        # A linear functional's gradient is its representer, repeated per batch row.
        dom = self.dom
        sample = dom.flatten_tree(xs)[0] if isinstance(dom, TreeSpace) else xs
        n = int(getattr(sample, "shape", (0,))[0])
        return _broadcast_space_element(dom, self.representer, n)


    def _derivative_core(self, x: Any, *args: Any, **kwargs: Any) -> "LinearFunctional":
        """This linear functional itself, even without a representer."""
        return self

    @property
    @abstractmethod
    def representer(self) -> Any:
        """
        Riesz representer of this functional when one is explicitly available.

        Matrix-free functionals may not have a stored representer and should
        raise ``NotImplementedError``.
        """



class InnerProductFunctional(LinearFunctional[Domain]):
    r"""
    The functional :math:`x \mapsto \operatorname{Re}\langle c, x\rangle_X` of a representer ``c``.

    The real part of this pairing is always used. This represents real
    differentials on complex parameter spaces.

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

    _fields = ("c:_c", "dom", "ctx", "check_level")
    _dynamic = ("c",)

    def _value_core(self, x: Any) -> Any:
        return self.domain._real_inner_core(self._c, x)

    def _vvalue_core(self, xs: Any) -> Any:
        return _batched_real_pairing(self.domain, self._c, xs)

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
        """Stored domain element ``c`` defining ``ell_c(x) = Re<c, x>``."""
        return self._c

    def __eq__(self, other: Any) -> bool:
        """Return whether another inner-product functional has the same representer."""
        if not self.same_math(other):              # Tier 1: backend
            return NotImplemented
        if self.domain != other.domain or self.codomain != other.codomain:                         # Tier 2: domain before allclose
            return False
        return bool(self.ops.allclose(                          # Tier 3: representer
            self.domain.flatten(self._c),
            other.domain.flatten(other._c),
            equal_nan=True,
        ))

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
        The callable must return real values; no projection is applied.
    """

    _fields = ("value:value_fn", "dom", "ctx", "vvalue:vvalue_fn", "check_level")

    def _value_core(self, x: Any) -> Any:
        return self.value_fn(x)

    def _vvalue_core(self, xs: Any) -> Any:
        if self.vvalue_fn is None:
            return self._vmap_vvalue(xs)
        return self.vvalue_fn(xs)

    def __init__(
        self,
        value: Callable[[Any], Any],
        dom: Domain,
        ctx: Context | str | None = None,
        vvalue: Callable[[Any], Any] | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        # Validate the callables, then bind the domain.
        if not callable(value):
            raise TypeError(f"value must be callable, got {type(value).__name__}.")
        if vvalue is not None and not callable(vvalue):
            raise TypeError(f"vvalue must be callable, got {type(vvalue).__name__}.")
        super().__init__(dom, ctx, check_level=check_level)
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
        if self.check_level != "none":
            _check_batched(self.codomain, values)  # real values, as the base vvalue checks
        if self._checks_at_least("standard"):
            shape = tuple(getattr(xs, "shape", ()))
            base = tuple(self.domain.shape)
            leading = shape if not base else shape[: len(shape) - len(base)]
            _check_scalar_shape(values, leading)
        return values
