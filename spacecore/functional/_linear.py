from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from ..space import Field
from ._base import Domain, Functional
from .._batching import _check_scalar_shape
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
    cod : Field or None, optional
        Scalar codomain; the domain's ``scalars`` by default.
    """

    def _grad_core(self, x: Any) -> Any:
        return self.representer

    def _vgrad_core(self, xs: Any) -> Any:
        # A linear functional's gradient is its representer, repeated per batch row.
        dom = self.dom
        sample = dom.flatten_tree(xs)[0] if isinstance(dom, TreeSpace) else xs
        n = int(getattr(sample, "shape", (0,))[0])
        return _broadcast_space_element(dom, self.representer, n)

    @property
    def scalar_field(self) -> str:
        """Field over which the map is linear, independent of storage dtype."""
        if self.domain.scalar_field == "real" or self.codomain.scalar_field == "real":
            return "real"
        return "complex"

    def derivative_at(self, x: Any, *args: Any, **kwargs: Any) -> "LinearFunctional":
        """Return this linear functional itself, even without a representer."""
        self.domain.check_member(x)
        return self

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
    The functional :math:`x \mapsto \langle c, x\rangle_X` of a representer ``c``.

    A real codomain or ``real_linear=True`` selects the real part of this
    pairing. This represents real differentials on complex parameter spaces.

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
    cod : Field or None, optional
        Scalar output field; defaults to the domain's scalar field.
    real_linear : bool, optional
        Evaluate the real pairing, even with a complex scalar codomain.
    """

    # A real codomain requests the real pairing, including on complex storage.
    _fields = ("c:_c", "dom", "ctx", "cod", "check_level", "real_linear")
    _dynamic = ("c",)

    def _value_core(self, x: Any) -> Any:
        inner = getattr(self.domain, "_inner_core", self.domain.inner)
        value = inner(self._c, x)
        return self.ops.real(value) if self.real_linear else value

    def _vvalue_core(self, xs: Any) -> Any:
        # <c, x_i> = conj(c) . x_i on flat coordinates; non-Euclidean spaces use R c.
        dom = self.dom
        ops = self.ops
        if dom.is_euclidean and len(tuple(dom.shape)) == 1:
            xs_flat = xs
            c_flat = ops.conj(self._c)
        else:
            c_dual = self._c if dom.is_euclidean else dom.riesz(self._c)
            c_flat = ops.conj(dom.flatten(c_dual))
            xs_flat = dom.flatten_batch(xs)
        value = xs_flat @ c_flat
        return ops.real(value) if self.real_linear else value

    def __init__(
        self,
        c: Any,
        dom: Domain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
        *,
        cod: Field | None = None,
        real_linear: bool = False,
    ) -> None:
        super().__init__(dom, ctx, check_level=check_level, cod=cod)
        self.real_linear = real_linear or self.codomain.scalar_field == "real"
        # Store the representer on the functional's backend, and check it belongs.
        self._c = _convert_space_element(self.domain, c)
        if self._checks_at_least("standard"):
            self.domain._check_member(self._c)

    @property
    def scalar_field(self) -> str:
        return "real" if self.real_linear else super().scalar_field

    @property
    def representer(self) -> Any:
        """Stored domain element ``c`` defining ``ell_c(x) = <c, x>``."""
        return self._c

    def __eq__(self, other: Any) -> bool:
        """Return whether another inner-product functional has the same representer."""
        if not self.same_math(other):              # Tier 1: backend
            return NotImplemented
        if self.domain != other.domain or self.codomain != other.codomain or self.real_linear != other.real_linear:                         # Tier 2: domain before allclose
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
    cod : Field or None, optional
        Scalar codomain; ``dom.scalars`` by default.
    real_linear : bool, optional
        Declare real linearity for a complex-valued callable on complex inputs.
        The callable is used unchanged; no real-part projection is applied.
    """

    _fields = ("value:value_fn", "dom", "ctx", "vvalue:vvalue_fn", "check_level", "cod", "real_linear")

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
        *,
        cod: Field | None = None,
        real_linear: bool = False,
    ) -> None:
        # Validate the callables, then bind the domain.
        if not callable(value):
            raise TypeError(f"value must be callable, got {type(value).__name__}.")
        if vvalue is not None and not callable(vvalue):
            raise TypeError(f"vvalue must be callable, got {type(vvalue).__name__}.")
        super().__init__(dom, ctx, check_level=check_level, cod=cod)
        self.real_linear = real_linear
        self.value_fn = value
        self.vvalue_fn = vvalue

    @property
    def scalar_field(self) -> str:
        """Allow a real-linear callable even when its output is complex."""
        return "real" if self.real_linear else super().scalar_field

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
        if self._checks_at_least("standard"):
            shape = tuple(getattr(xs, "shape", ()))
            base = tuple(self.domain.shape)
            leading = shape if not base else shape[: len(shape) - len(base)]
            _check_scalar_shape(values, leading)
        return values
