"""Elements of a space: a space plus one representation of a member of it.

An element is formal: it knows its space, it is checked against that space once,
at construction, and its arithmetic stays inside the space. The *representation*
is the subclass — :class:`DenseElement` stores the array, :class:`LowRankElement`
stores Hermitian factors — and each subclass names its own data fields.

Arithmetic runs through the same rule table as the operator algebra
(:mod:`spacecore.mapping`). A pair of representations with its own rule uses it
(``LowRank + LowRank`` concatenates factors); any other pair falls back to
``(Element, Element)``, which densifies both and adds in the space. Element times
element is not an operation and raises ``TypeError``.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from ..backend import PyTreeNode
from ..contextual import Context, ContextBound
from ..mapping import Algebraic, Scalar, register
from .checks import SpaceValidationError
from .concrete import HermitianSpace


class Element(Algebraic, PyTreeNode, ContextBound):
    """
    A member of ``space`` in one representation.

    The context and check level are the space's. Membership is checked once, by
    the public constructor; results of arithmetic are members by closure and are
    not re-checked. A subclass declares its data fields in ``_children`` and
    implements :meth:`densify`.

    Parameters
    ----------
    space : Space
        The space this element belongs to.
    *data : Any
        The representation's data fields, in ``_children`` order.
    """

    _children: ClassVar[tuple[str, ...]] = ()
    __array_ufunc__ = None  # NumPy defers to our reflected overloads instead of unwrapping us.

    def __init__(self, space: Any, *data: Any) -> None:
        # Adopt the space's context and check level.
        (self.space,) = self._bind_context(space.ctx, space)
        # Store the representation's named fields in declared order.
        for name, value in zip(self._children, data, strict=True):
            setattr(self, name, value)
        # The single membership check an element ever gets.
        self._check()

    @classmethod
    def _make(cls, space: Any, *data: Any) -> Element:
        """Build without binding or checks: for results known to be members, and pytree rebuilds."""
        # Skip __init__: no context binding, no check.
        obj = object.__new__(cls)
        obj.space, obj._ctx, obj._check_level = space, space.ctx, space.check_level
        for name, value in zip(cls._children, data):
            setattr(obj, name, value)
        return obj

    @abstractmethod
    def densify(self) -> Any:
        """Return the dense backend array this element denotes."""

    def _check(self) -> None:
        """Raise unless this element is a member of its space; subclasses may check cheaper."""
        self.space.check_member(self.densify())

    def _scale(self, s: Any) -> Element:
        """Return ``s * self``; the default densifies. ``s`` is already checked admissible."""
        return DenseElement._make(self.space, self.space.scale(s, self.densify()))

    def _scaled(self, s: Any) -> Element:
        self.space.check_scalar(s)
        return self._scale(s)

    def tree_flatten(self):
        """Expose the data fields as children and the space as static data."""
        return tuple(getattr(self, name) for name in self._children), self.space

    @classmethod
    def tree_unflatten(cls, space, children):
        """Rebuild without checks: transforms pass tracers and placeholders."""
        return cls._make(space, *children)

    def _convert(self, new_ctx: Context) -> Element:
        ops = self.ops

        def move(a: Any) -> Any:
            # Real data stays real (e.g. LowRank eigenvalues in a complex space).
            real = not ops.is_complex_dtype(ops.get_dtype(a))
            dtype = new_ctx.ops.real_dtype(new_ctx.dtype) if real else new_ctx.dtype
            return new_ctx.ops.asarray(a, dtype=dtype)

        # Move every field, then rebuild on the converted space.
        data = (move(getattr(self, name)) for name in self._children)
        return self._make(self.space.convert(new_ctx), *data)

    def _repr_body(self) -> str:
        return self.space._space_descriptor()


class DenseElement(Element):
    """
    An element stored as one dense backend array.

    Parameters
    ----------
    space : Space
        The space this element belongs to.
    array : DenseArray
        The element's coordinates, as accepted by ``space.check_member``.
    """

    _children = ("array",)

    def __init__(self, space: Any, array: Any) -> None:
        super().__init__(space, array)

    def densify(self) -> Any:
        """Return the stored array."""
        return self.array


class LowRankElement(Element):
    """
    A Hermitian element ``U diag(d) U*`` stored as its factors.

    Parameters
    ----------
    space : HermitianSpace
        The Hermitian space; ``U`` has ``space.n`` rows.
    U : DenseArray
        Factor of shape ``(n, r)``.
    d : DenseArray
        Real weights of shape ``(r,)``.
    """

    _children = ("U", "d")

    def __init__(self, space: Any, U: Any, d: Any) -> None:
        super().__init__(space, U, d)

    def densify(self) -> Any:
        """Return ``U diag(d) U*`` as a dense array."""
        return self.space.from_spectrum(self.d, self.U)  # symmetrizes away the ULP skew

    def _check(self) -> None:
        # ponytail: only HermitianSpace has a spectral low-rank form; widen when a second space does.
        space, ops = self.space, self.space.ops
        # U diag(d) U* only means a member of a Hermitian space.
        if not isinstance(space, HermitianSpace):
            raise SpaceValidationError(f"LowRankElement needs a HermitianSpace, got {space!r}.")
        if space.check_level == "none":
            return
        # Cheap checks: factor shapes fit, and d is real.
        if self.U.shape[-2] != space.n or tuple(self.d.shape) != tuple(self.U.shape[-1:]):
            raise SpaceValidationError(
                f"Factors U{tuple(self.U.shape)}, d{tuple(self.d.shape)} do not fit {space!r}."
            )
        if ops.is_complex_dtype(ops.get_dtype(self.d)):
            raise SpaceValidationError("LowRankElement weights d must be real.")
        # Real d makes U diag(d) U* Hermitian by construction; strict still checks the matrix.
        if space.check_level == "strict":
            self.densify()

    def _scale(self, s: Any) -> Element:
        return LowRankElement._make(self.space, self.U, s * self.d)


def _same_space(a: Element, b: Element) -> None:
    if a.space is not b.space and a.space != b.space:
        raise SpaceValidationError(f"Cannot combine elements of {a.space!r} and {b.space!r}.")


def _add(a: Element, b: Element) -> Element:
    _same_space(a, b)
    return DenseElement._make(a.space, a.space.add(a.densify(), b.densify()))


def _add_low_rank(a: LowRankElement, b: LowRankElement) -> Element:
    _same_space(a, b)
    cat = a.ops.concatenate
    return LowRankElement._make(a.space, cat((a.U, b.U), axis=-1), cat((a.d, b.d), axis=-1))


register("add", Element, Element, _add)
register("add", LowRankElement, LowRankElement, _add_low_rank)
register("mul", Scalar, Element, lambda s, x: x._scaled(s))
register("mul", Element, Scalar, lambda x, s: x._scaled(s))
