from __future__ import annotations

from numbers import Number
from typing import Any, Callable, Sequence

from ._base import LinOp, Domain, Codomain
from ._metric import _requires_euclidean_or_riesz
from .._checks import checked_method
from .._contextual import resolve_context_priority
from .._contextual._bound import _same_math_context
from ..backend import Context, jax_pytree_class
from ..space import VectorSpace


def is_scalar_like(value: Any) -> bool:
    """Return whether ``value`` can be used as a scalar multiplier for a ``LinOp``."""
    if isinstance(value, Number):
        return True
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape) == ()
    ndim = getattr(value, "ndim", None)
    return ndim == 0


def _conjugate_scalar(value: Any) -> Any:
    """Return the scalar conjugate when the value supports conjugation."""
    if hasattr(value, "conjugate"):
        return value.conjugate()
    if hasattr(value, "conj"):
        return value.conj()
    return value


def _leading_shape(space: Any, value: Any) -> tuple[int, ...]:
    """Infer leading dimensions from a batched value."""
    parts = getattr(space, "spaces", None)
    if parts is not None and isinstance(value, tuple) and value:
        return _leading_shape(parts[0], value[0])
    shape = tuple(getattr(value, "shape", ()))
    base = tuple(space.shape)
    return shape if not base else shape[: len(shape) - len(base)]


def _batched_zeros(space: Any, leading_shape: tuple[int, ...]) -> Any:
    """Return a batched zero value for ``space``."""
    parts = getattr(space, "spaces", None)
    if parts is not None:
        return tuple(_batched_zeros(part, leading_shape) for part in parts)
    return space.ops.zeros(leading_shape + tuple(space.shape), dtype=space.dtype)


def _require_same_context(ops: Sequence[LinOp]) -> Context:
    """Return the common context for algebra operands or raise."""
    ctx = ops[0].ctx
    for i, op in enumerate(ops[1:], start=1):
        if not _same_math_context(ops[0].ctx, op.ctx):
            raise ValueError(
                "All LinOp operands in an algebraic expression must have the same ctx; "
                f"operand 0 has ctx {ctx!r}, operand {i} has ctx {op.ctx!r}."
            )
    return ctx


def _same_space_for_algebra(left: Any, right: Any) -> bool:
    """Return whether two spaces are compatible for algebraic composition."""
    if left == right:
        return True
    if type(left) is not type(right):
        return False
    if tuple(left.shape) != tuple(right.shape):
        return False
    if not _same_math_context(left.ctx, right.ctx):
        return False
    try:
        return left.convert(right.ctx) == right
    except Exception:
        return False


def _require_linop(op: Any, name: str) -> LinOp:
    """Return ``op`` as a linear operator or raise a typed error."""
    if not isinstance(op, LinOp):
        raise TypeError(f"{name} must be a LinOp, got {type(op).__name__}.")
    return op


def _scalar_equal(value: Any, target: Any) -> bool:
    """Return whether two scalar-like values compare equal."""
    try:
        return bool(value == target)
    except Exception:
        return False


def _is_zero_scalar(value: Any) -> bool:
    """Return whether ``value`` is scalar-like zero."""
    return _scalar_equal(value, 0)


def _is_one_scalar(value: Any) -> bool:
    """Return whether ``value`` is scalar-like one."""
    return _scalar_equal(value, 1)


def _flatten_sum_terms(ops: Sequence[LinOp]) -> tuple[LinOp, ...]:
    """Flatten nested lazy sums into a tuple of terms."""
    terms: list[LinOp] = []
    for i, op in enumerate(ops):
        op = _require_linop(op, f"ops[{i}]")
        if isinstance(op, SumLinOp):
            terms.extend(_flatten_sum_terms(op.parts))
        else:
            terms.append(op)
    return tuple(terms)


def make_sum(ops: Sequence[LinOp]) -> LinOp:
    """
    Return a locally simplified lazy sum of linear operators.

    This factory performs only local algebraic canonicalization: nested
    ``SumLinOp`` nodes are flattened and ``ZeroLinOp`` terms are removed. It
    does not collect like terms, reorder operands, or attempt full symbolic
    optimization. All operands must have the same context, domain, and codomain
    before a simplified operator is returned.

    Parameters
    ----------
    ops : sequence of LinOp
        Nonempty sequence of operators with common domain and codomain.

    Returns
    -------
    LinOp
        Simplified lazy sum, a single operand, or a zero operator.
    """
    if not ops:
        raise ValueError("make_sum requires a nonempty sequence of LinOp operands.")

    terms = _flatten_sum_terms(ops)
    ctx = _require_same_context(terms)
    domain = terms[0].domain
    codomain = terms[0].codomain
    for i, op in enumerate(terms[1:], start=1):
        if (
            not _same_space_for_algebra(op.domain, domain)
            or not _same_space_for_algebra(op.codomain, codomain)
        ):
            raise ValueError(
                "All SumLinOp operands must have the same domain and codomain; "
                f"operand 0 maps {domain!r} -> {codomain!r}, "
                f"operand {i} maps {op.domain!r} -> {op.codomain!r}."
            )

    nonzero_terms = tuple(op for op in terms if not isinstance(op, ZeroLinOp))
    if not nonzero_terms:
        return ZeroLinOp(domain, codomain, ctx)
    if len(nonzero_terms) == 1:
        return nonzero_terms[0]
    return SumLinOp(nonzero_terms)


def make_scaled(scalar: Any, op: LinOp) -> LinOp:
    """
    Return a locally simplified scalar multiple of a linear operator.

    This factory performs only local algebraic canonicalization: zero and unit
    scalars are simplified, and nested ``ScaledLinOp`` nodes are folded into one
    scalar. It does not distribute scaling over sums or perform full symbolic
    optimization. Complex scalars retain the usual conjugated coefficient in
    ``rapply`` through ``ScaledLinOp``.

    Parameters
    ----------
    scalar : scalar-like
        Scalar coefficient multiplying ``op``.
    op : LinOp
        Operator to scale.

    Returns
    -------
    LinOp
        Simplified scalar multiple.
    """
    op = _require_linop(op, "op")
    if not is_scalar_like(scalar):
        raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")

    if _is_zero_scalar(scalar):
        return ZeroLinOp(op.domain, op.codomain, op.ctx)
    if _is_one_scalar(scalar):
        return op
    if isinstance(op, ZeroLinOp):
        return op
    if isinstance(op, ScaledLinOp):
        return make_scaled(scalar * op.scalar, op.op)
    return ScaledLinOp(scalar, op)


def make_composed(left: LinOp, right: LinOp) -> LinOp:
    """
    Return a locally simplified composition of two linear operators.

    This factory performs only local algebraic canonicalization: identity
    factors are removed and compositions with zero maps become zero maps. It
    preserves the binary ``ComposedLinOp`` representation and does not flatten
    multi-factor chains or attempt full symbolic optimization. Operands must
    have the same context and compatible middle spaces before a simplified
    operator is returned.

    Parameters
    ----------
    left : LinOp
        Operator applied second.
    right : LinOp
        Operator applied first.

    Returns
    -------
    LinOp
        Simplified lazy composition representing ``left @ right``.
    """
    left = _require_linop(left, "left")
    right = _require_linop(right, "right")
    _require_same_context((left, right))
    if not _same_space_for_algebra(right.codomain, left.domain):
        raise ValueError(
            "ComposedLinOp requires right.codomain == left.domain; "
            f"got {right.codomain!r} and {left.domain!r}."
        )

    if isinstance(right, IdentityLinOp):
        return left
    if isinstance(left, IdentityLinOp):
        return right
    if isinstance(left, ZeroLinOp):
        return ZeroLinOp(right.domain, left.codomain, left.ctx)
    if isinstance(right, ZeroLinOp):
        return ZeroLinOp(right.domain, left.codomain, left.ctx)
    return ComposedLinOp(left, right)


@jax_pytree_class
class ScaledLinOp(LinOp[Domain, Codomain]):
    r"""
    Lazy scalar multiple of a linear operator.

    ``ScaledLinOp(alpha, A)`` represents the mathematical operator
    ``alpha * A``. Its context is exactly ``A.ctx``; its domain is ``A.domain``
    and its codomain is ``A.codomain``. No dense matrix representation is
    formed.

    The forward action is ``apply(x) = alpha * A.apply(x)`` for
    ``x in A.domain``. The reverse action is
    ``rapply(y) = conj(alpha) * A.rapply(y)`` for ``y in A.codomain``, so
    complex scalars use the conjugated coefficient.

    Parameters
    ----------
    scalar : scalar-like
        Scalar multiplier.
    op : LinOp
        Operator being scaled.

    Attributes
    ----------
    scalar : scalar-like
        Stored scalar multiplier.
    op : LinOp
        Stored operand.
    """

    def __init__(self, scalar: Any, op: LinOp[Domain, Codomain]) -> None:
        op = _require_linop(op, "op")
        if not is_scalar_like(scalar):
            raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")
        super().__init__(op.domain, op.codomain, op.ctx)
        self.scalar = scalar
        self.op = op

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``scalar * op.apply(x)``."""
        y = self.op.apply(x)
        return self.codomain.scale(self.scalar, y)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Return ``conj(scalar) * op.rapply(y)``."""
        x = self.op.rapply(y)
        return self.domain.scale(_conjugate_scalar(self.scalar), x)

    def vapply(self, xs: Any) -> Any:
        """Return ``scalar * op.vapply(xs)``."""
        ys = self.op.vapply(xs)
        return self.codomain.scale_batch(self.scalar, ys)

    def rvapply(self, ys: Any) -> Any:
        """Return ``conj(scalar) * op.rvapply(ys)``."""
        xs = self.op.rvapply(ys)
        return self.domain.scale_batch(_conjugate_scalar(self.scalar), xs)

    def __eq__(self, other: Any) -> bool:
        """Return whether another scaled operator has the same scalar and operand."""
        if type(other) is type(self):
            return self.scalar == other.scalar and self.op == other.op
        return False

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        children = (self.scalar, self.op)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        scalar, op = children
        return cls(scalar, op)

    def _convert(self, new_ctx: Context) -> ScaledLinOp:
        """Convert the operand to ``new_ctx`` while preserving the scalar."""
        return ScaledLinOp(self.scalar, self.op.convert(new_ctx))


@jax_pytree_class
class SumLinOp(LinOp[Domain, Codomain]):
    r"""
    Lazy finite sum of linear operators with common spaces.

    ``SumLinOp((A1, ..., Ak))`` represents ``A1 + ... + Ak`` for a nonempty
    sequence of ``LinOp`` instances. All operands must have the same ``ctx``,
    the same domain, and the same codomain before construction. The resulting
    operator has that shared context, domain, and codomain.

    The forward action is ``apply(x) = sum_i Ai.apply(x)`` for the shared
    domain element ``x``. The reverse action is
    ``rapply(y) = sum_i Ai.rapply(y)`` for the shared codomain element ``y``.

    Parameters
    ----------
    ops : sequence of LinOp
        Nonempty sequence of operators with common context, domain, and
        codomain.

    Attributes
    ----------
    parts : tuple of LinOp
        Stored operands in the lazy sum.
    """

    def __init__(self, ops: Sequence[LinOp[Domain, Codomain]]) -> None:
        if not ops:
            raise ValueError("SumLinOp requires a nonempty sequence of LinOp operands.")
        parts = tuple(_require_linop(op, f"ops[{i}]") for i, op in enumerate(ops))
        ctx = _require_same_context(parts)
        domain = parts[0].domain
        codomain = parts[0].codomain
        for i, op in enumerate(parts[1:], start=1):
            if (
                not _same_space_for_algebra(op.domain, domain)
                or not _same_space_for_algebra(op.codomain, codomain)
            ):
                raise ValueError(
                    "All SumLinOp operands must have the same domain and codomain; "
                    f"operand 0 maps {domain!r} -> {codomain!r}, "
                    f"operand {i} maps {op.domain!r} -> {op.codomain!r}."
                )
        super().__init__(domain, codomain, ctx)
        self.ops_tuple = parts

    @property
    def parts(self) -> tuple[LinOp[Domain, Codomain], ...]:
        """Operators in this lazy sum."""
        return self.ops_tuple

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``sum_i ops[i].apply(x)``."""
        acc = self.ops_tuple[0].apply(x)
        for op in self.ops_tuple[1:]:
            yi = op.apply(x)
            acc = acc + yi if type(self.codomain) is VectorSpace else self.codomain.add(acc, yi)
        return acc

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Return ``sum_i ops[i].rapply(y)``."""
        acc = self.ops_tuple[0].rapply(y)
        for op in self.ops_tuple[1:]:
            xi = op.rapply(y)
            acc = acc + xi if type(self.domain) is VectorSpace else self.domain.add(acc, xi)
        return acc

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``sum_i ops[i].vapply(xs)``."""
        acc = self.ops_tuple[0].vapply(xs)
        for op in self.ops_tuple[1:]:
            yi = op.vapply(xs)
            acc = acc + yi if type(self.codomain) is VectorSpace else self.codomain.add_batch(acc, yi)
        return acc

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Return ``sum_i ops[i].rvapply(ys)``."""
        acc = self.ops_tuple[0].rvapply(ys)
        for op in self.ops_tuple[1:]:
            xi = op.rvapply(ys)
            acc = acc + xi if type(self.domain) is VectorSpace else self.domain.add_batch(acc, xi)
        return acc

    def __eq__(self, other: Any) -> bool:
        """Return whether another sum has the same operands."""
        if type(other) is type(self):
            return self.ops_tuple == other.ops_tuple
        return False

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        children = self.ops_tuple
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        return cls(tuple(children))

    def _convert(self, new_ctx: Context) -> SumLinOp:
        """Convert all operands to ``new_ctx``."""
        return SumLinOp(tuple(op.convert(new_ctx) for op in self.ops_tuple))


@jax_pytree_class
class ComposedLinOp(LinOp[Domain, Codomain]):
    r"""
    Lazy composition of two linear operators.

    ``ComposedLinOp(A, B)`` represents ``A @ B = A circ B``. The operands must
    have the same ``ctx`` before construction, and ``B.codomain`` must equal
    ``A.domain``. The resulting operator has domain ``B.domain`` and codomain
    ``A.codomain``.

    The forward action is ``apply(x) = A.apply(B.apply(x))`` for
    ``x in B.domain``. The reverse action is ``rapply(z) = B.rapply(A.rapply(z))``
    for ``z in A.codomain``.

    Parameters
    ----------
    left : LinOp
        Operator applied second.
    right : LinOp
        Operator applied first.

    Attributes
    ----------
    left : LinOp
        Left operand.
    right : LinOp
        Right operand.
    """

    def __init__(self, left: LinOp, right: LinOp) -> None:
        left = _require_linop(left, "left")
        right = _require_linop(right, "right")
        _require_same_context((left, right))
        if not _same_space_for_algebra(right.codomain, left.domain):
            raise ValueError(
                "ComposedLinOp requires right.codomain == left.domain; "
                f"got {right.codomain!r} and {left.domain!r}."
            )
        super().__init__(right.domain, left.codomain, left.ctx)
        self.left = left
        self.right = right

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``left.apply(right.apply(x))``."""
        return self.left.apply(self.right.apply(x))

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, z: Any) -> Any:
        """Return ``right.rapply(left.rapply(z))``."""
        return self.right.rapply(self.left.rapply(z))

    def vapply(self, xs: Any) -> Any:
        """Return ``left.vapply(right.vapply(xs))``."""
        return self.left.vapply(self.right.vapply(xs))

    def rvapply(self, zs: Any) -> Any:
        """Return ``right.rvapply(left.rvapply(zs))``."""
        return self.right.rvapply(self.left.rvapply(zs))

    def __eq__(self, other: Any) -> bool:
        """Return whether another composition has the same operands."""
        if type(other) is type(self):
            return self.left == other.left and self.right == other.right
        return False

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        children = (self.left, self.right)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        left, right = children
        return cls(left, right)

    def _convert(self, new_ctx: Context) -> ComposedLinOp:
        """Convert both operands to ``new_ctx``."""
        return ComposedLinOp(self.left.convert(new_ctx), self.right.convert(new_ctx))


@jax_pytree_class
class ZeroLinOp(LinOp[Domain, Codomain]):
    r"""
    Lazy zero map between two spaces.

    ``ZeroLinOp(X, Y)`` represents the linear map ``0 : X -> Y``. The context is
    resolved from the optional ``ctx`` argument and the two spaces, then both
    spaces are converted to that context. Its domain is ``X`` and its codomain
    is ``Y`` in the resolved context.

    The forward action is ``apply(x) = 0_Y`` for ``x in X``. The reverse action
    is ``rapply(y) = 0_X`` for ``y in Y``.

    Parameters
    ----------
    dom : Space
        Domain space.
    cod : Space
        Codomain space.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from the spaces.
    """

    def __init__(
        self,
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, cod, ctx)

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return the zero element of the codomain."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        """Return the codomain zero without membership checks."""
        return self.codomain.zeros()

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Return the zero element of the domain."""
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        """Return the domain zero without membership checks."""
        return self.domain.zeros()

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return the batched zero element of the codomain."""
        return _batched_zeros(self.codomain, _leading_shape(self.domain, xs))

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Return the batched zero element of the domain."""
        return _batched_zeros(self.domain, _leading_shape(self.codomain, ys))

    def to_dense(self) -> Any:
        """
        Return the dense tensor representation of the zero map.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        return self.ops.zeros(tuple(self.codomain.shape) + tuple(self.domain.shape), dtype=self.dtype)

    def is_hermitian(self) -> bool:
        """
        Return whether the zero map is Hermitian.

        Returns
        -------
        bool
            ``True`` exactly when domain and codomain are the same space.
        """
        return self.domain == self.codomain

    def __eq__(self, other: Any) -> bool:
        """Return whether another zero map has the same spaces."""
        if type(other) is type(self):
            return self.domain == other.domain and self.codomain == other.codomain
        return False

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        children = ()
        aux = (self.domain, self.codomain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        domain, codomain, ctx = aux
        return cls(domain, codomain, ctx)

    def _convert(self, new_ctx: Context) -> ZeroLinOp:
        """Convert domain and codomain spaces to ``new_ctx``."""
        return ZeroLinOp(self.domain.convert(new_ctx), self.codomain.convert(new_ctx), new_ctx)


@jax_pytree_class
class IdentityLinOp(LinOp[Domain, Domain]):
    r"""
    Lazy identity map on a space.

    ``IdentityLinOp(X)`` represents the identity operator ``I_X : X -> X``. The
    context is resolved from the optional ``ctx`` argument and the space, and the
    resulting operator has domain and codomain equal to ``X`` in that context.

    The forward action is ``apply(x) = x`` for ``x in X``. The reverse action is
    ``rapply(x) = x`` for ``x in X``.

    Parameters
    ----------
    space : Space
        Domain and codomain space.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from ``space``.
    """

    def __init__(self, space: Domain, ctx: Context | str | None = None) -> None:
        super().__init__(space, space, ctx)

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``x`` after domain validation."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        """Return ``x`` without membership checks."""
        return x

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, x: Any) -> Any:
        """Return ``x`` after codomain validation."""
        return self._rapply_unchecked(x)

    def _rapply_unchecked(self, x: Any) -> Any:
        """Return ``x`` without membership checks."""
        return x

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``xs`` after batched domain validation."""
        return xs

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, xs: Any) -> Any:
        """Return ``xs`` after batched codomain validation."""
        return xs

    def to_dense(self) -> Any:
        """
        Return the dense tensor representation of this identity map.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        size = 1
        for dim in self.domain.shape:
            size *= dim
        eye = self.ops.eye(size, dtype=self.dtype)
        return self.ops.reshape(eye, tuple(self.codomain.shape) + tuple(self.domain.shape))

    def is_hermitian(self) -> bool:
        """
        Return whether this identity operator is Hermitian.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    def __eq__(self, other: Any) -> bool:
        """Return whether another identity map has the same space."""
        if type(other) is type(self):
            return self.domain == other.domain
        return False

    def tree_flatten(self):
        """Flatten this operator for pytree registration."""
        children = ()
        aux = (self.domain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild this operator from pytree data."""
        domain, ctx = aux
        return cls(domain, ctx)

    def _convert(self, new_ctx: Context) -> IdentityLinOp:
        """Convert the identity space to ``new_ctx``."""
        return IdentityLinOp(self.domain.convert(new_ctx), new_ctx)


@jax_pytree_class
class MatrixFreeLinOp(LinOp[Domain, Codomain]):
    """
    Linear operator defined by user-supplied forward and reverse callables.

    ``MatrixFreeLinOp(apply, rapply, X, Y)`` represents a matrix-free map
    ``A : X -> Y`` without storing or materializing a matrix. The context is
    resolved from the optional ``ctx`` argument and the spaces, then the spaces
    are converted to that context.

    The forward action is ``apply(x) = apply_fn(x)`` for ``x in X``. The
    reverse action is ``rapply(y) = rapply_fn(y)`` for ``y in Y``. The supplied
    ``rapply`` callable must already be the true adjoint with respect to the
    declared domain and codomain inner products:
    ``<apply(x), y>_Y = <x, rapply(y)>_X``. It is not automatically corrected
    with Riesz maps. For non-Euclidean spaces, use
    :meth:`from_coordinate_adjoint` when you have a Euclidean coordinate
    adjoint. When checks are enabled, inputs and callable outputs are validated
    against the corresponding domain and codomain.

    Parameters
    ----------
    apply : callable
        Callable with signature ``apply(x: Any) -> Any`` implementing the
        forward map from ``dom`` to ``cod``.
    rapply : callable
        Callable with signature ``rapply(y: Any) -> Any`` implementing the
        true space adjoint map from ``cod`` back to ``dom``. For
        non-Euclidean spaces this is generally not the same as the Euclidean
        coordinate adjoint.
    dom : Space
        Domain space containing valid inputs for ``apply`` and outputs from
        ``rapply``.
    cod : Space
        Codomain space containing outputs from ``apply`` and valid inputs for
        ``rapply``.
    ctx : Context, str, or None, optional
        Optional context specification. An explicit context wins over inferred
        contexts from ``dom`` and ``cod``.
    vapply : callable or None, optional
        Optional callable with signature ``vapply(xs: Any) -> Any`` for batched
        forward application. If omitted, backend ``vmap`` fallback is used.
    rvapply : callable or None, optional
        Optional callable with signature ``rvapply(ys: Any) -> Any`` for
        batched adjoint application. If omitted, backend ``vmap`` fallback is
        used.

    Returns
    -------
    MatrixFreeLinOp
        Operator using the supplied callables for forward, adjoint, and
        optionally batched actions.
    """

    def __init__(
        self,
        apply: Callable[[Any], Any],
        rapply: Callable[[Any], Any],
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
        vapply: Callable[[Any], Any] | None = None,
        rvapply: Callable[[Any], Any] | None = None,
        *,
        _uses_coordinate_adjoint: bool = False,
        _coordinate_rapply_fn: Callable[[Any], Any] | None = None,
        _coordinate_rvapply_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        """
        Initialize a matrix-free linear operator.

        Parameters
        ----------
        apply:
            Callable ``apply(x)`` that accepts an element of ``dom`` and returns
            an element of ``cod``.
        rapply:
            Callable ``rapply(y)`` that accepts an element of ``cod`` and
            returns an element of ``dom``.
        dom:
            Domain space of the operator.
        cod:
            Codomain space of the operator.
        ctx:
            Optional context specification for the operator and converted
            spaces.
        vapply:
            Optional callable for batched forward application over ``dom``
            batches.
        rvapply:
            Optional callable for batched adjoint application over ``cod``
            batches.

        Returns
        -------
        None
            The initializer stores the callables and converted spaces on
            ``self``.
        """
        if not callable(apply):
            raise TypeError(f"apply must be callable, got {type(apply).__name__}.")
        if not callable(rapply):
            raise TypeError(f"rapply must be callable, got {type(rapply).__name__}.")
        if vapply is not None and not callable(vapply):
            raise TypeError(f"vapply must be callable, got {type(vapply).__name__}.")
        if rvapply is not None and not callable(rvapply):
            raise TypeError(f"rvapply must be callable, got {type(rvapply).__name__}.")
        super().__init__(dom, cod, ctx)
        self.apply_fn = apply
        self.rapply_fn = rapply
        self.vapply_fn = vapply
        self.rvapply_fn = rvapply
        self._uses_coordinate_adjoint = bool(_uses_coordinate_adjoint)
        if self._uses_coordinate_adjoint and _coordinate_rapply_fn is None:
            raise ValueError(
                "MatrixFreeLinOp coordinate-adjoint construction requires "
                "_coordinate_rapply_fn metadata."
            )
        if (
            not self._uses_coordinate_adjoint
            and (_coordinate_rapply_fn is not None or _coordinate_rvapply_fn is not None)
        ):
            raise ValueError(
                "MatrixFreeLinOp direct-adjoint construction cannot store "
                "coordinate-adjoint metadata."
            )
        self._coordinate_rapply_fn = _coordinate_rapply_fn
        self._coordinate_rvapply_fn = _coordinate_rvapply_fn

    @classmethod
    def from_coordinate_adjoint(
        cls,
        apply: Callable[[Any], Any],
        coordinate_rapply: Callable[[Any], Any],
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
        vapply: Callable[[Any], Any] | None = None,
        coordinate_rvapply: Callable[[Any], Any] | None = None,
    ) -> MatrixFreeLinOp:
        r"""
        Build a matrix-free operator from a Euclidean coordinate adjoint.

        ``coordinate_rapply`` implements the coordinate adjoint
        :math:`A^\dagger`. This constructor wraps it with the spaces' Riesz
        maps to form the true adjoint
        :math:`A^\sharp y = R_X^{-1} A^\dagger R_Y y`. The forward callable
        still defines the coordinate action ``A x``.

        Parameters
        ----------
        apply : callable
            Forward coordinate action from ``dom`` to ``cod``.
        coordinate_rapply : callable
            Euclidean coordinate adjoint from ``cod`` dual coordinates to
            ``dom`` dual coordinates.
        dom, cod : Space
            Domain and codomain spaces. Non-Euclidean spaces must provide
            usable Riesz maps.
        ctx : Context, str, or None, optional
            Optional context specification.
        vapply : callable or None, optional
            Optional batched forward application.
        coordinate_rvapply : callable or None, optional
            Optional batched Euclidean coordinate adjoint. If omitted, batched
            adjoints use backend ``vmap`` over the wrapped scalar ``rapply``.
        """
        if not callable(apply):
            raise TypeError(f"apply must be callable, got {type(apply).__name__}.")
        if not callable(coordinate_rapply):
            raise TypeError(
                "coordinate_rapply must be callable, "
                f"got {type(coordinate_rapply).__name__}."
            )

        resolved_ctx = resolve_context_priority(ctx, dom, cod)
        dom = dom.convert(resolved_ctx)
        cod = cod.convert(resolved_ctx)
        _requires_euclidean_or_riesz(dom, cod, "MatrixFreeLinOp.from_coordinate_adjoint")

        def rapply(y):
            yd = cod.riesz(y)
            x_dual = coordinate_rapply(yd)
            return dom.riesz_inverse(x_dual)

        rvapply = None
        if coordinate_rvapply is not None:
            if not callable(coordinate_rvapply):
                raise TypeError(
                    "coordinate_rvapply must be callable, "
                    f"got {type(coordinate_rvapply).__name__}."
                )

            def rvapply(ys):
                yd = cod.riesz(ys)
                x_dual = coordinate_rvapply(yd)
                return dom.riesz_inverse(x_dual)

        return cls(
            apply,
            rapply,
            dom,
            cod,
            resolved_ctx,
            vapply,
            rvapply,
            _uses_coordinate_adjoint=True,
            _coordinate_rapply_fn=coordinate_rapply,
            _coordinate_rvapply_fn=coordinate_rvapply,
        )

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """
        Apply the forward callable.

        Parameters
        ----------
        x:
            Element of ``self.domain`` passed to ``apply_fn``.

        Returns
        -------
        Any
            Element of ``self.codomain`` returned by ``apply_fn``.
        """
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        """
        Apply ``apply_fn`` without membership checks.

        Parameters
        ----------
        x:
            Value accepted by the user-supplied forward callable.

        Returns
        -------
        Any
            Raw forward-callable output.
        """
        return self.apply_fn(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """
        Apply the adjoint callable.

        Parameters
        ----------
        y:
            Element of ``self.codomain`` passed to ``rapply_fn``.

        Returns
        -------
        Any
            Element of ``self.domain`` returned by ``rapply_fn``.
        """
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        """
        Apply ``rapply_fn`` without membership checks.

        Parameters
        ----------
        y:
            Value accepted by the user-supplied adjoint callable.

        Returns
        -------
        Any
            Raw adjoint-callable output.
        """
        return self.rapply_fn(y)

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, xs: Any) -> Any:
        """
        Apply this operator to a batch of domain elements.

        Parameters
        ----------
        xs:
            Batched element of ``self.domain``.
        Returns
        -------
        Any
            Batched element of ``self.codomain`` produced by ``vapply_fn`` or
            by the fallback batching implementation.
        """
        if self.vapply_fn is None:
            return super().vapply(xs)
        return self.vapply_fn(xs)

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """
        Apply the adjoint operator to a batch of codomain elements.

        Parameters
        ----------
        ys:
            Batched element of ``self.codomain``.
        Returns
        -------
        Any
            Batched element of ``self.domain`` produced by ``rvapply_fn`` or by
            the fallback batching implementation.
        """
        if self.rvapply_fn is None:
            return super().rvapply(ys)
        return self.rvapply_fn(ys)

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            base_equal = (
                self.domain == other.domain
                and self.codomain == other.codomain
                and self.apply_fn is other.apply_fn
                and self.vapply_fn is other.vapply_fn
                and self._uses_coordinate_adjoint == other._uses_coordinate_adjoint
            )
            if not base_equal:
                return False
            if self._uses_coordinate_adjoint:
                return (
                    self._coordinate_rapply_fn is other._coordinate_rapply_fn
                    and self._coordinate_rvapply_fn is other._coordinate_rvapply_fn
                )
            return self.rapply_fn is other.rapply_fn and self.rvapply_fn is other.rvapply_fn
        return False

    def tree_flatten(self):
        children = ()
        if self._uses_coordinate_adjoint:
            aux = (
                self.apply_fn,
                None,
                self.domain,
                self.codomain,
                self.ctx,
                self.vapply_fn,
                None,
                True,
                self._coordinate_rapply_fn,
                self._coordinate_rvapply_fn,
            )
            return children, aux
        aux = (
            self.apply_fn,
            self.rapply_fn,
            self.domain,
            self.codomain,
            self.ctx,
            self.vapply_fn,
            self.rvapply_fn,
            False,
            None,
            None,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            apply_fn,
            rapply_fn,
            domain,
            codomain,
            ctx,
            vapply_fn,
            rvapply_fn,
            uses_coordinate_adjoint,
            coordinate_rapply_fn,
            coordinate_rvapply_fn,
        ) = aux
        if uses_coordinate_adjoint:
            return cls.from_coordinate_adjoint(
                apply_fn,
                coordinate_rapply_fn,
                domain,
                codomain,
                ctx,
                vapply_fn,
                coordinate_rvapply_fn,
            )
        return cls(apply_fn, rapply_fn, domain, codomain, ctx, vapply_fn, rvapply_fn)

    def _convert(self, new_ctx: Context) -> MatrixFreeLinOp:
        """
        Convert this matrix-free operator to ``new_ctx``.

        Parameters
        ----------
        new_ctx:
            Concrete target context for converted domain and codomain spaces.

        Returns
        -------
        MatrixFreeLinOp
            Operator with converted spaces and the same user-supplied
            callables.
        """
        if self._uses_coordinate_adjoint:
            return MatrixFreeLinOp.from_coordinate_adjoint(
                self.apply_fn,
                self._coordinate_rapply_fn,
                self.domain.convert(new_ctx),
                self.codomain.convert(new_ctx),
                new_ctx,
                self.vapply_fn,
                self._coordinate_rvapply_fn,
            )
        return MatrixFreeLinOp(
            self.apply_fn,
            self.rapply_fn,
            self.domain.convert(new_ctx),
            self.codomain.convert(new_ctx),
            new_ctx,
            self.vapply_fn,
            self.rvapply_fn,
        )


@jax_pytree_class
class _AdjointViewLinOp(LinOp[Codomain, Domain]):
    """
    Hermitian-adjoint view of a linear operator.

    ``A.H`` represents the adjoint view ``A*``. Its context is exactly
    ``A.ctx``; its domain is ``A.codomain`` and its codomain is ``A.domain``.
    ``A.H.H`` returns ``A`` rather than constructing another wrapper.

    The forward action is ``apply(y) = A.rapply(y)`` for ``y in A.codomain``.
    The reverse action is ``rapply(x) = A.apply(x)`` for ``x in A.domain``.
    """

    def __init__(self, op: LinOp[Domain, Codomain]) -> None:
        op = _require_linop(op, "op")
        super().__init__(op.codomain, op.domain, op.ctx)
        self.op = op

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, y: Any) -> Any:
        """Return ``op.rapply(y)``."""
        return self.op.rapply(y)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, x: Any) -> Any:
        """Return ``op.apply(x)``."""
        return self.op.apply(x)

    def vapply(self, ys: Any) -> Any:
        """Return ``op.rvapply(ys)`` over a batch."""
        return self.op.rvapply(ys)

    def rvapply(self, xs: Any) -> Any:
        """Return ``op.vapply(xs)`` over a batch."""
        return self.op.vapply(xs)

    @property
    def H(self) -> LinOp[Domain, Codomain]:
        """Original operator viewed as the adjoint of this adjoint view."""
        return self.op

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.op == other.op
        return False

    def tree_flatten(self):
        children = (self.op,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])

    def _convert(self, new_ctx: Context) -> _AdjointViewLinOp:
        return _AdjointViewLinOp(self.op.convert(new_ctx))
