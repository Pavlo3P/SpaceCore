from __future__ import annotations

from numbers import Number
from typing import Any, Sequence

from ._base import LinOp, Domain, Codomain
from ..backend import Context, jax_pytree_class


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
    if hasattr(value, "conjugate"):
        return value.conjugate()
    if hasattr(value, "conj"):
        return value.conj()
    return value


def _same_context(left: LinOp, right: LinOp) -> bool:
    return (
        left.ctx == right.ctx
        and left.ctx.dtype == right.ctx.dtype
        and left.ctx.enable_checks == right.ctx.enable_checks
    )


def _require_same_context(ops: Sequence[LinOp]) -> Context:
    ctx = ops[0].ctx
    for i, op in enumerate(ops[1:], start=1):
        if not _same_context(ops[0], op):
            raise ValueError(
                "All LinOp operands in an algebraic expression must have the same ctx; "
                f"operand 0 has ctx {ctx!r}, operand {i} has ctx {op.ctx!r}."
            )
    return ctx


def _require_linop(op: Any, name: str) -> LinOp:
    if not isinstance(op, LinOp):
        raise TypeError(f"{name} must be a LinOp, got {type(op).__name__}.")
    return op


def _scalar_equal(value: Any, target: Any) -> bool:
    try:
        return bool(value == target)
    except Exception:
        return False


def _is_zero_scalar(value: Any) -> bool:
    return _scalar_equal(value, 0)


def _is_one_scalar(value: Any) -> bool:
    return _scalar_equal(value, 1)


def _flatten_sum_terms(ops: Sequence[LinOp]) -> tuple[LinOp, ...]:
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
    """
    if not ops:
        raise ValueError("make_sum requires a nonempty sequence of LinOp operands.")

    terms = _flatten_sum_terms(ops)
    ctx = _require_same_context(terms)
    domain = terms[0].domain
    codomain = terms[0].codomain
    for i, op in enumerate(terms[1:], start=1):
        if op.domain != domain or op.codomain != codomain:
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
    """
    left = _require_linop(left, "left")
    right = _require_linop(right, "right")
    _require_same_context((left, right))
    if right.codomain != left.domain:
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
    """
    Lazy scalar multiple of a linear operator.

    ``ScaledLinOp(alpha, A)`` represents the mathematical operator
    ``alpha * A``. Its context is exactly ``A.ctx``; its domain is ``A.domain``
    and its codomain is ``A.codomain``. No dense matrix representation is
    formed.

    The forward action is ``apply(x) = alpha * A.apply(x)`` for
    ``x in A.domain``. The reverse action is
    ``rapply(y) = conj(alpha) * A.rapply(y)`` for ``y in A.codomain``, so
    complex scalars use the conjugated coefficient.
    """

    def __init__(self, scalar: Any, op: LinOp[Domain, Codomain]) -> None:
        op = _require_linop(op, "op")
        if not is_scalar_like(scalar):
            raise TypeError(f"scalar must be scalar-like, got {type(scalar).__name__}.")
        super().__init__(op.domain, op.codomain, op.ctx)
        self.scalar = scalar
        self.op = op

    def apply(self, x: Any) -> Any:
        """Return ``scalar * op.apply(x)``."""
        return self.scalar * self.op.apply(x)

    def rapply(self, y: Any) -> Any:
        """Return ``conj(scalar) * op.rapply(y)``."""
        return _conjugate_scalar(self.scalar) * self.op.rapply(y)

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.scalar == other.scalar and self.op == other.op
        return False

    def tree_flatten(self):
        children = (self.scalar, self.op)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        scalar, op = children
        return cls(scalar, op)

    def _convert(self, new_ctx: Context) -> ScaledLinOp:
        return ScaledLinOp(self.scalar, self.op.convert(new_ctx))


@jax_pytree_class
class SumLinOp(LinOp[Domain, Codomain]):
    """
    Lazy finite sum of linear operators with common spaces.

    ``SumLinOp((A1, ..., Ak))`` represents ``A1 + ... + Ak`` for a nonempty
    sequence of ``LinOp`` instances. All operands must have the same ``ctx``,
    the same domain, and the same codomain before construction. The resulting
    operator has that shared context, domain, and codomain.

    The forward action is ``apply(x) = sum_i Ai.apply(x)`` for the shared
    domain element ``x``. The reverse action is
    ``rapply(y) = sum_i Ai.rapply(y)`` for the shared codomain element ``y``.
    """

    def __init__(self, ops: Sequence[LinOp[Domain, Codomain]]) -> None:
        if not ops:
            raise ValueError("SumLinOp requires a nonempty sequence of LinOp operands.")
        parts = tuple(_require_linop(op, f"ops[{i}]") for i, op in enumerate(ops))
        ctx = _require_same_context(parts)
        domain = parts[0].domain
        codomain = parts[0].codomain
        for i, op in enumerate(parts[1:], start=1):
            if op.domain != domain or op.codomain != codomain:
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

    def apply(self, x: Any) -> Any:
        """Return ``sum_i ops[i].apply(x)``."""
        acc = self.ops_tuple[0].apply(x)
        for op in self.ops_tuple[1:]:
            acc = self.codomain.add(acc, op.apply(x))
        return acc

    def rapply(self, y: Any) -> Any:
        """Return ``sum_i ops[i].rapply(y)``."""
        acc = self.ops_tuple[0].rapply(y)
        for op in self.ops_tuple[1:]:
            acc = self.domain.add(acc, op.rapply(y))
        return acc

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.ops_tuple == other.ops_tuple
        return False

    def tree_flatten(self):
        children = self.ops_tuple
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(tuple(children))

    def _convert(self, new_ctx: Context) -> SumLinOp:
        return SumLinOp(tuple(op.convert(new_ctx) for op in self.ops_tuple))


@jax_pytree_class
class ComposedLinOp(LinOp[Domain, Codomain]):
    """
    Lazy composition of two linear operators.

    ``ComposedLinOp(A, B)`` represents ``A @ B = A circ B``. The operands must
    have the same ``ctx`` before construction, and ``B.codomain`` must equal
    ``A.domain``. The resulting operator has domain ``B.domain`` and codomain
    ``A.codomain``.

    The forward action is ``apply(x) = A.apply(B.apply(x))`` for
    ``x in B.domain``. The reverse action is ``rapply(z) = B.rapply(A.rapply(z))``
    for ``z in A.codomain``.
    """

    def __init__(self, left: LinOp, right: LinOp) -> None:
        left = _require_linop(left, "left")
        right = _require_linop(right, "right")
        _require_same_context((left, right))
        if right.codomain != left.domain:
            raise ValueError(
                "ComposedLinOp requires right.codomain == left.domain; "
                f"got {right.codomain!r} and {left.domain!r}."
            )
        super().__init__(right.domain, left.codomain, left.ctx)
        self.left = left
        self.right = right

    def apply(self, x: Any) -> Any:
        """Return ``left.apply(right.apply(x))``."""
        return self.left.apply(self.right.apply(x))

    def rapply(self, z: Any) -> Any:
        """Return ``right.rapply(left.rapply(z))``."""
        return self.right.rapply(self.left.rapply(z))

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.left == other.left and self.right == other.right
        return False

    def tree_flatten(self):
        children = (self.left, self.right)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        left, right = children
        return cls(left, right)

    def _convert(self, new_ctx: Context) -> ComposedLinOp:
        return ComposedLinOp(self.left.convert(new_ctx), self.right.convert(new_ctx))


@jax_pytree_class
class ZeroLinOp(LinOp[Domain, Codomain]):
    """
    Lazy zero map between two spaces.

    ``ZeroLinOp(X, Y)`` represents the linear map ``0 : X -> Y``. The context is
    resolved from the optional ``ctx`` argument and the two spaces, then both
    spaces are converted to that context. Its domain is ``X`` and its codomain
    is ``Y`` in the resolved context.

    The forward action is ``apply(x) = 0_Y`` for ``x in X``. The reverse action
    is ``rapply(y) = 0_X`` for ``y in Y``.
    """

    def __init__(
        self,
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
    ) -> None:
        super().__init__(dom, cod, ctx)

    def apply(self, x: Any) -> Any:
        """Return the zero element of the codomain."""
        if self._enable_checks:
            self.domain._check_member(x)
        return self.codomain.zeros()

    def rapply(self, y: Any) -> Any:
        """Return the zero element of the domain."""
        if self._enable_checks:
            self.codomain._check_member(y)
        return self.domain.zeros()

    def to_dense(self) -> Any:
        """
        Return the dense tensor representation of the zero map.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        return self.ops.zeros(tuple(self.codomain.shape) + tuple(self.domain.shape), dtype=self.dtype)

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.domain == other.domain and self.codomain == other.codomain
        return False

    def tree_flatten(self):
        children = ()
        aux = (self.domain, self.codomain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, codomain, ctx = aux
        return cls(domain, codomain, ctx)

    def _convert(self, new_ctx: Context) -> ZeroLinOp:
        return ZeroLinOp(self.domain.convert(new_ctx), self.codomain.convert(new_ctx), new_ctx)


@jax_pytree_class
class IdentityLinOp(LinOp[Domain, Domain]):
    """
    Lazy identity map on a space.

    ``IdentityLinOp(X)`` represents the identity operator ``I_X : X -> X``. The
    context is resolved from the optional ``ctx`` argument and the space, and the
    resulting operator has domain and codomain equal to ``X`` in that context.

    The forward action is ``apply(x) = x`` for ``x in X``. The reverse action is
    ``rapply(x) = x`` for ``x in X``.
    """

    def __init__(self, space: Domain, ctx: Context | str | None = None) -> None:
        super().__init__(space, space, ctx)

    def apply(self, x: Any) -> Any:
        """Return ``x`` after domain validation."""
        if self._enable_checks:
            self.domain._check_member(x)
        return x

    def rapply(self, x: Any) -> Any:
        """Return ``x`` after codomain validation."""
        if self._enable_checks:
            self.codomain._check_member(x)
        return x

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

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return self.domain == other.domain
        return False

    def tree_flatten(self):
        children = ()
        aux = (self.domain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, ctx = aux
        return cls(domain, ctx)

    def _convert(self, new_ctx: Context) -> IdentityLinOp:
        return IdentityLinOp(self.domain.convert(new_ctx), new_ctx)


@jax_pytree_class
class MatrixFreeLinOp(LinOp[Domain, Codomain]):
    """
    Linear operator defined by user-supplied forward and reverse callables.

    ``MatrixFreeLinOp(apply, rapply, X, Y)`` represents a matrix-free map
    ``A : X -> Y`` without storing or materializing a matrix. The context is
    resolved from the optional ``ctx`` argument and the spaces, then the spaces
    are converted to that context.

    The forward action is ``apply(x) = apply_fn(x)`` for ``x in X``. The reverse
    action is ``rapply(y) = rapply_fn(y)`` for ``y in Y``. When checks are
    enabled, inputs and callable outputs are validated against the corresponding
    domain and codomain.
    """

    def __init__(
        self,
        apply: Any,
        rapply: Any,
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
    ) -> None:
        if not callable(apply):
            raise TypeError(f"apply must be callable, got {type(apply).__name__}.")
        if not callable(rapply):
            raise TypeError(f"rapply must be callable, got {type(rapply).__name__}.")
        super().__init__(dom, cod, ctx)
        self.apply_fn = apply
        self.rapply_fn = rapply

    def apply(self, x: Any) -> Any:
        """Return ``apply_fn(x)``."""
        if self._enable_checks:
            self.domain._check_member(x)
        y = self.apply_fn(x)
        if self._enable_checks:
            self.codomain._check_member(y)
        return y

    def rapply(self, y: Any) -> Any:
        """Return ``rapply_fn(y)``."""
        if self._enable_checks:
            self.codomain._check_member(y)
        x = self.rapply_fn(y)
        if self._enable_checks:
            self.domain._check_member(x)
        return x

    def __eq__(self, other: Any) -> bool:
        if type(other) is type(self):
            return (
                self.domain == other.domain
                and self.codomain == other.codomain
                and self.apply_fn is other.apply_fn
                and self.rapply_fn is other.rapply_fn
            )
        return False

    def tree_flatten(self):
        children = ()
        aux = (self.apply_fn, self.rapply_fn, self.domain, self.codomain, self.ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        apply_fn, rapply_fn, domain, codomain, ctx = aux
        return cls(apply_fn, rapply_fn, domain, codomain, ctx)

    def _convert(self, new_ctx: Context) -> MatrixFreeLinOp:
        return MatrixFreeLinOp(
            self.apply_fn,
            self.rapply_fn,
            self.domain.convert(new_ctx),
            self.codomain.convert(new_ctx),
            new_ctx,
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

    def apply(self, y: Any) -> Any:
        """Return ``op.rapply(y)``."""
        return self.op.rapply(y)

    def rapply(self, x: Any) -> Any:
        """Return ``op.apply(x)``."""
        return self.op.apply(x)

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
