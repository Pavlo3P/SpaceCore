from __future__ import annotations

import warnings
from typing import Any, Callable, Sequence, cast

from ._base import LinOp, Domain, Codomain
from ..mapping import (
    Scalar, ScaledNode, SumNode, ZeroNode, make_scaled_of, make_sum_of, register,
    same_space_for_algebra as _same_space_for_algebra,
)
from ..space import DenseElement, Element
from ._metric import _requires_euclidean_or_riesz, metric_rapply, metric_rvapply
from .._checks import CheckLevel, minimum_check_level
from .._checks import checked_method
from ..space.base import CoordinateSpace, InnerProductSpace
from ..space._capabilities import _CAP_BATCH, _CAP_COORDINATE, probe_element, require
from .._repr import summarize_value
from ..contextual import Context
from .._lazy_algebra import conjugate_scalar as _conjugate_scalar
from .._lazy_algebra import (
    is_scalar_like as is_scalar_like,  # re-exported for linop/_base.py
)


def _compose_chain(op: Any) -> tuple:
    """Flatten ``op`` into its leaf operators in application order.

    This is the *fusion rule*: a composed operator exposes a precomputed
    ``_apply_chain`` (its leaves, first applied first); every other operator
    contributes itself as a single-element chain. Concatenating the chains of a
    composition's operands fuses an arbitrarily deep ``A @ B @ C @ ...`` into one
    flat sequence, so the composed core loops once instead of re-walking the
    nested binary tree.
    """
    chain = getattr(op, "_apply_chain", None)
    return chain if chain is not None else (op,)


def _leading_shape(space: Any, value: Any) -> tuple[int, ...]:
    """Infer the leading batch dimensions of ``value`` relative to ``space``."""
    # Product spaces: batch shape of the first component. Else strip the trailing space shape.
    parts = getattr(space, "spaces", None)
    if parts is not None and isinstance(value, tuple) and value:
        return _leading_shape(parts[0], value[0])
    shape = tuple(getattr(value, "shape", ()))
    base = tuple(space.shape)
    return shape if not base else shape[: len(shape) - len(base)]


def _batched_zeros(space: Any, leading: tuple[int, ...]) -> Any:
    """Return a batched zero element of ``space`` with leading shape ``leading``."""
    # Product spaces get a tuple of per-component zero batches.
    parts = getattr(space, "spaces", None)
    if parts is not None:
        return tuple(_batched_zeros(part, leading) for part in parts)
    return space.ops.zeros(leading + tuple(space.shape), dtype=space.dtype)


#: Sentinel for "the caller did not mention ``euclidean_adjoint`` at all".
#: Passing ``False`` explicitly is an assertion that ``rapply`` already carries the
#: geometry, so it silences the advisory; omitting it means the question was never
#: considered, which is the only case worth warning about.
_ADJOINT_UNSPECIFIED = object()


def _require_same_context(ops: Sequence[LinOp]) -> Context:
    """Return the common mathematical context for algebra operands or raise.

    Operands must share a mathematical context (:meth:`Context.same_math`). The
    surviving ``check_level`` is a property of the resulting bound object and is
    combined (via the minimum) when the container binds its operands, not here.
    """
    # Every operand must share operand 0's math context.
    ctx = ops[0].ctx
    for i, op in enumerate(ops[1:], start=1):
        if not ctx.same_math(op.ctx):
            raise ValueError(
                "All LinOp operands in an algebraic expression must have the same ctx; "
                f"operand 0 has ctx {ctx!r}, operand {i} has ctx {op.ctx!r}."
            )
    return ctx


def _require_linop(op: Any, name: str) -> LinOp:
    """Return ``op`` as a linear operator or raise a typed error."""
    if not isinstance(op, LinOp):
        raise TypeError(f"{name} must be a LinOp, got {type(op).__name__}.")
    return op


def make_sum(ops: Sequence[LinOp]) -> LinOp:
    """
    Return the lazy sum of operators, with nested sums flattened and zero terms dropped.

    Parameters
    ----------
    ops : sequence of LinOp
        Operators sharing context, domain and codomain.

    Returns
    -------
    LinOp
        The sum, the single remaining operand, or a zero operator.
    """
    return make_sum_of(LinOp, ops)


def make_scaled(scalar: Any, op: LinOp) -> LinOp:
    """
    Return ``scalar * op``: ``0`` gives zero, ``1`` gives ``op``, nested scalars fold.

    Parameters
    ----------
    scalar : scalar-like
        Coefficient.
    op : LinOp
        Operator to scale.

    Returns
    -------
    LinOp
        The scaled operator.
    """
    return make_scaled_of(LinOp, scalar, op)


def make_composed(left: LinOp, right: LinOp) -> LinOp:
    """
    Return ``left @ right``, dropping identity factors and collapsing zero factors.

    Parameters
    ----------
    left : LinOp
        Operator applied second.
    right : LinOp
        Operator applied first; ``right.codomain`` must equal ``left.domain``.

    Returns
    -------
    LinOp
        The composition.
    """
    # Validate: both LinOps, same context, matching middle space.
    left = _require_linop(left, "left")
    right = _require_linop(right, "right")
    _require_same_context((left, right))
    if not _same_space_for_algebra(right.codomain, left.domain):
        raise ValueError(
            "ComposedLinOp requires right.codomain == left.domain; "
            f"got {right.codomain!r} and {left.domain!r}."
        )

    # Local rewrites: identity factors vanish, zero factors give a zero map.
    if isinstance(right, IdentityLinOp):
        return left
    if isinstance(left, IdentityLinOp):
        return right
    if isinstance(left, ZeroLinOp):
        return ZeroLinOp(right.domain, left.codomain, left.ctx)
    if isinstance(right, ZeroLinOp):
        return ZeroLinOp(right.domain, left.codomain, left.ctx)
    return ComposedLinOp(left, right)


class ScaledLinOp(ScaledNode, LinOp[Domain, Codomain]):
    r"""
    Lazy ``scalar * op``; its adjoint is ``conj(scalar) * op.H``.

    Parameters
    ----------
    scalar : scalar-like
        Coefficient.
    op : LinOp
        Operator to scale.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def _apply_core(self, x: Any) -> Any:
        y = self.op._apply_core(x)
        scale = getattr(self.codomain, "_scale_core", self.codomain.scale)
        return scale(self.scalar, y)

    def _rapply_core(self, y: Any) -> Any:
        x = self.op._rapply_core(y)
        scale = getattr(self.domain, "_scale_core", self.domain.scale)
        return scale(_conjugate_scalar(self.scalar), x)

    def _vapply_core(self, xs: Any) -> Any:
        ys = self.op._vapply_core(xs)
        return self.codomain.scale_batch(self.scalar, ys)

    def __init__(self, scalar: Any, op: LinOp[Domain, Codomain], check_level: CheckLevel | None = None) -> None:
        super().__init__(scalar, op, check_level)

    @property
    def op(self) -> LinOp[Domain, Codomain]:
        """The scaled operand."""
        return self.operand

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``scalar * op.apply(x)``."""
        return self._apply_core(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Return ``conj(scalar) * op.rapply(y)``."""
        return self._rapply_core(y)

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``scalar * op.vapply(xs)``."""
        require(self.domain, _CAP_BATCH, "ScaledLinOp.vapply")
        require(self.codomain, _CAP_BATCH, "ScaledLinOp.vapply")
        return self._vapply_core(xs)

    def rvapply(self, ys: Any) -> Any:
        """Return ``conj(scalar) * op.rvapply(ys)``."""
        require(self.domain, _CAP_BATCH, "ScaledLinOp.rvapply")
        require(self.codomain, _CAP_BATCH, "ScaledLinOp.rvapply")
        # Adjoint of s A is conj(s) A*.
        xs = self.op.rvapply(ys)
        return self.domain.scale_batch(_conjugate_scalar(self.scalar), xs)

    def fuse(self, *, materialize: bool = False) -> LinOp:
        r"""Fuse the operand and fold the scalar into a dense matrix (ADR-021).

        When the fused operand is dense, replace ``c · A`` with one
        :class:`DenseLinOp` holding ``c · M_A``; otherwise keep the scaling lazy,
        so a matrix-free operand is never densified. Adjoint-consistent: the
        fused operator's adjoint is ``conj(c) · M_A^*`` (with the metric), exactly
        the lazy ``ScaledLinOp`` adjoint.
        """
        from ._dense import DenseLinOp

        # Fuse the operand; a dense result absorbs the scalar into its matrix.
        op = self.op.fuse(materialize=materialize)
        if isinstance(op, DenseLinOp):
            matrix = self.scalar * op.to_matrix()
            tensor = self.ops.reshape(
                matrix, tuple(op.codomain.shape) + tuple(op.domain.shape)
            )
            return DenseLinOp(tensor, op.domain, op.codomain, self.ctx)
        return make_scaled(self.scalar, op)

    def is_hermitian(self) -> bool | None:
        """
        Return ``op.is_hermitian()`` for a real scalar, else ``None`` (undecided).

        Returns
        -------
        bool | None
            Structural Hermiticity, or ``None`` when unknown.
        """
        if _conjugate_scalar(self.scalar) == self.scalar:
            return self.op.is_hermitian()
        return None

    def _repr_body(self) -> str:
        return f"{summarize_value(self.scalar)} · {self.op._short_repr()}"


class SumLinOp(SumNode, LinOp[Domain, Codomain]):
    r"""
    Lazy sum of operators sharing context, domain and codomain.

    Parameters
    ----------
    ops : sequence of LinOp
        The summands.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def _apply_core(self, x: Any) -> Any:
        # A1 x + ... + Ak x, with the codomain's unchecked add when it has one.
        parts = self.ops_tuple
        add = getattr(self.codomain, "_add_core", self.codomain.add)
        acc = parts[0]._apply_core(x)
        for part in parts[1:]:
            acc = add(acc, part._apply_core(x))
        return acc

    def _rapply_core(self, y: Any) -> Any:
        # A1* y + ... + Ak* y, added in the domain.
        parts = self.ops_tuple
        add = getattr(self.domain, "_add_core", self.domain.add)
        acc = parts[0]._rapply_core(y)
        for part in parts[1:]:
            acc = add(acc, part._rapply_core(y))
        return acc

    def _vapply_core(self, xs: Any) -> Any:
        # Row-wise sum of the batched applies.
        parts = self.ops_tuple
        add_batch = self.codomain.add_batch
        acc = parts[0]._vapply_core(xs)
        for part in parts[1:]:
            acc = add_batch(acc, part._vapply_core(xs))
        return acc

    def __init__(self, ops: Sequence[LinOp[Domain, Codomain]], check_level: CheckLevel | None = None) -> None:
        super().__init__(ops, check_level)

    @property
    def ops_tuple(self) -> tuple[LinOp[Domain, Codomain], ...]:
        """The summed operators (alias of ``parts``)."""
        return self.parts

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``sum_i ops[i].apply(x)``."""
        return self._apply_core(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Return ``sum_i ops[i].rapply(y)``."""
        return self._rapply_core(y)

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``sum_i ops[i].vapply(xs)``."""
        require(self.domain, _CAP_BATCH, "SumLinOp.vapply")
        require(self.codomain, _CAP_BATCH, "SumLinOp.vapply")
        return self._vapply_core(xs)

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Return ``sum_i ops[i].rvapply(ys)``."""
        require(self.domain, _CAP_BATCH, "SumLinOp.rvapply")
        require(self.codomain, _CAP_BATCH, "SumLinOp.rvapply")
        # Adjoint of a sum is the sum of adjoints.
        add_batch = self.domain.add_batch
        acc = self.ops_tuple[0].rvapply(ys)
        for op in self.ops_tuple[1:]:
            acc = add_batch(acc, op.rvapply(ys))
        return acc

    def fuse(self, *, materialize: bool = False) -> LinOp:
        r"""Fuse each term and combine the dense terms into one ``DenseLinOp`` (ADR-021).

        Fuse every term, sum the matrices of the densely-fusible ones into a
        single :class:`DenseLinOp`, and keep the remaining (matrix-free or
        structured) terms as lazy summands — so a matrix-free term is never
        densified. Adjoint-consistent and additive: ``(A + B)^* = A^* + B^*``.
        Combining reassociates the term order, so equality holds up to rounding.
        """
        from ._dense import DenseLinOp

        # Fuse each term, then add up the dense ones into one matrix;
        # matrix-free terms stay as lazy summands.
        fused = [p.fuse(materialize=materialize) for p in self.parts]
        dense = [p for p in fused if isinstance(p, DenseLinOp)]
        if len(dense) < 2:
            return make_sum(fused)
        matrix = dense[0].to_matrix()
        for d in dense[1:]:
            matrix = matrix + d.to_matrix()
        ref = dense[0]
        tensor = self.ops.reshape(
            matrix, tuple(ref.codomain.shape) + tuple(ref.domain.shape)
        )
        combined = DenseLinOp(tensor, ref.domain, ref.codomain, self.ctx)
        others = [p for p in fused if not isinstance(p, DenseLinOp)]
        return make_sum([combined, *others])

    def is_hermitian(self) -> bool | None:
        """
        Return whether this lazy sum is structurally Hermitian.

        The adjoint is additive, ``(A1 + ... + Ak)* = A1* + ... + Ak*``, so a
        sum is self-adjoint when every term is. If all operands are provably
        Hermitian this returns ``True``; otherwise the verdict is not cheaply
        decidable (a sum of non-Hermitian terms may still be Hermitian) and
        ``None`` is returned. ``False`` is never returned.

        Returns
        -------
        bool | None
            ``True`` when every part is provably Hermitian, otherwise ``None``.
        """
        if all(op.is_hermitian() is True for op in self.parts):
            return True
        return None

    def _repr_body(self) -> str:
        from .._repr import truncated_join

        return truncated_join((op._short_repr() for op in self.ops_tuple), " + ")


class ComposedLinOp(LinOp[Domain, Codomain]):
    r"""
    Lazy composition ``left @ right``; the adjoint is ``right.H @ left.H``.

    Parameters
    ----------
    left : LinOp
        Operator applied second.
    right : LinOp
        Operator applied first; ``right.codomain`` must equal ``left.domain``.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def _apply_core(self, x: Any) -> Any:
        for leaf in self._apply_chain:
            x = leaf._apply_core(x)
        return x

    def _rapply_core(self, z: Any) -> Any:
        for leaf in reversed(self._apply_chain):
            z = leaf._rapply_core(z)
        return z

    def _vapply_core(self, xs: Any) -> Any:
        for leaf in self._apply_chain:
            xs = leaf._vapply_core(xs)
        return xs

    def __init__(
        self,
        left: LinOp,
        right: LinOp,
        check_level: CheckLevel | None = None,
    ) -> None:
        # Validate: both LinOps, same context, matching middle space.
        left = _require_linop(left, "left")
        right = _require_linop(right, "right")
        _require_same_context((left, right))
        if not _same_space_for_algebra(right.codomain, left.domain):
            raise ValueError(
                "ComposedLinOp requires right.codomain == left.domain; "
                f"got {right.codomain!r} and {left.domain!r}."
            )
        # Bind with right's domain and left's codomain.
        if check_level is None:
            check_level = minimum_check_level((left.check_level, right.check_level))
        super().__init__(right.domain, left.codomain, left.ctx, check_level=check_level)
        self.left = left
        self.right = right
        # Fuse the (possibly nested) composition into one flat chain of leaf
        # operators in application order — right applied first, then left.
        # Cached at construction so every apply runs a single check-free loop
        # instead of re-walking the binary ComposedLinOp tree.
        self._apply_chain = _compose_chain(right) + _compose_chain(left)

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``left.apply(right.apply(x))``."""
        return self._apply_core(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, z: Any) -> Any:
        """Return ``right.rapply(left.rapply(z))``."""
        return self._rapply_core(z)

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``left.vapply(right.vapply(xs))``."""
        require(self.domain, _CAP_BATCH, "ComposedLinOp.vapply")
        require(self.codomain, _CAP_BATCH, "ComposedLinOp.vapply")
        return self._vapply_core(xs)

    def rvapply(self, zs: Any) -> Any:
        """Return ``right.rvapply(left.rvapply(zs))``."""
        require(self.domain, _CAP_BATCH, "ComposedLinOp.rvapply")
        require(self.codomain, _CAP_BATCH, "ComposedLinOp.rvapply")
        return self.right.rvapply(self.left.rvapply(zs))

    def fuse(self, *, materialize: bool = False) -> LinOp:
        r"""Fuse a composition of dense operators into one ``DenseLinOp`` (ADR-021).

        Fuse each operand first, then — when both fused operands are dense —
        replace ``A @ B`` with a single :class:`DenseLinOp` holding the matrix
        product :math:`M_A M_B`. Any operand that does not fuse to a dense
        operator (matrix-free leaves, sparse, structured) keeps the composition
        lazy, so a matrix-free operand is never densified. The fused matrix is
        adjoint-consistent on any geometry: the shared middle-space Riesz maps
        cancel, so the fused operator's metric adjoint equals
        ``B* @ A*`` up to floating-point rounding.
        """
        from ._dense import DenseLinOp

        # Two dense factors multiply into one matrix; otherwise stay lazy.
        left = self.left.fuse(materialize=materialize)
        right = self.right.fuse(materialize=materialize)
        if isinstance(left, DenseLinOp) and isinstance(right, DenseLinOp):
            ops = self.ops
            matrix = ops.matmul(left.to_matrix(), right.to_matrix())
            tensor = ops.reshape(
                matrix, tuple(left.codomain.shape) + tuple(right.domain.shape)
            )
            return DenseLinOp(tensor, right.domain, left.codomain, self.ctx)
        return make_composed(left, right)

    def is_hermitian(self) -> bool | None:
        """
        Return ``True`` for a Gram product ``R.H @ R`` (or ``L @ L.H``), else ``None``.

        Returns
        -------
        bool | None
            ``True`` when structurally Hermitian, ``None`` when unknown.
        """
        if self.left == self.right.H:
            return True
        return None

    def __eq__(self, other: Any) -> bool:
        """Return whether another composition has the same operands, in order."""
        if not self.same_math(other):              # Tier 1: backend
            return NotImplemented
        return self.left == other.left and self.right == other.right

    def _repr_body(self) -> str:
        return f"{self.left._short_repr()} ∘ {self.right._short_repr()}"

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


class ZeroLinOp(ZeroNode, LinOp[Domain, Codomain]):
    r"""
    Lazy zero map from ``dom`` to ``cod``.

    Parameters
    ----------
    dom : Space
        Domain.
    cod : Space
        Codomain.
    ctx : Context, str, or None, optional
        Backend context; resolved from the spaces when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def _apply_core(self, x: Any) -> Any:
        return self.codomain.zeros()

    def _rapply_core(self, y: Any) -> Any:
        return self.domain.zeros()

    def _vapply_core(self, xs: Any) -> Any:
        return _batched_zeros(self.codomain, _leading_shape(self.domain, xs))

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return the zero element of the codomain."""
        return self._apply_core(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Return the zero element of the domain."""
        return self._rapply_core(y)

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return the batched zero element of the codomain."""
        require(self.domain, _CAP_BATCH, "ZeroLinOp.vapply")
        require(self.codomain, _CAP_BATCH, "ZeroLinOp.vapply")
        return self._vapply_core(xs)

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Return the batched zero element of the domain."""
        require(self.domain, _CAP_BATCH, "ZeroLinOp.rvapply")
        require(self.codomain, _CAP_BATCH, "ZeroLinOp.rvapply")
        return _batched_zeros(self.domain, _leading_shape(self.codomain, ys))

    def to_dense(self) -> Any:
        """
        Return the dense tensor representation of the zero map.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        require(self.domain, _CAP_COORDINATE, "ZeroLinOp.to_dense")
        require(self.codomain, _CAP_COORDINATE, "ZeroLinOp.to_dense")
        return self.ops.zeros(
            tuple(self.codomain.shape) + tuple(self.domain.shape), dtype=self.dtype
        )

    def is_hermitian(self) -> bool:
        """
        Return whether the zero map is Hermitian.

        Returns
        -------
        bool
            ``True`` exactly when domain and codomain are the same space.
        """
        return self.domain == self.codomain


class IdentityLinOp(LinOp[Domain, Domain]):
    r"""
    Lazy identity map on ``space``.

    Parameters
    ----------
    space : Space
        Domain and codomain.
    ctx : Context, str, or None, optional
        Backend context; resolved from ``space`` when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def _apply_core(self, x: Any) -> Any:
        return x

    _rapply_core = _apply_core
    def _vapply_core(self, xs: Any) -> Any:
        return xs

    def __init__(
        self,
        space: Domain,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        super().__init__(space, space, ctx, check_level=check_level)

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Return ``x`` after domain validation."""
        return self._apply_core(x)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, x: Any) -> Any:
        """Return ``x`` after codomain validation."""
        return self._rapply_core(x)

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``xs`` after batched domain validation."""
        require(self.domain, _CAP_BATCH, "IdentityLinOp.vapply")
        require(self.codomain, _CAP_BATCH, "IdentityLinOp.vapply")
        return xs

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, xs: Any) -> Any:
        """Return ``xs`` after batched codomain validation."""
        require(self.domain, _CAP_BATCH, "IdentityLinOp.rvapply")
        require(self.codomain, _CAP_BATCH, "IdentityLinOp.rvapply")
        return xs

    def to_dense(self) -> Any:
        """
        Return the dense tensor representation of this identity map.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        require(self.domain, _CAP_COORDINATE, "IdentityLinOp.to_dense")
        require(self.codomain, _CAP_COORDINATE, "IdentityLinOp.to_dense")
        # The identity matrix, reshaped to cod.shape + dom.shape.
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
        if not self.same_math(other):              # Tier 1: backend
            return NotImplemented
        return self.domain == other.domain                      # Tier 2 (square: cod == dom)

    def _repr_body(self) -> str:
        from .._repr import describe_space

        return describe_space(self.domain)

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


class MatrixFreeLinOp(LinOp[Domain, Codomain]):
    """
    Linear operator given by forward and adjoint callables; no matrix is stored.

    ``rapply`` must be the adjoint in the declared inner products,
    ``<apply(x), y>_Y = <x, rapply(y)>_X``. It is used as given unless
    ``euclidean_adjoint=True``, and no adjoint test runs at construction.

    Parameters
    ----------
    apply : callable
        Forward map ``apply(x)`` from ``dom`` to ``cod``.
    rapply : callable
        Adjoint map ``rapply(y)`` from ``cod`` to ``dom``.
    dom : Space
        Domain.
    cod : Space
        Codomain.
    ctx : Context, str, or None, optional
        Backend context; inferred from ``dom`` and ``cod`` when omitted.
    vapply : callable or None, optional
        Batched forward map; ``vmap`` of ``apply`` by default.
    rvapply : callable or None, optional
        Batched adjoint; ``vmap`` of ``rapply`` by default.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    euclidean_adjoint : bool, optional
        ``True``: ``rapply`` is the coordinate adjoint and is wrapped into
        ``R_X^-1 A^T R_Y`` (see :meth:`from_coordinate_adjoint`). ``False``: it is the
        metric adjoint. Omitting it on a non-Euclidean space warns, because a
        coordinate adjoint there is silently wrong.
    """

    def _apply_core(self, x: Any) -> Any:
        return self.apply_fn(x)

    def _rapply_core(self, y: Any) -> Any:
        return self.rapply_fn(y)

    def _vapply_core(self, xs: Any) -> Any:
        if self.vapply_fn is None:
            return self.ops.vmap(self.apply, in_axes=0, out_axes=0)(xs)
        return self.vapply_fn(xs)

    def __init__(
        self,
        apply: Callable[[Any], Any],
        rapply: Callable[[Any], Any],
        dom: Domain,
        cod: Codomain,
        ctx: Context | str | None = None,
        vapply: Callable[[Any], Any] | None = None,
        rvapply: Callable[[Any], Any] | None = None,
        check_level: CheckLevel | None = None,
        *,
        euclidean_adjoint: bool | Any = _ADJOINT_UNSPECIFIED,
    ) -> None:
        # Validate the callables, then bind the spaces.
        if not callable(apply):
            raise TypeError(f"apply must be callable, got {type(apply).__name__}.")
        if not callable(rapply):
            raise TypeError(f"rapply must be callable, got {type(rapply).__name__}.")
        if vapply is not None and not callable(vapply):
            raise TypeError(f"vapply must be callable, got {type(vapply).__name__}.")
        if rvapply is not None and not callable(rvapply):
            raise TypeError(f"rvapply must be callable, got {type(rvapply).__name__}.")
        super().__init__(dom, cod, ctx, check_level=check_level)

        # Does the adjoint need Riesz maps? Only on a non-Euclidean inner-product space.
        # Without coordinates there is no coordinate adjoint to confuse with, so a
        # native space never warns.
        has_inner = all(
            isinstance(space, InnerProductSpace) for space in (self.domain, self.codomain)
        )
        non_euclidean = has_inner and any(
            isinstance(space, CoordinateSpace) and not space.is_euclidean
            for space in (self.domain, self.codomain)
        )
        if euclidean_adjoint is True:
            # The caller supplied the *coordinate* adjoint A^dagger; wrap it into
            # the metric adjoint R_X^-1 A^dagger R_Y once, here.
            try:
                _requires_euclidean_or_riesz(
                    self.domain,
                    self.codomain,
                    "MatrixFreeLinOp(euclidean_adjoint=True) / "
                    "MatrixFreeLinOp.from_coordinate_adjoint",
                )
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
            rapply, rvapply = self._wrap_coordinate_adjoint(rapply, rvapply)
        elif non_euclidean and euclidean_adjoint is _ADJOINT_UNSPECIFIED:
            # The callable is stored verbatim and *trusted* as the metric adjoint.
            # On a non-Euclidean geometry a coordinate adjoint is simply wrong here
            # and nothing downstream detects it, so say so once, at construction.
            # Silent on Euclidean spaces (the two coincide) and silent when the
            # caller passed ``euclidean_adjoint=False`` explicitly, which asserts
            # that the supplied adjoint already carries the geometry.
            warnings.warn(
                "MatrixFreeLinOp stores `rapply` verbatim as the metric adjoint, but "
                f"{'domain' if not self.domain.is_euclidean else 'codomain'} geometry is "
                "non-Euclidean, where the metric adjoint R_X^-1 A^dagger R_Y differs from "
                "the coordinate adjoint. Pass euclidean_adjoint=True to have the coordinate "
                "adjoint wrapped for you, or confirm `rapply` already carries the geometry.",
                UserWarning,
                stacklevel=2,
            )

        # Store the callables; strict mode also dot-tests the adjoint.
        self.apply_fn = apply
        self.rapply_fn = rapply
        self.vapply_fn = vapply
        self.rvapply_fn = rvapply
        if self._checks_at_least("strict"):
            self._check_adjoint_consistency()

    def _wrap_coordinate_adjoint(
        self,
        coordinate_rapply: Callable[[Any], Any],
        coordinate_rvapply: Callable[[Any], Any] | None,
    ) -> tuple[Callable[[Any], Any], Callable[[Any], Any] | None]:
        """Return ``(rapply, rvapply)`` wrapping coordinate adjoints in Riesz maps."""
        # Wrap coordinate adjoints into metric ones: R_X^-1 A^T R_Y.
        dom, cod, ops = self.domain, self.codomain, self.ctx.ops

        def wrapped_rapply(y: Any) -> Any:
            return metric_rapply(dom, cod, coordinate_rapply, y)

        wrapped_rvapply: Callable[[Any], Any] | None = None
        if coordinate_rvapply is not None:

            def _wrapped_rvapply(ys: Any) -> Any:
                return metric_rvapply(
                    dom, cod, coordinate_rapply, coordinate_rvapply, ys,
                    opname="MatrixFreeLinOp", ops=ops,
                )

            wrapped_rvapply = _wrapped_rvapply
        return wrapped_rapply, wrapped_rvapply

    def _check_adjoint_consistency(self) -> None:
        """Probe the declared adjoint identity on deterministic space elements."""
        # Dot test <A x, y> = <x, A* y> on deterministic probe vectors.
        if not all(
            isinstance(space, InnerProductSpace)
            for space in (self.domain, self.codomain)
        ):
            return

        x = probe_element(self.domain, self.ops, self.dtype)
        y = probe_element(self.codomain, self.ops, self.dtype)
        if x is None or y is None:
            # Neither coordinates nor ``ones``: the space cannot build a non-zero
            # element, so there is nothing to probe with. Construction-time
            # geometry warnings still apply.
            return

        try:
            ax = self.apply_fn(x)
            ahy = self.rapply_fn(y)
            self.codomain._check_member(ax)
            self.domain._check_member(ahy)
            lhs = cast(Any, self.codomain).inner(ax, y)
            rhs = cast(Any, self.domain).inner(x, ahy)
            consistent = bool(self.ops.allclose(lhs, rhs))
        except Exception as exc:
            raise ValueError(
                "Strict matrix-free adjoint consistency check could not be completed."
            ) from exc
        if not consistent:
            raise ValueError(
                "Strict matrix-free adjoint consistency check failed: "
                "<A x, y> != <x, A* y> for the deterministic probe."
            )

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

        The stored adjoint is the metric one, ``R_X^-1 A^T R_Y``, which is the given
        callable itself on Euclidean spaces. A non-Euclidean space without usable Riesz
        maps is rejected.

        Parameters
        ----------
        apply : callable
            Forward map from ``dom`` to ``cod``.
        coordinate_rapply : callable
            Euclidean coordinate adjoint of ``apply``.
        dom, cod : Space
            Domain and codomain.
        ctx : Context, str, or None, optional
            Backend context.
        vapply : callable or None, optional
            Batched forward map.
        coordinate_rvapply : callable or None, optional
            Batched coordinate adjoint, wrapped the same way; ``vmap`` of the wrapped
            adjoint by default.
        """
        # Validate the callables, then build via __init__ with euclidean_adjoint=True.
        if not callable(apply):
            raise TypeError(f"apply must be callable, got {type(apply).__name__}.")
        if not callable(coordinate_rapply):
            raise TypeError(
                f"coordinate_rapply must be callable, got {type(coordinate_rapply).__name__}."
            )
        if coordinate_rvapply is not None:
            if not callable(coordinate_rvapply):
                raise TypeError(
                    f"coordinate_rvapply must be callable, got {type(coordinate_rvapply).__name__}."
                )

        # The wrapping now lives in __init__ behind euclidean_adjoint=True, so the
        # two entry points cannot drift apart, and this path raises no warning:
        # passing the coordinate adjoint is exactly what it is documented to take.
        return cls(
            apply,
            coordinate_rapply,
            dom,
            cod,
            ctx,
            vapply,
            coordinate_rvapply,
            euclidean_adjoint=True,
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
        return self._apply_core(x)

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
        require(self.domain, InnerProductSpace, "MatrixFreeLinOp.rapply")
        require(self.codomain, InnerProductSpace, "MatrixFreeLinOp.rapply")
        return self._rapply_core(y)

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
        require(self.domain, _CAP_BATCH, "MatrixFreeLinOp.vapply")
        require(self.codomain, _CAP_BATCH, "MatrixFreeLinOp.vapply")
        # Use the supplied batched map, else the generic vmap fallback.
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
        require(self.domain, _CAP_BATCH, "MatrixFreeLinOp.rvapply")
        require(self.codomain, _CAP_BATCH, "MatrixFreeLinOp.rvapply")
        # Use the supplied batched adjoint, else the generic vmap fallback.
        if self.rvapply_fn is None:
            return super().rvapply(ys)
        return self.rvapply_fn(ys)

    def fuse(self, *, materialize: bool = False) -> LinOp:
        """Stay matrix-free unless ``materialize=True`` is requested (ADR-021/ADR-008).

        By default a matrix-free operator is never densified, so it returns
        itself. With ``materialize=True`` the caller explicitly opts into
        densification: the operator is probed into a dense tensor (via
        ``to_dense``, the basis sweep) and returned as a :class:`DenseLinOp`,
        letting an enclosing expression collapse to a single dense operator.
        """
        # Matrix-free stays lazy unless densifying was requested.
        if not materialize:
            return self
        from ._dense import DenseLinOp

        return DenseLinOp(self.to_dense(), self.domain, self.codomain, self.ctx)

    def __eq__(self, other: Any) -> bool:
        if not self.same_math(other):              # Tier 1: backend
            return NotImplemented
        # Tier 2: spaces + callable identity. Extensional equality of callables
        # is undecidable, so 'is' is the only sound comparison.
        if self.domain != other.domain or self.codomain != other.codomain:
            return False
        return (
            self.apply_fn is other.apply_fn
            and self.vapply_fn is other.vapply_fn
            and self.rapply_fn is other.rapply_fn
            and self.rvapply_fn is other.rvapply_fn
        )

    def tree_flatten(self):
        # No array children: the callables and spaces are static data.
        children = ()
        aux = (
            self.apply_fn,
            self.rapply_fn,
            self.domain,
            self.codomain,
            self.ctx,
            self.vapply_fn,
            self.rvapply_fn,
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
        ) = aux
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
        # ``self.rapply_fn`` is already the metric adjoint by construction -- either
        # supplied that way, or wrapped once at __init__ under
        # ``euclidean_adjoint=True``. Declaring ``False`` here is therefore both
        # correct (re-wrapping would apply the Riesz maps twice) and quiet: the
        # construction advisory belongs to the original call, not to every convert.
        # The stored rapply is already the metric adjoint, so no re-wrap.
        return MatrixFreeLinOp(
            self.apply_fn,
            self.rapply_fn,
            self.domain.convert(new_ctx),
            self.codomain.convert(new_ctx),
            new_ctx,
            self.vapply_fn,
            self.rvapply_fn,
            euclidean_adjoint=False,
        )


class _AdjointViewLinOp(LinOp[Codomain, Domain]):
    """
    Adjoint view ``op.H``: ``apply`` is ``op.rapply`` and ``rapply`` is ``op.apply``; ``op.H.H is op``.

    Parameters
    ----------
    op : LinOp
        Operator whose adjoint is viewed.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def _apply_core(self, y: Any) -> Any:
        return self.op._rapply_core(y)

    def _rapply_core(self, x: Any) -> Any:
        return self.op._apply_core(x)

    def _vapply_core(self, ys: Any) -> Any:
        return self.op.rvapply(ys)

    def __init__(
        self,
        op: LinOp[Domain, Codomain],
        check_level: CheckLevel | None = None,
    ) -> None:
        # The view swaps domain and codomain.
        op = _require_linop(op, "op")
        if check_level is None:
            check_level = op.check_level
        super().__init__(op.codomain, op.domain, op.ctx, check_level=check_level)
        self.op = op

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, y: Any) -> Any:
        """Return ``op.rapply(y)``."""
        return self._apply_core(y)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, x: Any) -> Any:
        """Return ``op.apply(x)``."""
        return self._rapply_core(x)

    def vapply(self, ys: Any) -> Any:
        """Return ``op.rvapply(ys)`` over a batch."""
        require(self.domain, _CAP_BATCH, "_AdjointViewLinOp.vapply")
        require(self.codomain, _CAP_BATCH, "_AdjointViewLinOp.vapply")
        return self.op.rvapply(ys)

    def rvapply(self, xs: Any) -> Any:
        """Return ``op.vapply(xs)`` over a batch."""
        require(self.domain, _CAP_BATCH, "_AdjointViewLinOp.rvapply")
        require(self.codomain, _CAP_BATCH, "_AdjointViewLinOp.rvapply")
        return self.op.vapply(xs)

    def fuse(self, *, materialize: bool = False) -> LinOp:
        """Fuse the wrapped operand; return the adjoint of the fused operator (ADR-021).

        ``A.H.fuse()`` is ``A.fuse().H`` — the inner expression is fused (e.g. a
        composition collapses to one dense operator) and the adjoint view wraps
        the result, which keeps the metric adjoint correct on any geometry. A
        matrix-free operand stays matrix-free under its adjoint (or is densified
        when ``materialize=True``).
        """
        return self.op.fuse(materialize=materialize).H

    @property
    def H(self) -> LinOp[Domain, Codomain]:
        """Original operator viewed as the adjoint of this adjoint view."""
        return self.op

    def __eq__(self, other: Any) -> bool:
        if not self.same_math(other):              # Tier 1: backend
            return NotImplemented
        return self.op == other.op

    def _repr_class_name(self) -> str:
        """Present a clean public label for the private adjoint-view class."""
        return "AdjointLinOp"

    def _repr_body(self) -> str:
        return f"{self.op._short_repr()}.H"

    def tree_flatten(self):
        children = (self.op,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])

    def _convert(self, new_ctx: Context) -> _AdjointViewLinOp:
        return _AdjointViewLinOp(self.op.convert(new_ctx))


LinOp._Sum, LinOp._Scaled, LinOp._Zero = SumLinOp, ScaledLinOp, ZeroLinOp

register("add", LinOp, LinOp, lambda a, b: make_sum((a, b)))
register("mul", Scalar, LinOp, make_scaled)
register("mul", LinOp, Scalar, lambda op, s: make_scaled(s, op))
register("matmul", LinOp, LinOp, make_composed)
register("matmul", LinOp, Element, lambda A, x: DenseElement._make(A.codomain, A.apply(x.densify())))
