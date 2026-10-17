from __future__ import annotations

from functools import reduce
from typing import Any, Sequence

from ._base import LinOp, Domain, Codomain, _require_linop
from ..mapping import (
    Scalar, ScaledNode, SumNode, ZeroNode, make_scaled_of, make_sum_of, register,
    same_space_for_algebra as _same_space_for_algebra,
)
from ..space import DenseElement, Element
from .._checks import CheckLevel, minimum_check_level
from .._checks import checked_method
from ..space._capabilities import _CAP_BATCH, _CAP_COORDINATE
from .._repr import summarize_value
from ..mapping._lazy_algebra import conjugate_scalar as _conjugate_scalar
from ._matrix_free import IdentityLinOp


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


def _check_composable(left: LinOp, right: LinOp) -> tuple[LinOp, LinOp]:
    """Validate ``left @ right``: both LinOps, same context, matching middle space."""
    left = _require_linop(left, "left")
    right = _require_linop(right, "right")
    if not left.ctx.same_math(right.ctx):
        raise ValueError(
            "All LinOp operands in an algebraic expression must have the same ctx; "
            f"operand 0 has ctx {left.ctx!r}, operand 1 has ctx {right.ctx!r}."
        )
    if not _same_space_for_algebra(right.codomain, left.domain):
        raise ValueError(
            "ComposedLinOp requires right.codomain == left.domain; "
            f"got {right.codomain!r} and {left.domain!r}."
        )
    return left, right


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
    left, right = _check_composable(left, right)

    # Local rewrites: identity factors vanish, zero factors give a zero map.
    if isinstance(right, IdentityLinOp):
        return left
    if isinstance(left, IdentityLinOp):
        return right
    if isinstance(left, ZeroLinOp) or isinstance(right, ZeroLinOp):
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

    _fields = ("scalar", "op:operand")
    _dynamic = ("scalar", "op")

    def _apply_core(self, x: Any) -> Any:
        y = self.op._apply_core(x)
        scale = self.codomain._scale_core
        return scale(self.scalar, y)

    def _rapply_core(self, y: Any) -> Any:
        x = self.op._rapply_core(y)
        scale = self.domain._scale_core
        return scale(_conjugate_scalar(self.scalar), x)

    def _vapply_core(self, xs: Any) -> Any:
        ys = self.op._vapply_core(xs)
        return self.codomain._scale_batch_core(self.scalar, ys)

    def __init__(self, scalar: Any, op: LinOp[Domain, Codomain], check_level: CheckLevel | None = None) -> None:
        super().__init__(scalar, op, check_level)

    @property
    def op(self) -> LinOp[Domain, Codomain]:
        """The scaled operand."""
        return self.operand

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``scalar * op.vapply(xs)``."""
        self._require(_CAP_BATCH, "ScaledLinOp.vapply")
        return self._vapply_core(xs)

    def rvapply(self, ys: Any) -> Any:
        """Return ``conj(scalar) * op.rvapply(ys)``."""
        self._require(_CAP_BATCH, "ScaledLinOp.rvapply")
        # Adjoint of s A is conj(s) A*.
        xs = self.op.rvapply(ys)
        return self.domain._scale_batch_core(_conjugate_scalar(self.scalar), xs)

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

    _fields = ("ops:parts",)
    _dynamic = ("ops",)

    def _apply_core(self, x: Any) -> Any:
        # A1 x + ... + Ak x, with the codomain's unchecked add when it has one.
        return reduce(self.codomain._add_core, (part._apply_core(x) for part in self.ops_tuple))

    def _rapply_core(self, y: Any) -> Any:
        # A1* y + ... + Ak* y, added in the domain.
        return reduce(self.domain._add_core, (part._rapply_core(y) for part in self.ops_tuple))

    def _vapply_core(self, xs: Any) -> Any:
        # Row-wise sum of the batched applies.
        return reduce(self.codomain._add_batch_core, (part._vapply_core(xs) for part in self.ops_tuple))

    def __init__(self, ops: Sequence[LinOp[Domain, Codomain]], check_level: CheckLevel | None = None) -> None:
        super().__init__(ops, check_level)

    @property
    def ops_tuple(self) -> tuple[LinOp[Domain, Codomain], ...]:
        """The summed operators (alias of ``parts``)."""
        return self.parts

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``sum_i ops[i].vapply(xs)``."""
        self._require(_CAP_BATCH, "SumLinOp.vapply")
        return self._vapply_core(xs)

    @checked_method(in_space="codomain", out_space="domain", in_batched=True, out_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Return ``sum_i ops[i].rvapply(ys)``."""
        self._require(_CAP_BATCH, "SumLinOp.rvapply")
        # Adjoint of a sum is the sum of adjoints.
        return reduce(self.domain._add_batch_core, (op.rvapply(ys) for op in self.ops_tuple))

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

    _fields = ("left", "right")
    _dynamic = ("left", "right")

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
        left, right = _check_composable(left, right)
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

    @checked_method(in_space="domain", out_space="codomain", in_batched=True, out_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return ``left.vapply(right.vapply(xs))``."""
        self._require(_CAP_BATCH, "ComposedLinOp.vapply")
        return self._vapply_core(xs)

    def rvapply(self, zs: Any) -> Any:
        """Return ``right.rvapply(left.rvapply(zs))``."""
        self._require(_CAP_BATCH, "ComposedLinOp.rvapply")
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
        Return ``True`` for an adjoint-palindromic chain (``R.H @ R``, ``L @ V @ L.H`` with ``V`` self-adjoint), else ``None``.

        Returns
        -------
        bool | None
            ``True`` when structurally Hermitian, ``None`` when unknown.
        """
        # The chain c_k ... c_0 is self-adjoint when c_i# == c_(k-i) for every i and the
        # middle factor, if any, is self-adjoint: R# R, L V L# and their nestings.
        chain = self._apply_chain
        k = len(chain)
        if all(chain[i].H == chain[k - 1 - i] for i in range(k // 2)) and (
            k % 2 == 0 or chain[k // 2].is_hermitian() is True
        ):
            return True
        return None

    def _repr_body(self) -> str:
        return f"{self.left._short_repr()} ∘ {self.right._short_repr()}"

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

    @checked_method(in_space="domain", in_batched=True)
    def vapply(self, xs: Any) -> Any:
        """Return the batched zero element of the codomain."""
        self._require(_CAP_BATCH, "ZeroLinOp.vapply")
        return self._vapply_core(xs)

    @checked_method(in_space="codomain", in_batched=True)
    def rvapply(self, ys: Any) -> Any:
        """Return the batched zero element of the domain."""
        self._require(_CAP_BATCH, "ZeroLinOp.rvapply")
        return _batched_zeros(self.domain, _leading_shape(self.codomain, ys))

    def to_dense(self) -> Any:
        """
        Return the dense tensor representation of the zero map.

        The returned array has shape ``self.codomain.shape + self.domain.shape``.
        """
        self._require(_CAP_COORDINATE, "ZeroLinOp.to_dense")
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




LinOp._Sum, LinOp._Scaled, LinOp._Zero = SumLinOp, ScaledLinOp, ZeroLinOp

register("add", LinOp, LinOp, lambda a, b: make_sum((a, b)))
register("mul", Scalar, LinOp, make_scaled)
register("mul", LinOp, Scalar, lambda op, s: make_scaled(s, op))
register("matmul", LinOp, LinOp, make_composed)
register("matmul", LinOp, Element, lambda A, x: DenseElement._make(A.codomain, A.apply(x.densify())))
