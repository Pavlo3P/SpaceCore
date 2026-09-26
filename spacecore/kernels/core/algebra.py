"""Concrete core kernels for the lazy LinOp algebra.

Each function here is the check-free *core* of one algebra operator's apply
paths: it receives the operator instance (already boundary-validated by the
public ``apply``/``rapply``/``vapply``) and the operand, and returns the result
without re-validating intermediates.

The functions are duck-typed on the operator — they read ``op.left``,
``op.scalar``, ``op.ops_tuple``, ``op._apply_chain``, etc. — so this module does
**not** import the operator classes from :mod:`spacecore.linop`. That keeps the
kernels subpackage free of a circular dependency: ``linop`` imports ``kernels``,
never the reverse.

The operator classes bind these functions as class attributes
(``_apply_core = scaled_apply_core``).
"""
from __future__ import annotations

from typing import Any


# Re-exported: the scalar predicates/helpers live together in
# :mod:`spacecore._lazy_algebra` (stdlib-only, so importable from anywhere),
# next to ``is_scalar_like`` / ``scalar_eq`` which share their duck-typing
# assumptions. Kept importable from here for the kernel call sites.
from ..._lazy_algebra import conjugate_scalar as conjugate_scalar  # noqa: F401


def compose_chain(op: Any) -> tuple:
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


def leading_shape(space: Any, value: Any) -> tuple[int, ...]:
    """Infer the leading batch dimensions of ``value`` relative to ``space``."""
    # Product spaces: batch shape of the first component. Else strip the trailing space shape.
    parts = getattr(space, "spaces", None)
    if parts is not None and isinstance(value, tuple) and value:
        return leading_shape(parts[0], value[0])
    shape = tuple(getattr(value, "shape", ()))
    base = tuple(space.shape)
    return shape if not base else shape[: len(shape) - len(base)]


def batched_zeros(space: Any, leading: tuple[int, ...]) -> Any:
    """Return a batched zero element of ``space`` with leading shape ``leading``."""
    # Product spaces get a tuple of per-component zero batches.
    parts = getattr(space, "spaces", None)
    if parts is not None:
        return tuple(batched_zeros(part, leading) for part in parts)
    return space.ops.zeros(leading + tuple(space.shape), dtype=space.dtype)


# ---------------------------------------------------------------------------
# Composition — fused flat chain of leaf cores
# ---------------------------------------------------------------------------


def _composed_chain_apply(chain: Any, x: Any) -> Any:
    """Run each leaf core in application order (generic composed apply)."""
    for leaf in chain:
        x = leaf._apply_core(x)
    return x


def composed_apply_core(op: Any, x: Any) -> Any:
    return _composed_chain_apply(op._apply_chain, x)


def composed_rapply_core(op: Any, z: Any) -> Any:
    for leaf in reversed(op._apply_chain):
        z = leaf._rapply_core(z)
    return z


def composed_vapply_core(op: Any, xs: Any) -> Any:
    for leaf in op._apply_chain:
        xs = leaf._vapply_core(xs)
    return xs


# ---------------------------------------------------------------------------
# Scaling — fold the scalar through the operand's core
# ---------------------------------------------------------------------------
def scaled_apply_core(op: Any, x: Any) -> Any:
    y = op.op._apply_core(x)
    scale = getattr(op.codomain, "_scale_core", op.codomain.scale)
    return scale(op.scalar, y)


def scaled_rapply_core(op: Any, y: Any) -> Any:
    x = op.op._rapply_core(y)
    scale = getattr(op.domain, "_scale_core", op.domain.scale)
    return scale(conjugate_scalar(op.scalar), x)


def scaled_vapply_core(op: Any, xs: Any) -> Any:
    ys = op.op._vapply_core(xs)
    return op.codomain.scale_batch(op.scalar, ys)


# ---------------------------------------------------------------------------
# Sum — accumulate the operands' cores via the space's check-free add
# ---------------------------------------------------------------------------
def sum_apply_core(op: Any, x: Any) -> Any:
    # A1 x + ... + Ak x, with the codomain's unchecked add when it has one.
    parts = op.ops_tuple
    add = getattr(op.codomain, "_add_core", op.codomain.add)
    acc = parts[0]._apply_core(x)
    for part in parts[1:]:
        acc = add(acc, part._apply_core(x))
    return acc


def sum_rapply_core(op: Any, y: Any) -> Any:
    # A1* y + ... + Ak* y, added in the domain.
    parts = op.ops_tuple
    add = getattr(op.domain, "_add_core", op.domain.add)
    acc = parts[0]._rapply_core(y)
    for part in parts[1:]:
        acc = add(acc, part._rapply_core(y))
    return acc


def sum_vapply_core(op: Any, xs: Any) -> Any:
    # Row-wise sum of the batched applies.
    parts = op.ops_tuple
    add_batch = op.codomain.add_batch
    acc = parts[0]._vapply_core(xs)
    for part in parts[1:]:
        acc = add_batch(acc, part._vapply_core(xs))
    return acc


# ---------------------------------------------------------------------------
# Adjoint view — swap forward/adjoint cores of the wrapped operator
# ---------------------------------------------------------------------------
def adjoint_apply_core(op: Any, y: Any) -> Any:
    return op.op._rapply_core(y)


def adjoint_rapply_core(op: Any, x: Any) -> Any:
    return op.op._apply_core(x)


def adjoint_vapply_core(op: Any, ys: Any) -> Any:
    return op.op.rvapply(ys)


# ---------------------------------------------------------------------------
# Identity / Zero — trivial cores
# ---------------------------------------------------------------------------
def identity_apply_core(op: Any, x: Any) -> Any:
    return x


def identity_vapply_core(op: Any, xs: Any) -> Any:
    return xs


def zero_apply_core(op: Any, x: Any) -> Any:
    return op.codomain.zeros()


def zero_rapply_core(op: Any, y: Any) -> Any:
    return op.domain.zeros()


def zero_vapply_core(op: Any, xs: Any) -> Any:
    return batched_zeros(op.codomain, leading_shape(op.domain, xs))


# ---------------------------------------------------------------------------
# Matrix-free — call the user-supplied callables directly
# ---------------------------------------------------------------------------
def matrixfree_apply_core(op: Any, x: Any) -> Any:
    return op.apply_fn(x)


def matrixfree_rapply_core(op: Any, y: Any) -> Any:
    return op.rapply_fn(y)


def matrixfree_vapply_core(op: Any, xs: Any) -> Any:
    if op.vapply_fn is None:
        return op.ops.vmap(op.apply, in_axes=0, out_axes=0)(xs)
    return op.vapply_fn(xs)
