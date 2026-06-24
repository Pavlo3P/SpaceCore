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

The operator classes opt in via the :func:`spacecore.kernels.core_kernels` class
decorator (see :mod:`spacecore.kernels._core`); the kernel sets registered at the
bottom of this module are what those declarations resolve.
"""
from __future__ import annotations

from typing import Any

from ._rules import CoreKernelSet, register_core_kernels
from ..specs._dispatch import dispatch, should_consult_dispatch

# ---------------------------------------------------------------------------
# Shared rules / helpers
# ---------------------------------------------------------------------------
def conjugate_scalar(value: Any) -> Any:
    """Return the scalar conjugate when the value supports conjugation."""
    if hasattr(value, "conjugate"):
        return value.conjugate()
    if hasattr(value, "conj"):
        return value.conj()
    return value


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
    parts = getattr(space, "spaces", None)
    if parts is not None and isinstance(value, tuple) and value:
        return leading_shape(parts[0], value[0])
    shape = tuple(getattr(value, "shape", ()))
    base = tuple(space.shape)
    return shape if not base else shape[: len(shape) - len(base)]


def batched_zeros(space: Any, leading: tuple[int, ...]) -> Any:
    """Return a batched zero element of ``space`` with leading shape ``leading``."""
    parts = getattr(space, "spaces", None)
    if parts is not None:
        return tuple(batched_zeros(part, leading) for part in parts)
    return space.ops.zeros(leading + tuple(space.shape), dtype=space.dtype)


# ---------------------------------------------------------------------------
# Composition — fused flat chain of leaf cores
# ---------------------------------------------------------------------------
#
# ADR-016 approves the composed apply chain as a dispatch call site. The flat
# fused loop below is the ``generic`` fallback; a dispatch-eligible spec
# registered under ``"linop.composed.apply"`` (none ships on by default) would
# be selected for it when dispatch is on. The ``should_consult_dispatch`` guard
# keeps the default (dispatch ``off``, non-strict) path a single bound-method
# call away from today's loop — the core layer's zero-cost guarantee holds.
_COMPOSED_APPLY_KEY = "linop.composed.apply"


def _composed_chain_apply(chain: Any, x: Any) -> Any:
    """Generic composed apply: run each leaf core in application order."""
    for leaf in chain:
        x = leaf._apply_core(x)
    return x


def composed_apply_core(op: Any, x: Any) -> Any:
    chain = op._apply_chain
    if should_consult_dispatch(op.ctx):
        return dispatch(
            _COMPOSED_APPLY_KEY, chain, x, generic=_composed_chain_apply, ctx=op.ctx
        )
    return _composed_chain_apply(chain, x)


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
    parts = op.ops_tuple
    add = getattr(op.codomain, "_add_core", op.codomain.add)
    acc = parts[0]._apply_core(x)
    for part in parts[1:]:
        acc = add(acc, part._apply_core(x))
    return acc


def sum_rapply_core(op: Any, y: Any) -> Any:
    parts = op.ops_tuple
    add = getattr(op.domain, "_add_core", op.domain.add)
    acc = parts[0]._rapply_core(y)
    for part in parts[1:]:
        acc = add(acc, part._rapply_core(y))
    return acc


def sum_vapply_core(op: Any, xs: Any) -> Any:
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


# ---------------------------------------------------------------------------
# Registration — the rules that operator classes resolve by name
# ---------------------------------------------------------------------------
register_core_kernels(CoreKernelSet(
    "composed", composed_apply_core, composed_rapply_core, composed_vapply_core,
    notes="Fused flat chain of leaf cores; no per-link rewrapping or re-check.",
))
register_core_kernels(CoreKernelSet(
    "scaled", scaled_apply_core, scaled_rapply_core, scaled_vapply_core,
    notes="Fold the scalar through the operand core via the space scale helper.",
))
register_core_kernels(CoreKernelSet(
    "sum", sum_apply_core, sum_rapply_core, sum_vapply_core,
    notes="Accumulate operand cores via the space's check-free add.",
))
register_core_kernels(CoreKernelSet(
    "adjoint", adjoint_apply_core, adjoint_rapply_core, adjoint_vapply_core,
    notes="Swap the wrapped operator's forward and adjoint cores.",
))
register_core_kernels(CoreKernelSet(
    "identity", identity_apply_core, identity_apply_core, identity_vapply_core,
    notes="Pass the operand through unchanged.",
))
register_core_kernels(CoreKernelSet(
    "zero", zero_apply_core, zero_rapply_core, zero_vapply_core,
    notes="Return the codomain/domain zero element.",
))
register_core_kernels(CoreKernelSet(
    "matrixfree", matrixfree_apply_core, matrixfree_rapply_core, matrixfree_vapply_core,
    notes="Invoke the user-supplied apply/rapply/vapply callables directly.",
))
