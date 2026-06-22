"""Composed-chain apply kernel.

The generic path for ``A @ B @ C @ x`` is
``ComposedLinOp(A, ComposedLinOp(B, C)).apply(x)``. Each nested
``apply`` runs the :func:`spacecore._checks.checked_method` wrapper on
the intermediate value, which is correct but pays the validation cost on
each link of the chain.

The :func:`composed_chain_apply_optimized` kernel takes a flat sequence
of operators ``(A, B, C, ...)`` (in application order: rightmost applied
first) and an input ``x``, and applies each operator in turn directly,
skipping the per-step ``checked_method`` decorator overhead on the
intermediate results.

This kernel does **not** change any default code path. It is an
alternative entry point for callers that already know they want chain
semantics on validated inputs and outputs. The first input and the last
output remain user-visible and would still be validated by the surrounding
LinOp boundary if a user wraps the result.
"""
from __future__ import annotations

from typing import Any, Sequence

from ._policy import KernelSpec
from ._registry import registry


def composed_chain_apply_generic(linops: Sequence[Any], x: Any) -> Any:
    """Reference implementation: full ``ComposedLinOp`` chain.

    Builds a right-folded ``ComposedLinOp`` tree and calls ``.apply(x)``.
    This is the implementation an optimized kernel must match
    numerically.
    """
    if not linops:
        return x
    from spacecore.linop._algebra import ComposedLinOp

    composed = linops[-1]
    for op in reversed(linops[:-1]):
        composed = ComposedLinOp(op, composed)
    return composed.apply(x)


def composed_chain_apply_optimized(linops: Sequence[Any], x: Any) -> Any:
    """Optimized implementation: in-order ``.apply`` calls, no rewrapping.

    Equivalent to ``linops[0].apply(linops[1].apply(... linops[-1].apply(x)))``.
    Skips the per-link ``ComposedLinOp`` allocation and the intermediate
    ``ComposedLinOp.apply`` ``@checked_method`` wrapper; each underlying
    operator's own check policy still applies to its own apply call.
    """
    out = x
    for op in reversed(linops):
        out = op.apply(out)
    return out


def composed_chain_apply_applicable(linops: Sequence[Any], x: Any) -> bool:
    """Always applicable: any nonempty sequence of compatible LinOps.

    Compatibility (codomain matches the next operator's domain) is a
    precondition of constructing the generic ``ComposedLinOp`` chain as
    well, so this kernel inherits the same constraint without checking
    it here.
    """
    return True


SPEC = registry.register(
    KernelSpec(
        name="composed-chain-apply",
        generic=composed_chain_apply_generic,
        optimized=composed_chain_apply_optimized,
        applicable=composed_chain_apply_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::test_composed_chain_apply_matches_generic"
        ),
        benchmark_id="kernels.composed_chain_apply",
        rtol=0.0,
        atol=0.0,
        notes=(
            "Identical sequence of ``.apply`` calls without wrapping in "
            "ComposedLinOp. Used by inner loops that already know the "
            "chain is well-typed."
        ),
    )
)
