"""Structural simplification kernels for the composed apply chain.

These are dispatch-eligible :class:`KernelSpec` objects routed at the
``"linop.composed.apply"`` call site (``spacecore.kernels.core.algebra``). They
optimize the *algebra* of a composition by exploiting two exact identities:

* **Zero annihilation.** A composition that contains a zero map *is* the zero
  map: ``A @ 0 @ B = 0``. When any leaf in the flattened chain is a
  ``ZeroLinOp``, the whole chain's result is the codomain's zero element, so the
  apply collapses to one ``codomain.zeros()`` instead of walking every leaf.
* **Identity elision.** ``A @ I @ B = A @ B``: an identity leaf is a no-op, so
  the apply skips it instead of paying a (cheap but nonzero) bound-method call.

Both are **exact** for the LinOp contract (a linear operator maps a true zero to
a true zero, and the identity returns its input unchanged), so both ship with
``rtol == atol == 0`` and are dispatch-eligible. Neither allocates more than the
``generic`` path, so neither needs a ``cost``.

Why a *runtime* dispatch when ``make_composed`` already canonicalizes ``A @ I``
and ``A @ 0`` at construction: that canonicalization is local and binary, and it
is bypassed whenever a ``ComposedLinOp`` is built directly — JAX pytree
``tree_unflatten`` (``cls(left, right)``), ``_convert`` rebuilding across
backends, or an explicit ``ComposedLinOp(A, IdentityLinOp(X))``. The dispatcher
catches those flattened chains; with dispatch off (the default) nothing changes.

Leaf kind is read from the duck-typed ``_core_kernel_set`` attribute that the
``@core_kernels(...)`` decorator stamps on each operator class, so this module
imports nothing from :mod:`spacecore.linop` and forms no import cycle.
"""
from __future__ import annotations

from typing import Any, Sequence

from ._policy import KernelSpec
from ._registry import registry

_COMPOSED_APPLY_KEY = "linop.composed.apply"


def _leaf_kind(leaf: Any) -> str | None:
    """Return the leaf's core-kernel-set name (``"identity"``/``"zero"``/...)."""
    return getattr(leaf, "_core_kernel_set", None)


def composed_chain_apply_generic(chain: Sequence[Any], x: Any) -> Any:
    """Reference: apply each leaf core in application order.

    Byte-identical to the ``"linop.composed.apply"`` call site's inline path
    (``spacecore.kernels.core.algebra._composed_chain_apply``); re-exposed here
    so the correctness test can call it directly.
    """
    for leaf in chain:
        x = leaf._apply_core(x)
    return x


# ---------------------------------------------------------------------------
# Zero annihilation
# ---------------------------------------------------------------------------
def composed_zero_applicable(chain: Sequence[Any], x: Any) -> bool:
    """Applicable when the chain contains a zero map.

    A nonempty chain is required (an empty chain is the identity, not zero).
    """
    return any(_leaf_kind(leaf) == "zero" for leaf in chain)


def composed_zero_optimized(chain: Sequence[Any], x: Any) -> Any:
    """Short-circuit to the codomain zero element.

    The chain's overall codomain is the last-applied leaf's codomain
    (``chain[-1]`` is applied last). For any finite linear chain this equals the
    generic result exactly: the zero leaf produces a true zero and every
    subsequent linear leaf maps it to a true zero.
    """
    return chain[-1].codomain.zeros()


# ---------------------------------------------------------------------------
# Identity elision
# ---------------------------------------------------------------------------
def composed_identity_applicable(chain: Sequence[Any], x: Any) -> bool:
    """Applicable when at least one leaf is an identity to skip."""
    return any(_leaf_kind(leaf) == "identity" for leaf in chain)


def composed_identity_optimized(chain: Sequence[Any], x: Any) -> Any:
    """Apply only the non-identity leaves; identity leaves are no-ops."""
    for leaf in chain:
        if _leaf_kind(leaf) == "identity":
            continue
        x = leaf._apply_core(x)
    return x


ZERO_SPEC = registry.register(
    KernelSpec(
        name="composed-zero-annihilation",
        generic=composed_chain_apply_generic,
        optimized=composed_zero_optimized,
        applicable=composed_zero_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::TestComposedZeroAnnihilation::test_matches_generic"
        ),
        benchmark_id="kernels.composed_zero_annihilation",
        rtol=0.0,
        atol=0.0,
        dispatch_key=_COMPOSED_APPLY_KEY,
        priority=20,
        notes=(
            "A @ 0 @ B = 0: collapse the whole chain to codomain.zeros(). Higher "
            "priority than identity elision so a chain with both routes here."
        ),
    )
)


IDENTITY_SPEC = registry.register(
    KernelSpec(
        name="composed-identity-elision",
        generic=composed_chain_apply_generic,
        optimized=composed_identity_optimized,
        applicable=composed_identity_applicable,
        correctness_ref=(
            "tests/kernels/test_kernels_match_generic.py"
            "::TestComposedIdentityElision::test_matches_generic"
        ),
        benchmark_id="kernels.composed_identity_elision",
        rtol=0.0,
        atol=0.0,
        dispatch_key=_COMPOSED_APPLY_KEY,
        priority=10,
        notes=(
            "A @ I @ B = A @ B: skip identity leaves. Exact; only fires on "
            "chains built directly (pytree round-trip, _convert), since "
            "make_composed elides identities at construction."
        ),
    )
)
