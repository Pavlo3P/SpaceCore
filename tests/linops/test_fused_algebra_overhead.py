"""Near-zero-overhead contract for lazy LinOp algebra.

The lazy algebra operators (`ComposedLinOp`, `ScaledLinOp`, `SumLinOp`, the
adjoint view, identity, zero, matrix-free) share a ``_apply_core`` fast path:
the public ``apply``/``rapply``/``vapply`` validate the input and output once at
the operator boundary, then run a check-free fused core. Compositions are
flattened into a single ``_apply_chain`` at construction so an arbitrarily deep
``A @ B @ C @ ...`` runs one loop instead of re-walking a binary tree.

These tests pin two guarantees:

1. *Correctness* — the fused core is numerically identical to applying each
   operator's public ``apply`` in sequence.
2. *Overhead* — a composite validates membership exactly once on each side,
   regardless of how many operators are fused inside it (no per-link re-check).
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy


def _spy_member_checks(space):
    """Count every ``_check_member`` call on ``space``; return the counter list.

    ``_check_member`` is the single hot-path validator ``checked_method`` invokes
    for both input and output checks, so counting it measures exactly how many
    membership validations an apply performs. The instance method is patched
    (not the class) so ``type(space)`` is unchanged — the built-in fast paths
    that key on exact space type (e.g. ``SumLinOp``'s raw add) still apply.
    """
    counter = [0]
    original = space._check_member

    def counting(x):
        counter[0] += 1
        return original(x)

    space._check_member = counting
    return counter


@pytest.fixture
def ctx():
    # check_level="standard" is the default; make it explicit so the per-call
    # validation under test is actually active.
    return sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")


def _dense(ctx, X, Y, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((int(Y.shape[0]), int(X.shape[0])))
    return sc.DenseLinOp(ctx.asarray(A), X, Y, ctx)


def _endo_chain(ctx, X, seeds):
    """Build a composition ``ops[0] @ ops[1] @ ... `` of endomorphisms on ``X``."""
    ops = [_dense(ctx, X, X, s) for s in seeds]
    chain = ops[0]
    for op in ops[1:]:
        chain = chain @ op
    return chain, ops


def _free(ctx, X, seed):
    """A matrix-free endomorphism on ``X`` whose core is a pure backend callable.

    Its ``_apply_core``/``_rapply_core`` run no checked space ops, so every
    membership check observed during apply comes from an operator *boundary* —
    this isolates the overhead contributed by the lazy-algebra wrappers under
    test from any per-leaf internal validation (e.g. ``DenseLinOp.unflatten``).
    """
    rng = np.random.default_rng(seed)
    M = ctx.asarray(rng.standard_normal((int(X.shape[0]), int(X.shape[0]))))
    return sc.MatrixFreeLinOp(lambda v: M @ v, lambda v: M.T @ v, X, X, ctx)


def _free_chain(ctx, X, seeds):
    ops = [_free(ctx, X, s) for s in seeds]
    chain = ops[0]
    for op in ops[1:]:
        chain = chain @ op
    return chain, ops


# ---------------------------------------------------------------------------
# Chain fusion
# ---------------------------------------------------------------------------
def test_composition_flattens_into_one_chain(ctx):
    X = sc.DenseCoordinateSpace((4,), ctx)
    chain, ops = _endo_chain(ctx, X, range(5))
    # Every leaf fuses into a single flat chain; no nested ComposedLinOp nodes.
    assert len(chain._apply_chain) == len(ops)
    assert all(not isinstance(o, sc.ComposedLinOp) for o in chain._apply_chain)


def test_balanced_composition_also_flattens(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    a = _dense(ctx, X, X, 0)
    b = _dense(ctx, X, X, 1)
    c = _dense(ctx, X, X, 2)
    d = _dense(ctx, X, X, 3)
    # (a @ b) @ (c @ d) must still fuse to four leaves in application order.
    chain = (a @ b) @ (c @ d)
    assert chain._apply_chain == (d, c, b, a)


# ---------------------------------------------------------------------------
# Correctness of the fused core
# ---------------------------------------------------------------------------
def test_fused_apply_matches_sequential_reference(ctx):
    X = sc.DenseCoordinateSpace((4,), ctx)
    chain, ops = _endo_chain(ctx, X, range(5))
    x = ctx.asarray(np.arange(1.0, 5.0))

    ref = x
    for op in reversed(ops):  # right-most applied first
        ref = op.apply(ref)

    np.testing.assert_allclose(to_numpy(chain.apply(x)), to_numpy(ref))


def test_fused_rapply_matches_sequential_reference(ctx):
    X = sc.DenseCoordinateSpace((4,), ctx)
    chain, ops = _endo_chain(ctx, X, range(5))
    z = ctx.asarray(np.linspace(-1.0, 1.0, 4))

    ref = z
    for op in ops:  # adjoint applies left-most first
        ref = op.rapply(ref)

    np.testing.assert_allclose(to_numpy(chain.rapply(z)), to_numpy(ref))


def test_scaled_composed_adjoint_mix_matches_reference(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    A = _dense(ctx, X, X, 1)
    B = _dense(ctx, X, X, 2)
    op = 2.5 * (A.H @ B)
    x = ctx.asarray(np.asarray([1.0, -2.0, 3.0]))

    expected = 2.5 * A.rapply(B.apply(x))
    np.testing.assert_allclose(to_numpy(op.apply(x)), to_numpy(expected))


# ---------------------------------------------------------------------------
# Boundary-only validation (the overhead guarantee)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [1, 2, 5, 8])
def test_composition_adds_no_wrapper_checks_with_depth(ctx, depth):
    """The composition wrappers contribute exactly the two boundary checks.

    With pure matrix-free leaves (zero internal checks), an arbitrarily deep
    ``A @ B @ ...`` validates once on input and once on output — the fused chain
    never re-validates intermediates, so the count is flat in ``depth``.
    """
    X = sc.DenseCoordinateSpace((3,), ctx)
    counter = _spy_member_checks(X)
    chain, _ = _free_chain(ctx, X, range(depth))
    x = ctx.asarray(np.asarray([1.0, 2.0, 3.0]))

    counter[0] = 0
    chain.apply(x)
    assert counter[0] == 2, f"depth={depth} re-validated intermediates"


def test_scaled_of_composed_validates_once(ctx):
    X = sc.DenseCoordinateSpace((3,), ctx)
    counter = _spy_member_checks(X)
    op = 3.0 * (_free(ctx, X, 1) @ _free(ctx, X, 2))
    x = ctx.asarray(np.asarray([1.0, 0.0, -1.0]))

    counter[0] = 0
    op.apply(x)
    assert counter[0] == 2


def test_normal_equations_adjoint_validates_once(ctx):
    """``A.H @ A`` (the solver normal-equations path) checks only at the boundary."""
    X = sc.DenseCoordinateSpace((3,), ctx)
    counter = _spy_member_checks(X)
    A = _free(ctx, X, 7)
    op = A.H @ A
    x = ctx.asarray(np.asarray([2.0, -1.0, 0.5]))

    counter[0] = 0
    op.apply(x)
    assert counter[0] == 2


def test_identity_term_in_sum_does_not_recheck(ctx):
    """``A + I``: the identity term participates in the fused core without re-checking."""
    X = sc.DenseCoordinateSpace((3,), ctx)
    counter = _spy_member_checks(X)
    A = _free(ctx, X, 3)
    op = A + sc.IdentityLinOp(X, ctx)
    x = ctx.asarray(np.asarray([1.0, 2.0, 3.0]))

    expected = A.apply(x) + x
    counter[0] = 0
    y = op.apply(x)
    # Boundary check only: one input + one output for the sum.
    assert counter[0] == 2
    np.testing.assert_allclose(to_numpy(y), to_numpy(expected))


# ---------------------------------------------------------------------------
# JAX jit safety: the cached ``_apply_chain`` survives pytree round-trips
# ---------------------------------------------------------------------------
def test_composed_chain_is_jit_safe():
    pytest.importorskip("jax")
    import jax

    jctx = sc.Context(sc.JaxOps(), dtype=np.float32, check_level="none")
    X = sc.DenseCoordinateSpace((3,), jctx)
    chain, ops = _endo_chain(jctx, X, range(4))
    x = jctx.asarray(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    jitted = jax.jit(lambda op, v: op.apply(v))
    out = jitted(chain, x)
    np.testing.assert_allclose(to_numpy(out), to_numpy(chain.apply(x)), rtol=1e-5)
