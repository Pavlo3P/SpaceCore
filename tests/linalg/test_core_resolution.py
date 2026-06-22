"""Safe consumption of check-free cores in the iterative-solver hot loops.

The solvers validate their inputs once, at entry, then run the loop through the
operators'/spaces' check-free ``_*_core`` kernels. The resolution is *safe*: a
core is used only when it is consistent with the public method, so a user space
that overrides ``inner`` (custom geometry) without overriding ``_inner_core``
keeps its override. These tests pin both properties.
"""
from __future__ import annotations

import numpy as np

import spacecore as sc
from spacecore.linalg._utils import resolve_apply, resolve_core


def make_ctx():
    return sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")


class _WeightedSpace(sc.DenseCoordinateSpace):
    """Overrides ``inner`` (weighted geometry) but inherits ``_inner_core``."""

    def __init__(self, weights, ctx):
        w = ctx.asarray(weights)
        super().__init__(tuple(w.shape), ctx)
        self.weights = w

    def inner(self, x, y):
        return self.ops.vdot(x, self.weights * y)

    def _convert(self, new_ctx):
        return _WeightedSpace(new_ctx.asarray(self.weights), new_ctx)


# ---------------------------------------------------------------------------
# resolve_core safety
# ---------------------------------------------------------------------------
def test_resolve_core_uses_core_for_builtin_space():
    X = sc.DenseCoordinateSpace((3,), make_ctx())
    # inner and _inner_core come from the same class -> the core is consistent.
    assert resolve_core(X, "inner", "_inner_core") == X._inner_core


def test_resolve_core_defers_to_overriding_subclass():
    ctx = make_ctx()
    space = _WeightedSpace([1.0, 4.0], ctx)
    inner = resolve_core(space, "inner", "_inner_core")
    # The override wins over the inherited built-in core.
    assert inner == space.inner
    x = ctx.asarray([1.0, 2.0])
    y = ctx.asarray([3.0, 4.0])
    np.testing.assert_allclose(
        float(inner(x, y)), float(space.ops.vdot(x, space.weights * y))
    )
    # The euclidean core would give a different (wrong) answer here.
    assert float(inner(x, y)) != float(space._inner_core(x, y))


def test_resolve_apply_uses_core_for_builtin_linop():
    ctx = make_ctx()
    X = sc.DenseCoordinateSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray(np.eye(3)), X, X, ctx)
    assert resolve_apply(A) == A._apply_core


# ---------------------------------------------------------------------------
# Solvers validate once at entry, not per iteration
# ---------------------------------------------------------------------------
def _count_member_checks(space, fn):
    counter = [0]
    original = space._check_member

    def counting(v):
        counter[0] += 1
        return original(v)

    space._check_member = counting
    try:
        fn()
    finally:
        space._check_member = original
    return counter[0]


def test_cg_membership_checks_are_constant_in_iterations():
    ctx = make_ctx()
    X = sc.DenseCoordinateSpace((5,), ctx)
    rng = np.random.default_rng(0)
    M = rng.standard_normal((5, 5))
    A = sc.DenseLinOp(ctx.asarray(M @ M.T + 5 * np.eye(5)), X, X, ctx)
    b = ctx.asarray(rng.standard_normal(5))

    n_short = _count_member_checks(X, lambda: sc.cg(A, b, tol=1e-15, maxiter=3))
    n_long = _count_member_checks(X, lambda: sc.cg(A, b, tol=1e-15, maxiter=50))
    # Entry validates b and x0 once; the loop adds nothing, so the count does not
    # grow with the iteration budget.
    assert n_short == n_long, f"CG re-validates per iteration: {n_short} vs {n_long}"
    assert n_long <= 4


def test_lanczos_weighted_geometry_still_honored():
    """The safe resolution keeps a custom-geometry space correct end to end."""
    ctx = make_ctx()
    space = _WeightedSpace([1.0, 4.0], ctx)
    matrix = ctx.asarray([[2.0, 1.0], [0.25, 0.75]])
    op = sc.MatrixFreeLinOp(lambda x: matrix @ x, lambda x: matrix @ x, space, space, ctx)
    result = sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=2, tol=1e-12)
    expected = float(np.min(np.linalg.eigvals(np.asarray(matrix)).real))
    np.testing.assert_allclose(float(np.asarray(result.eigenvalue)), expected, rtol=1e-6, atol=1e-6)
