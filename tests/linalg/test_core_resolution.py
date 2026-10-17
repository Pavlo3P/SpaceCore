"""Iterative solvers run on check-free cores and still honour user overrides.

The solvers validate their inputs once, at entry, then call the operators' and
spaces' ``_*_core`` methods. A user space or operator that overrides a public method
(``inner``, ``apply``) without decorating it has that override installed as its core
(``install_public_checks``), so the solvers use it. These tests pin both properties.
"""
from __future__ import annotations

import numpy as np

import spacecore as sc


def make_ctx():
    sc.set_check_level("standard")
    return sc.Context(sc.NumpyOps(), dtype=np.float64)


class _WeightedSpace(sc.DenseCoordinateSpace):
    """A custom geometry written as the ``_inner_core`` the public ``inner`` wraps."""

    def __init__(self, weights, ctx):
        w = ctx.asarray(weights)
        super().__init__(tuple(w.shape), ctx)
        self.weights = w

    def _inner_core(self, x, y):
        return self.ops.vdot(x, self.weights * y)

    def _convert(self, new_ctx):
        return _WeightedSpace(new_ctx.asarray(self.weights), new_ctx)


# ---------------------------------------------------------------------------
# Public overrides are honoured by the solvers
# ---------------------------------------------------------------------------
class _PublicInnerSpace(sc.DenseCoordinateSpace):
    """A custom geometry written the older way: an undecorated public ``inner``."""

    def __init__(self, weights, ctx):
        w = ctx.asarray(weights)
        super().__init__(tuple(w.shape), ctx)
        self.weights = w

    def inner(self, x, y):
        return self.ops.vdot(x, self.weights * y)

    def _convert(self, new_ctx):
        return _PublicInnerSpace(new_ctx.asarray(self.weights), new_ctx)


def test_custom_core_is_what_inner_and_solvers_use():
    ctx = make_ctx()
    space = _WeightedSpace([1.0, 4.0], ctx)
    x, y = ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0])
    np.testing.assert_allclose(float(space.inner(x, y)), float(space.ops.vdot(x, space.weights * y)))


def test_a_public_inner_override_is_the_core_the_solvers_use():
    ctx = make_ctx()
    space = _PublicInnerSpace([1.0, 4.0], ctx)
    assert type(space)._inner_core is vars(_PublicInnerSpace)["_inner_core"]
    matrix = ctx.asarray([[2.0, 1.0], [0.25, 0.75]])
    op = sc.MatrixFreeLinOp(lambda x: matrix @ x, lambda x: matrix @ x, space, space, ctx)
    result = sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=2, tol=1e-12)
    expected = float(np.min(np.linalg.eigvals(np.asarray(matrix)).real))
    np.testing.assert_allclose(float(np.asarray(result.eigenvalue)), expected, rtol=1e-6, atol=1e-6)


def test_a_public_apply_override_is_what_cg_applies():
    ctx = make_ctx()
    X = sc.DenseCoordinateSpace((2,), ctx)
    calls = []

    class _Diagonal(sc.LinOp):
        _fields = ("dom", "cod", "ctx")

        def apply(self, x):  # undecorated public override
            calls.append(1)
            return ctx.asarray([2.0, 4.0]) * x

        def rapply(self, y):
            return self.apply(y)

        def is_hermitian(self):
            return True

    result = sc.cg(_Diagonal(X, X, ctx), ctx.asarray([2.0, 4.0]), tol=1e-12)
    np.testing.assert_allclose(np.asarray(result.x), [1.0, 1.0])
    assert len(calls) > 1


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
