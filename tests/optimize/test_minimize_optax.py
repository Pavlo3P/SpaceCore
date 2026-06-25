"""Tests for :func:`spacecore.minimize_optax` (ADR-018).

optax is JAX-native, so these tests require both jax and optax. They cover
convergence on Euclidean and weighted dense spaces, the pytree pass-through for a
tree space, the callback, the ``steps=0`` no-op, and the optimizer-state usage.
"""
from __future__ import annotations

import numpy as np
import optree
import pytest

import spacecore as sc

from tests._helpers import has_jax, to_numpy
from tests.linalg._helpers import make_ctx
from tests.optimize._helpers import euclidean_problem, has_optax, weighted_problem

pytestmark = pytest.mark.skipif(
    not (has_jax() and has_optax()), reason="minimize_optax requires jax and optax"
)


def _jax_ctx():
    return make_ctx("jax", np.float32)


def test_euclidean_convergence():
    import optax

    ctx = _jax_ctx()
    X, F, x_star = euclidean_problem(ctx)

    x = sc.minimize_optax(F, X.zeros(), optax.adam(1e-1), steps=1000)

    np.testing.assert_allclose(to_numpy(x), x_star, atol=1e-2)


def test_weighted_convergence():
    """The riesz coordinate-gradient handoff drives optax to the metric minimizer."""
    import optax

    ctx = _jax_ctx()
    X, F, x_star = weighted_problem(ctx)

    x = sc.minimize_optax(F, X.zeros(), optax.adam(5e-2), steps=4000)

    np.testing.assert_allclose(to_numpy(x), x_star, atol=2e-2)


def test_tree_space_pytree_passthrough():
    import optax

    ctx = _jax_ctx()
    treedef = optree.tree_structure((0, 0))
    Xa = sc.DenseCoordinateSpace((2,), ctx)
    Xb = sc.DenseCoordinateSpace((1,), ctx)
    X = sc.TreeSpace(treedef, (Xa, Xb), ctx=ctx)
    F = sc.LinOpQuadraticForm(sc.IdentityLinOp(X, ctx))
    x0 = (ctx.asarray([3.0, -1.0]), ctx.asarray([2.0]))

    x = sc.minimize_optax(F, x0, optax.sgd(0.2), steps=300)

    assert isinstance(x, tuple) and len(x) == 2
    np.testing.assert_allclose(to_numpy(X.flatten(x)), 0.0, atol=1e-3)


def test_tree_element_x0_is_normalized():
    """A bound ``TreeElement`` x0 is normalized so apply_updates does not collide."""
    import optax

    ctx = _jax_ctx()
    treedef = optree.tree_structure((0, 0))
    Xa = sc.DenseCoordinateSpace((2,), ctx)
    Xb = sc.DenseCoordinateSpace((1,), ctx)
    X = sc.TreeSpace(treedef, (Xa, Xb), ctx=ctx)
    F = sc.LinOpQuadraticForm(sc.IdentityLinOp(X, ctx))
    # The idiomatic SpaceCore element is a bound TreeElement, which is its own
    # pytree and previously collided with the raw-tuple gradient at step 1.
    x0 = X.element((ctx.asarray([3.0, -1.0]), ctx.asarray([2.0])))
    assert isinstance(x0, sc.TreeElement)

    x = sc.minimize_optax(F, x0, optax.sgd(0.2), steps=50)

    np.testing.assert_allclose(to_numpy(X.flatten(x)), 0.0, atol=1e-3)


def test_callback_records_trajectory():
    import optax

    ctx = _jax_ctx()
    X, F, _ = euclidean_problem(ctx)
    values = []

    def callback(step, params):
        values.append(float(F.value(params)))

    sc.minimize_optax(F, X.zeros(), optax.sgd(1e-1), steps=10, callback=callback)

    assert len(values) == 10
    assert values[-1] < values[0]  # objective decreased over the run


def test_zero_steps_returns_initial_point():
    import optax

    ctx = _jax_ctx()
    X, F, _ = euclidean_problem(ctx)
    x0 = ctx.asarray([0.3, -0.7])

    x = sc.minimize_optax(F, x0, optax.adam(1e-1), steps=0)

    np.testing.assert_array_equal(to_numpy(x), to_numpy(x0))
