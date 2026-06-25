"""Tests for :func:`spacecore.minimize_scipy` (ADR-018).

Covers: convergence to a known minimizer on Euclidean and weighted spaces and
across NumPy/JAX backends; the structured (tree) round-trip; the ``x_element``
marshalling-back field; the ``jac`` policy (riesz default, finite-difference,
forwarded callable/string); and ``**kw`` forwarding (``bounds``, ``method``).
"""
from __future__ import annotations

import numpy as np
import optree
import pytest

import spacecore as sc

from tests._helpers import to_numpy
from tests.linalg._helpers import make_ctx, numpy_jax_params
from tests.optimize._helpers import euclidean_problem, flat_fun, weighted_problem


class TestConvergence:
    @pytest.mark.parametrize("backend_name,dtype", numpy_jax_params())
    def test_euclidean_minimizer(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        X, F, x_star = euclidean_problem(ctx)

        result = sc.minimize_scipy(F, X.zeros(), options={"gtol": 1e-12, "ftol": 1e-15})

        assert bool(result.success)
        atol = 1e-5 if backend_name == "numpy" else 1e-3
        np.testing.assert_allclose(to_numpy(result.x_element), x_star, atol=atol)

    def test_weighted_minimizer(self):
        ctx = make_ctx("numpy", np.float64)
        X, F, x_star = weighted_problem(ctx)

        result = sc.minimize_scipy(F, X.zeros(), options={"gtol": 1e-12, "ftol": 1e-15})

        assert bool(result.success)
        np.testing.assert_allclose(to_numpy(result.x_element), x_star, rtol=1e-5, atol=1e-6)

    def test_tree_space_roundtrip(self):
        """Structured (tree) elements flatten/unflatten transparently."""
        ctx = make_ctx("numpy", np.float64)
        treedef = optree.tree_structure((0, 0))
        Xa = sc.DenseCoordinateSpace((2,), ctx)
        Xb = sc.DenseCoordinateSpace((1,), ctx)
        X = sc.TreeSpace(treedef, (Xa, Xb), ctx=ctx)
        F = sc.LinOpQuadraticForm(sc.IdentityLinOp(X, ctx))
        x0 = (ctx.asarray([3.0, -1.0]), ctx.asarray([2.0]))

        result = sc.minimize_scipy(F, x0, options={"gtol": 1e-10})

        assert isinstance(result.x_element, tuple) and len(result.x_element) == 2
        np.testing.assert_allclose(to_numpy(X.flatten(result.x_element)), 0.0, atol=1e-6)


class TestMarshalling:
    def test_x_element_is_domain_element_and_x_stays_flat(self):
        ctx = make_ctx("numpy", np.float64)
        X, F, x_star = euclidean_problem(ctx)

        result = sc.minimize_scipy(F, X.zeros())

        # ``x`` keeps SciPy's flat convention; ``x_element`` is the domain element.
        assert np.asarray(result.x).shape == (2,)
        np.testing.assert_allclose(to_numpy(X.flatten(result.x_element)), result.x, atol=1e-12)


class TestJacPolicy:
    def test_finite_difference_when_jac_false(self):
        """``jac=False`` lets SciPy approximate the gradient and still converges."""
        ctx = make_ctx("numpy", np.float64)
        X, F, x_star = euclidean_problem(ctx)

        result = sc.minimize_scipy(F, X.zeros(), jac=False, options={"gtol": 1e-8})

        assert bool(result.success)
        np.testing.assert_allclose(to_numpy(result.x_element), x_star, atol=1e-4)

    def test_string_jac_is_forwarded(self):
        ctx = make_ctx("numpy", np.float64)
        X, F, x_star = euclidean_problem(ctx)

        result = sc.minimize_scipy(F, X.zeros(), jac="2-point", options={"gtol": 1e-8})

        np.testing.assert_allclose(to_numpy(result.x_element), x_star, atol=1e-4)

    def test_riesz_jac_matches_a_manual_jac(self):
        """The default riesz jac equals a hand-written coordinate-gradient jac."""
        ctx = make_ctx("numpy", np.float64)
        X, F, x_star = weighted_problem(ctx)

        def manual_jac(v):
            x = X.unflatten(ctx.asarray(np.asarray(v)))
            return np.asarray(X.flatten(X.riesz(F.grad(x))), dtype=np.float64)

        auto = sc.minimize_scipy(F, X.zeros(), options={"gtol": 1e-12})
        manual = sc.minimize_scipy(F, X.zeros(), jac=manual_jac, options={"gtol": 1e-12})

        np.testing.assert_allclose(to_numpy(auto.x_element), to_numpy(manual.x_element), atol=1e-8)


class TestKeywordForwarding:
    def test_bounds_are_respected_in_flat_coordinates(self):
        ctx = make_ctx("numpy", np.float64)
        X, F, x_star = euclidean_problem(ctx)  # unconstrained min at (1, 2)

        result = sc.minimize_scipy(
            F, X.zeros(), method="L-BFGS-B", bounds=[(0.0, 0.5), (0.0, 0.5)]
        )

        x = to_numpy(result.x_element)
        assert np.all(x <= 0.5 + 1e-8) and np.all(x >= -1e-8)
        # The active-bound solution caps both coordinates at 0.5.
        np.testing.assert_allclose(x, [0.5, 0.5], atol=1e-6)

    def test_method_override(self):
        ctx = make_ctx("numpy", np.float64)
        X, F, x_star = euclidean_problem(ctx)

        result = sc.minimize_scipy(F, X.zeros(), method="CG", options={"gtol": 1e-8})

        np.testing.assert_allclose(to_numpy(result.x_element), x_star, atol=1e-4)


def test_objective_seen_by_scipy_matches_fun_helper():
    """The flattened objective is exactly ``F.value`` of the unflattened input."""
    ctx = make_ctx("numpy", np.float64)
    X, F, _ = weighted_problem(ctx)
    fun = flat_fun(F, X)
    v = np.array([0.2, -0.4, 0.9])
    np.testing.assert_allclose(fun(v), float(F.value(X.unflatten(ctx.asarray(v)))))
