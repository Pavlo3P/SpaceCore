"""Contract and guard-rail tests for the ``spacecore.optimize`` adapters."""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore.optimize._common import (
    coordinate_gradient,
    domain_with_geometry,
    require_functional,
    require_real_field,
)

from tests._helpers import has_jax
from tests.linalg._helpers import make_ctx
from tests.optimize._helpers import euclidean_problem, has_optax


@pytest.fixture
def ctx():
    return make_ctx("numpy", np.float64)


class TestFunctionalGuard:
    @pytest.mark.parametrize("fn", [sc.minimize_scipy, sc.line_search_scipy, sc.minimize_optax])
    def test_non_functional_rejected(self, fn):
        with pytest.raises(TypeError, match="requires a spacecore Functional"):
            if fn is sc.line_search_scipy:
                fn(object(), [1.0], [1.0])
            elif fn is sc.minimize_optax:
                fn(object(), [1.0], None, max_iter=1)
            else:
                fn(object(), [1.0])


class TestComplexFieldGuard:
    def test_minimize_scipy_rejects_complex(self, ):
        ctx = make_ctx("numpy", np.complex128)
        X = sc.DenseCoordinateSpace((2,), ctx)
        F = sc.LinOpQuadraticForm(sc.IdentityLinOp(X, ctx))
        with pytest.raises(ValueError, match="complex domain"):
            sc.minimize_scipy(F, X.zeros())

    def test_line_search_scipy_rejects_complex(self):
        ctx = make_ctx("numpy", np.complex128)
        X = sc.DenseCoordinateSpace((2,), ctx)
        F = sc.LinOpQuadraticForm(sc.IdentityLinOp(X, ctx))
        with pytest.raises(ValueError, match="complex domain"):
            sc.line_search_scipy(F, X.zeros(), X.zeros())

    def test_require_real_field_helper(self, ctx):
        X = sc.DenseCoordinateSpace((2,), ctx)
        require_real_field(X, "test")  # real: no raise
        cctx = make_ctx("numpy", np.complex128)
        with pytest.raises(ValueError, match="complex domain"):
            require_real_field(sc.DenseCoordinateSpace((2,), cctx), "test")


class TestInnerProductGuard:
    def test_domain_without_geometry_rejected(self):
        """A domain without inner-product geometry has no Riesz map; refuse it."""
        from types import SimpleNamespace

        # The guard only inspects ``F.domain``; a non-InnerProductSpace domain
        # (here a bare object) must be rejected with a Riesz-map message.
        F = SimpleNamespace(domain=object())
        with pytest.raises(TypeError, match="InnerProductSpace"):
            domain_with_geometry(F, "test")

    def test_inner_product_domain_accepted(self, ctx):
        X, F, _ = euclidean_problem(ctx)
        assert domain_with_geometry(F, "test") is X


class TestOptaxBackendGuard:
    def test_optax_rejects_numpy_backend(self, ctx):
        if not has_optax():
            pytest.skip("optax is not installed")
        import optax

        X, F, _ = euclidean_problem(ctx)
        with pytest.raises(TypeError, match="JAX-backed"):
            sc.minimize_optax(F, X.zeros(), optax.sgd(0.1), max_iter=1)

    @pytest.mark.skipif(not has_jax(), reason="jax is not installed")
    def test_optax_negative_max_iter(self):
        if not has_optax():
            pytest.skip("optax is not installed")
        import optax

        ctx = make_ctx("jax", np.float32)
        X, F, _ = euclidean_problem(ctx)
        with pytest.raises(ValueError, match="non-negative"):
            sc.minimize_optax(F, X.zeros(), optax.sgd(0.1), max_iter=-1)


class TestScipyUnsupportedParameters:
    def test_minimize_scipy_rejects_args(self, ctx):
        X, F, _ = euclidean_problem(ctx)
        with pytest.raises(ValueError, match="does not support the SciPy 'args'"):
            sc.minimize_scipy(F, X.zeros(), args=(1.0,))

    def test_line_search_scipy_rejects_args(self, ctx):
        X, F, _ = euclidean_problem(ctx)
        with pytest.raises(ValueError, match="does not support the SciPy 'args'"):
            sc.line_search_scipy(F, X.zeros(), X.zeros(), args=(1.0,))

    def test_minimize_scipy_rejects_complex_step_jac(self, ctx):
        X, F, _ = euclidean_problem(ctx)
        with pytest.raises(ValueError, match="jac='cs'"):
            sc.minimize_scipy(F, X.zeros(), jac="cs", method="BFGS")


def test_require_functional_passes_through(ctx):
    X, F, _ = euclidean_problem(ctx)
    assert require_functional(F, "test") is F


def test_coordinate_gradient_centralizes_riesz(ctx):
    X, F, _ = euclidean_problem(ctx)
    x = ctx.asarray([1.0, 2.0])
    np.testing.assert_array_equal(
        np.asarray(coordinate_gradient(F, X, x)),
        np.asarray(X.riesz(F.grad(x))),
    )
