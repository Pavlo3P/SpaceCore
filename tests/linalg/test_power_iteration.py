"""Tests for :func:`spacecore.power_iteration` and ``_SelfAdjointAction``.

Checklist section 8, ``power_iteration`` + internal ``_SelfAdjointAction``:

* Estimates the dominant (largest-modulus) eigenpair on a ``LinOp`` and on a
  ``QuadraticForm`` Hessian action, including negative-dominant and zero
  operators.
* Geometry awareness: metric Rayleigh quotient and residual norm on weighted
  spaces.
* Quadratic-form fast scalar diagnostics (``hess_quad`` / ``hess_residual_norm``)
  and the generic ``domain.inner`` fallback.
* Public dispatch builds a ``_SelfAdjointAction`` before the numeric core; the
  core carries no dispatch logic and no ``check_every`` argument.
* Rejects non-Hermitian and rectangular operators; validates ``check_every``.
* JAX ``jit`` round-trip preserves correctness.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import spacecore as sc

from spacecore.linalg import _power as power_mod
from tests._helpers import has_jax, jax_real_dtype, to_numpy
from tests.linalg._helpers import backend_params, make_ctx


def _metric_spd_matrix(ctx):
    weights = np.asarray([2.0, 5.0, 11.0])
    symmetric_spd = np.asarray([[6.0, 1.0, 0.5], [1.0, 5.0, -0.25], [0.5, -0.25, 4.0]])
    return ctx.asarray(symmetric_spd / weights[:, None])


def _weighted_space(ctx):
    return sc.DenseCoordinateSpace(
        (3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0, 11.0]))
    )


# ===========================================================================
# Dominant eigenpair
# ===========================================================================
class TestDominantEigenpair:
    @pytest.mark.parametrize("backend_name,dtype", backend_params())
    def test_estimates_dominant_eigenpair(self, backend_name, dtype):
        ctx = make_ctx(backend_name, dtype)
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

        result = sc.power_iteration(A, x0=ctx.asarray([1.0, 1.0]), tol=1e-5, maxiter=60)

        np.testing.assert_allclose(to_numpy(result.eigenvalue), 5.0, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.abs(to_numpy(result.eigenvector)), [0.0, 1.0], rtol=1e-4, atol=1e-4)
        assert bool(to_numpy(result.converged))

    def test_converges_for_negative_dominant_eigenvalue(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DiagonalLinOp(ctx.asarray([-7.0, 2.0]), space, ctx)

        result = sc.power_iteration(A, x0=ctx.asarray([1.0, 1.0]), tol=1e-8, maxiter=60)

        np.testing.assert_allclose(to_numpy(result.eigenvalue), -7.0, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(np.abs(to_numpy(result.eigenvector)), [1.0, 0.0], atol=1e-8)
        assert bool(to_numpy(result.converged))

    def test_zero_operator_does_not_produce_nan(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DiagonalLinOp(ctx.asarray([0.0, 0.0]), space, ctx)

        result = sc.power_iteration(A, x0=ctx.asarray([1.0, 1.0]), tol=1e-8, maxiter=10)

        assert np.all(np.isfinite(to_numpy(result.eigenvector)))
        np.testing.assert_allclose(to_numpy(result.eigenvalue), 0.0, atol=1e-12)
        np.testing.assert_allclose(to_numpy(result.residual_norm), 0.0, atol=1e-12)
        assert bool(to_numpy(result.converged))

    @pytest.mark.parametrize("operator_kind", ["dense", "matrix-free"])
    def test_symmetric_reference_and_deterministic_start(self, operator_kind):
        ctx = make_ctx(check_level="standard")
        space = sc.DenseCoordinateSpace((3,), ctx)
        matrix = np.asarray([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
        if operator_kind == "matrix-free":
            backend = ctx.asarray(matrix)
            operator = sc.MatrixFreeLinOp(lambda x: backend @ x, lambda y: backend.T @ y, space, space, ctx)
        else:
            operator = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)
        x0 = ctx.asarray([1.0, -0.25, 0.5])

        first = sc.power_iteration(operator, x0=x0, tol=1e-10, maxiter=100)
        second = sc.power_iteration(operator, x0=x0, tol=1e-10, maxiter=100)
        expected = np.linalg.eigvalsh(matrix)[-1]
        residual = operator.apply(first.eigenvector) - first.eigenvalue * first.eigenvector

        np.testing.assert_allclose(to_numpy(first.eigenvalue), expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(to_numpy(space.norm(residual)), 0.0, atol=1e-9)
        np.testing.assert_allclose(to_numpy(first.eigenvalue), to_numpy(second.eigenvalue))
        np.testing.assert_allclose(to_numpy(first.eigenvector), to_numpy(second.eigenvector))
        assert int(to_numpy(first.num_iters)) == int(to_numpy(second.num_iters))
        assert bool(to_numpy(first.converged))


# ===========================================================================
# Application accounting & check_every backward-compat
# ===========================================================================
class TestApplicationAccounting:
    def test_one_application_per_iteration_after_initial_product(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        matrix = ctx.asarray([[2.0, 0.0], [0.0, 5.0]])
        calls = {"apply": 0}

        def apply(x):
            calls["apply"] += 1
            return matrix @ x

        A = sc.MatrixFreeLinOp(apply, apply, space, space, ctx)
        result = sc.power_iteration(A, x0=ctx.asarray([1.0, 1.0]), tol=1e-4, maxiter=60, check_every=1000)

        assert calls["apply"] == int(result.num_iters) + 1
        np.testing.assert_allclose(to_numpy(result.eigenvalue), 5.0, rtol=1e-4, atol=1e-4)
        assert bool(to_numpy(result.converged))

    def test_large_check_every_no_longer_delays_convergence(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DiagonalLinOp(ctx.asarray([1.0, 5.0]), space, ctx)

        result = sc.power_iteration(
            A, x0=ctx.asarray([0.0, 1.0]), tol=1e-12, maxiter=65, check_every=10_000
        )

        assert int(to_numpy(result.num_iters)) == 1
        np.testing.assert_allclose(to_numpy(result.eigenvalue), 5.0, rtol=1e-12, atol=1e-12)
        assert bool(to_numpy(result.converged))

    def test_validates_check_every_for_backward_compatibility(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0]), space, ctx)

        sc.power_iteration(A, check_every=1, maxiter=1)
        with pytest.raises(ValueError, match="check_every"):
            sc.power_iteration(A, check_every=0, maxiter=1)


# ===========================================================================
# Geometry awareness
# ===========================================================================
class TestWeightedGeometry:
    def test_uses_metric_rayleigh_quotient_and_norm(self):
        ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
        space = _weighted_space(ctx)
        matrix = _metric_spd_matrix(ctx)
        A = sc.DenseLinOp(matrix, space, space, ctx)
        x0 = ctx.asarray([1.0, 0.25, -0.5])

        result = sc.power_iteration(A, x0=x0, tol=1e-10, maxiter=80)
        Ax = A.apply(result.eigenvector)
        rayleigh = space.inner(result.eigenvector, Ax) / space.inner(result.eigenvector, result.eigenvector)
        residual = space.add(Ax, space.scale(-result.eigenvalue, result.eigenvector))
        expected = max(np.linalg.eigvals(to_numpy(matrix)).real)

        np.testing.assert_allclose(to_numpy(result.eigenvalue), to_numpy(rayleigh), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(to_numpy(result.eigenvalue), expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(to_numpy(result.residual_norm), to_numpy(space.norm(residual)), atol=1e-12)
        assert bool(result.converged)


# ===========================================================================
# QuadraticForm input
# ===========================================================================
class TestQuadraticForm:
    def test_accepts_quadratic_form_hessian_action(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
        q = sc.LinOpQuadraticForm(op, ctx=ctx)
        x0 = ctx.asarray([1.0, 1.0])

        op_result = sc.power_iteration(op, x0=x0, tol=1e-5, maxiter=60)
        q_result = sc.power_iteration(q, x0=x0, tol=1e-5, maxiter=60)

        np.testing.assert_allclose(to_numpy(q_result.eigenvalue), to_numpy(op_result.eigenvalue))
        np.testing.assert_allclose(
            np.abs(to_numpy(q_result.eigenvector)), np.abs(to_numpy(op_result.eigenvector)), rtol=1e-6, atol=1e-6
        )

    def test_uses_fast_scalar_diagnostics_when_available(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)

        class CountingQuadraticForm(sc.QuadraticForm):
            def __init__(self):
                super().__init__(space, ctx)
                self.diagonal = ctx.asarray([2.0, 5.0])
                self.calls = {"hess_apply": 0, "hess_quad": 0, "hess_residual_norm": 0}

            def value(self, x):
                return 0.5 * self.domain.inner(x, self.hess_apply(x))

            def grad(self, x):
                return self.hess_apply(x)

            def hess_apply(self, x):
                self.calls["hess_apply"] += 1
                return self.diagonal * x

            def hess_quad(self, x, Hx=None):
                self.calls["hess_quad"] += 1
                if Hx is None:
                    Hx = self.diagonal * x
                np.testing.assert_allclose(to_numpy(Hx), to_numpy(self.diagonal * x))
                return self.domain.inner(x, Hx)

            def hess_residual_norm(self, x, Hx, eigenvalue):
                self.calls["hess_residual_norm"] += 1
                return self.domain.norm(Hx - eigenvalue * x)

            def tree_flatten(self):
                return (), ()

            @classmethod
            def tree_unflatten(cls, aux, children):
                return cls()

        q = CountingQuadraticForm()
        result = sc.power_iteration(q, x0=ctx.asarray([1.0, 1.0]), tol=1e-4, maxiter=60)

        assert q.calls["hess_apply"] == int(result.num_iters) + 1
        assert q.calls["hess_quad"] == int(result.num_iters)
        assert q.calls["hess_residual_norm"] == int(result.num_iters)
        np.testing.assert_allclose(to_numpy(result.eigenvalue), 5.0, rtol=1e-4, atol=1e-4)
        assert bool(to_numpy(result.converged))

    def test_falls_back_to_domain_inner_without_fast_diagnostics(self):
        ctx = make_ctx()

        class CountingVectorSpace(sc.DenseCoordinateSpace):
            def __init__(self, shape, ctx):
                super().__init__(shape, ctx)
                self.inner_calls = 0

            def inner(self, x, y):
                self.inner_calls += 1
                return super().inner(x, y)

            def _convert(self, new_ctx):
                if new_ctx == self.ctx:
                    return self
                return CountingVectorSpace(self.shape, new_ctx)

        space = CountingVectorSpace((2,), ctx)

        class FallbackQuadraticForm(sc.QuadraticForm):
            def __init__(self):
                super().__init__(space, ctx)
                self.diagonal = ctx.asarray([2.0, 5.0])
                self.hess_apply_calls = 0

            def value(self, x):
                return 0.5 * self.domain.inner(x, self.hess_apply(x))

            def grad(self, x):
                return self.hess_apply(x)

            def hess_apply(self, x):
                self.hess_apply_calls += 1
                return self.diagonal * x

            def tree_flatten(self):
                return (), ()

            @classmethod
            def tree_unflatten(cls, aux, children):
                return cls()

        q = FallbackQuadraticForm()
        result = sc.power_iteration(q, x0=ctx.asarray([1.0, 1.0]), tol=1e-4, maxiter=60)

        assert q.hess_apply_calls == int(result.num_iters) + 1
        assert q.domain.inner_calls >= int(result.num_iters)
        np.testing.assert_allclose(to_numpy(result.eigenvalue), 5.0, rtol=1e-4, atol=1e-4)


# ===========================================================================
# Dispatch & _SelfAdjointAction internals
# ===========================================================================
class TestDispatch:
    def test_dispatches_quadratic_form_before_core(self, monkeypatch):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
        q = sc.LinOpQuadraticForm(op, ctx=ctx)
        x0 = ctx.asarray([1.0, 0.0])
        captured = {}

        def fake_core(action, x, tol, maxiter):
            captured["action"] = action
            captured["x"] = x
            return ctx.asarray(0.0), x, ctx.asarray(True), 0, ctx.asarray(0.0)

        monkeypatch.setattr(power_mod, "_power_iteration_core", fake_core)
        result = power_mod.power_iteration(q, x0=x0, maxiter=1)

        assert result.eigenvector is x0
        action = captured["action"]
        assert isinstance(action, power_mod._SelfAdjointAction)
        assert action.domain == q.domain
        # _SelfAdjointAction exposes ops/dtype from its context.
        assert action.ops is ctx.ops
        assert action.dtype == ctx.dtype
        probe = ctx.asarray([1.0, 2.0])
        np.testing.assert_allclose(action.apply(probe), q.hess_apply(probe))

    def test_core_has_no_dispatch_logic(self):
        source = inspect.getsource(power_mod._power_iteration_core)
        for token in ("isinstance", "hasattr", "getattr", "_SelfAdjointAction(", "PowerIterationResult("):
            assert token not in source

    def test_core_has_no_check_every_argument(self):
        assert "check_every" not in inspect.signature(power_mod._power_iteration_core).parameters


# ===========================================================================
# Rejections
# ===========================================================================
class TestRejections:
    def test_rejects_known_non_hermitian_operator(self):
        ctx = make_ctx()
        space = sc.DenseCoordinateSpace((2,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, 3.0]]), space, space, ctx)
        with pytest.raises(ValueError, match="Hermitian"):
            sc.power_iteration(A)

    def test_rejects_rectangular_operator(self):
        ctx = make_ctx()
        domain = sc.DenseCoordinateSpace((2,), ctx)
        codomain = sc.DenseCoordinateSpace((3,), ctx)
        A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)
        with pytest.raises(ValueError, match="square LinOp"):
            sc.power_iteration(A)

    def test_rejects_non_linop_non_quadratic(self):
        with pytest.raises(TypeError, match="LinOp or QuadraticForm"):
            sc.power_iteration(np.eye(2))


# ===========================================================================
# JAX jit
# ===========================================================================
@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
class TestJit:
    def test_jit_compiles_with_operator_argument(self):
        import jax

        ctx = make_ctx("jax", jax_real_dtype())
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

        run = jax.jit(lambda A, x: sc.power_iteration(A, x0=x, maxiter=60).eigenvalue)
        np.testing.assert_allclose(to_numpy(run(op, ctx.asarray([1.0, 1.0]))), 5.0, rtol=1e-5, atol=1e-5)

    def test_jit_compiles_with_quadratic_form_argument(self):
        import jax

        ctx = make_ctx("jax", jax_real_dtype())
        space = sc.DenseCoordinateSpace((2,), ctx)
        op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
        q = sc.LinOpQuadraticForm(op, ctx=ctx)

        run = jax.jit(lambda quad, x: sc.power_iteration(quad, x0=x, maxiter=60).eigenvalue)
        np.testing.assert_allclose(to_numpy(run(q, ctx.asarray([1.0, 1.0]))), 5.0, rtol=1e-5, atol=1e-5)
