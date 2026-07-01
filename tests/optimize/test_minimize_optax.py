"""Tests for :func:`spacecore.minimize_optax` (ADR-018, 0.4.2 W1).

``minimize_optax`` is the compiled, convergence-aware optax driver: the whole
loop runs inside ``jax.jit(jax.lax.while_loop(...))`` with the fused
``F.value_and_grad`` cached once per iteration. These tests (jax + optax) cover
convergence + early stop, the weighted-metric Riesz handoff, tree / bound-element
pass-through, the ``max_iter``/finiteness stop reasons, the ``project`` hook, the
four-column history + ``progress_callback`` replay, the guarantee that
``value_and_grad`` is evaluated once (not per iteration), and the input guards.
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


class _CountingQuadratic(sc.Functional):
    """``F(x) = 1/2 sum(x^2)`` that counts ``value_and_grad`` (Python) calls."""

    calls = 0

    def value(self, x, *args, **kwargs):
        return self.ops.sum(x * x) * 0.5

    def grad(self, x, *args, **kwargs):
        return x

    def value_and_grad(self, x, *args, **kwargs):
        type(self).calls += 1
        return self.value(x), self.grad(x)

    def tree_flatten(self):
        return (), (self.domain, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, ctx = aux
        return cls(domain, ctx)

    def _convert(self, new_ctx):
        return _CountingQuadratic(self.domain.convert(new_ctx), new_ctx)


# ===========================================================================
# Convergence + early stop + pass-through
# ===========================================================================
class TestConvergence:
    def test_euclidean_converges_and_stops_early(self):
        import optax

        ctx = _jax_ctx()
        X, F, x_star = euclidean_problem(ctx)
        res = sc.minimize_optax(F, X.zeros(), optax.adam(1e-1), max_iter=2000, tol=1e-5, verbose=0)

        assert res.success and res.status == 0
        assert res.message == "converged"
        assert res.num_iters < 2000  # stopped on the gradient tolerance, not the cap
        assert res.final_grad_norm <= 1e-5
        # One fused value_and_grad per iteration plus the initial evaluation
        # (adam does no line search, so n_linesearch_steps is 0).
        assert res.n_linesearch_steps == 0
        assert res.nfev == res.njev == res.num_iters + 1
        np.testing.assert_allclose(to_numpy(res.x_element), x_star, atol=1e-2)

    def test_weighted_metric_handoff(self):
        """X.riesz coordinate-gradient handoff drives optax to the metric minimizer."""
        import optax

        ctx = _jax_ctx()
        X, F, x_star = weighted_problem(ctx)
        res = sc.minimize_optax(F, X.zeros(), optax.adam(5e-2), max_iter=8000, tol=1e-5, verbose=0)

        assert res.success
        np.testing.assert_allclose(to_numpy(res.x_element), x_star, atol=2e-2)

    def test_tree_space_passthrough(self):
        import optax

        ctx = _jax_ctx()
        treedef = optree.tree_structure((0, 0))
        Xa = sc.DenseCoordinateSpace((2,), ctx)
        Xb = sc.DenseCoordinateSpace((1,), ctx)
        X = sc.TreeSpace(treedef, (Xa, Xb), ctx=ctx)
        F = sc.LinOpQuadraticForm(sc.IdentityLinOp(X, ctx))
        x0 = (ctx.asarray([3.0, -1.0]), ctx.asarray([2.0]))

        res = sc.minimize_optax(F, x0, optax.sgd(0.2), max_iter=500, tol=1e-4, verbose=0)

        # x_element is a bound domain element (not the raw optimizer pytree).
        assert isinstance(res.x_element, sc.TreeElement)
        X.check_member(res.x_element)
        np.testing.assert_allclose(to_numpy(X.flatten(res.x_element)), 0.0, atol=1e-3)

    def test_tree_element_x0_is_normalized(self):
        """A bound ``TreeElement`` x0 is normalized so apply_updates does not collide."""
        import optax

        ctx = _jax_ctx()
        treedef = optree.tree_structure((0, 0))
        Xa = sc.DenseCoordinateSpace((2,), ctx)
        Xb = sc.DenseCoordinateSpace((1,), ctx)
        X = sc.TreeSpace(treedef, (Xa, Xb), ctx=ctx)
        F = sc.LinOpQuadraticForm(sc.IdentityLinOp(X, ctx))
        x0 = X.element((ctx.asarray([3.0, -1.0]), ctx.asarray([2.0])))
        assert isinstance(x0, sc.TreeElement)

        res = sc.minimize_optax(F, x0, optax.sgd(0.2), max_iter=200, tol=1e-4, verbose=0)

        np.testing.assert_allclose(to_numpy(X.flatten(res.x_element)), 0.0, atol=1e-3)


# ===========================================================================
# Stop reasons: max-iter and nonfinite
# ===========================================================================
class TestStopReasons:
    def test_max_iter_cap(self):
        import optax

        ctx = _jax_ctx()
        X, F, _ = euclidean_problem(ctx)
        res = sc.minimize_optax(F, X.zeros(), optax.adam(1e-3), max_iter=3, tol=0.0, verbose=0)

        assert res.num_iters == 3
        assert res.status == 1 and not res.success
        assert res.message == "maximum iterations reached"

    def test_max_iter_zero_returns_initial_point(self):
        import optax

        ctx = _jax_ctx()
        X, F, _ = euclidean_problem(ctx)
        x0 = ctx.asarray([0.3, -0.7])
        res = sc.minimize_optax(F, x0, optax.adam(1e-1), max_iter=0, tol=1e-6, verbose=0)

        assert res.num_iters == 0
        assert res.nfev == res.njev == 1  # only the initial evaluation
        np.testing.assert_array_equal(to_numpy(res.x_element), to_numpy(x0))

    def test_nfev_njev_scale_with_iterations(self):
        import optax

        ctx = _jax_ctx()
        X, F, _ = euclidean_problem(ctx)
        res = sc.minimize_optax(F, X.zeros(), optax.sgd(1e-2), max_iter=17, tol=0.0, verbose=0)

        assert res.num_iters == 17
        assert res.n_linesearch_steps == 0
        assert res.nfev == 18 and res.njev == 18  # num_iters + 1

    def test_lbfgs_reports_linesearch_separately(self):
        import optax

        ctx = _jax_ctx()
        X, F, x_star = euclidean_problem(ctx)
        res = sc.minimize_optax(F, X.zeros(), optax.lbfgs(), max_iter=100, tol=1e-6, verbose=0)

        assert res.success
        assert res.n_linesearch_steps > 0  # lbfgs runs a line search
        # nfev/njev count only the driver's own value_and_grad calls; the optax
        # line-search evaluations are reported separately, not folded in.
        assert res.nfev == res.njev == res.num_iters + 1
        np.testing.assert_allclose(to_numpy(res.x_element), x_star, atol=1e-4)

    def test_nonfinite_diverges(self):
        import optax

        ctx = _jax_ctx()
        X, F, _ = weighted_problem(ctx)
        # A wildly oversized step overflows to inf within a few iterations.
        res = sc.minimize_optax(F, X.zeros(), optax.sgd(1e20), max_iter=1000, tol=1e-6, verbose=0)

        assert res.status == 2 and not res.success
        assert "nonfinite" in res.message


# ===========================================================================
# project hook
# ===========================================================================
class TestProject:
    def test_project_keeps_iterate_in_set(self):
        import optax

        ctx = _jax_ctx()
        X, F, _ = weighted_problem(ctx)
        # Retraction pins the third coordinate to zero after every update.
        project = lambda p: p.at[2].set(0.0)  # noqa: E731
        res = sc.minimize_optax(
            F, X.zeros(), optax.adam(5e-2), max_iter=200, tol=1e-8, project=project, verbose=0
        )

        assert float(to_numpy(res.x_element)[2]) == 0.0


# ===========================================================================
# History columns + callback replay
# ===========================================================================
class TestHistory:
    def test_four_columns_and_delta_is_objective_difference(self):
        import optax

        ctx = _jax_ctx()
        X, F, _ = euclidean_problem(ctx)
        # Record every iteration so value_delta[k] == value[k] - value[k-1].
        res = sc.minimize_optax(
            F, X.zeros(), optax.sgd(1e-1), max_iter=30, tol=0.0,
            history_every=1, log_every=1000, verbose=0,
        )
        h = res.history
        assert set(h) == {"iteration", "value", "value_delta", "grad_norm"}
        assert h["iteration"][0] == 0 and h["value_delta"][0] == 0.0
        # Column 3 is the per-iteration objective change F_k - F_{k-1}.
        np.testing.assert_allclose(h["value_delta"][1:], np.diff(h["value"]), atol=1e-6)
        assert h["value"][-1] < h["value"][0]  # objective decreased

    def test_progress_callback_replayed_from_history(self):
        import optax

        ctx = _jax_ctx()
        X, F, _ = euclidean_problem(ctx)
        rows = []
        res = sc.minimize_optax(
            F, X.zeros(), optax.adam(1e-1), max_iter=500, tol=1e-5,
            history_every=25, verbose=0, progress_callback=rows.append,
        )

        assert len(rows) == len(res.history["iteration"])
        assert rows and set(rows[0]) == {"iteration", "value", "value_delta", "grad_norm"}

    def test_record_history_false_empty(self):
        import optax

        ctx = _jax_ctx()
        X, F, _ = euclidean_problem(ctx)
        res = sc.minimize_optax(
            F, X.zeros(), optax.adam(1e-1), max_iter=50, tol=1e-5,
            record_history=False, verbose=0,
        )
        assert res.history == {}


# ===========================================================================
# One value_and_grad per iteration (traced once, not O(iterations))
# ===========================================================================
class TestSingleEvaluation:
    def test_value_and_grad_call_count_independent_of_max_iter(self):
        import optax

        ctx = _jax_ctx()
        X = sc.DenseCoordinateSpace((3,), ctx)

        counts = {}
        for max_iter in (5, 500):
            _CountingQuadratic.calls = 0
            sc.minimize_optax(
                _CountingQuadratic(X, ctx), X.zeros(), optax.sgd(1e-1),
                max_iter=max_iter, tol=1e-9, verbose=0,
            )
            counts[max_iter] = _CountingQuadratic.calls

        # The loop body is traced once, so the count does not grow with max_iter:
        # one initial evaluation + one traced body evaluation.
        assert counts[5] == counts[500]
        assert counts[5] <= 3


# ===========================================================================
# Gradient norm / finiteness (complex-correct)
# ===========================================================================
class TestGradientNorm:
    def test_l2_norm_is_complex_correct(self):
        import jax.numpy as jnp

        from spacecore.optimize._optax import _tree_l2_norm

        # Purely imaginary gradient must have a nonzero norm (regression: a
        # real()-before-square norm would report 0 here).
        g = jnp.array([0 + 1j, 0 + 2j])
        assert float(_tree_l2_norm(g)) == pytest.approx(np.sqrt(5.0), rel=1e-5)
        assert float(_tree_l2_norm(jnp.array([3 + 4j]))) == pytest.approx(5.0, rel=1e-5)

    def test_all_finite_checks_leaves_including_complex(self):
        import jax.numpy as jnp

        from spacecore.optimize._optax import _tree_all_finite

        assert bool(_tree_all_finite((jnp.array([1.0, 2.0]), jnp.array([3.0]))))
        assert not bool(_tree_all_finite((jnp.array([1.0, jnp.inf]),)))
        # A non-finite imaginary part is caught even though the real part is finite.
        assert not bool(_tree_all_finite((jnp.array([1 + 0j, complex(0, float("nan"))]),)))

    def test_linesearch_steps_sums_all_counters(self):
        import collections

        import jax.numpy as jnp

        from spacecore.optimize._optax import _linesearch_steps

        Info = collections.namedtuple("Info", ["num_linesearch_steps"])
        # Two separate line-search infos nested in a chain-like state are summed,
        # not just the first one found.
        state = (Info(jnp.asarray(3)), ("other", Info(jnp.asarray(4))))
        assert int(_linesearch_steps(state)) == 7
        # No counter present -> None (gradient-transformation optimizers).
        assert _linesearch_steps(("no", "counters")) is None


# ===========================================================================
# Input guards
# ===========================================================================
class TestGuards:
    def test_rejects_non_jax_domain(self):
        import optax

        ctx = make_ctx("numpy", np.float64)
        X, F, _ = euclidean_problem(ctx)
        with pytest.raises(TypeError, match="JAX-backed"):
            sc.minimize_optax(F, X.zeros(), optax.adam(1e-1), max_iter=10)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_iter": -1}, "max_iter"),
            ({"tol": -1.0}, "tol"),
            ({"log_every": 0}, "log_every"),
            ({"history_every": 0}, "history_every"),
        ],
    )
    def test_rejects_bad_parameters(self, kwargs, match):
        import optax

        ctx = _jax_ctx()
        X, F, _ = euclidean_problem(ctx)
        with pytest.raises(ValueError, match=match):
            sc.minimize_optax(F, X.zeros(), optax.adam(1e-1), **kwargs)
