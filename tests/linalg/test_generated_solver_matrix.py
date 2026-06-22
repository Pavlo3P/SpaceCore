"""Solver x ``check_level`` coverage matrix for every iterative solver.

Checklist section 10, Cross-cutting integration:

* Each public iterative solver (:func:`spacecore.cg`, :func:`spacecore.lsqr`,
  :func:`spacecore.lanczos_smallest`, :func:`spacecore.power_iteration`) runs
  and returns a sane, converged result at every supported
  :data:`spacecore.CHECK_LEVELS` value (``none`` / ``cheap`` / ``standard`` /
  ``strict``).
* ``cg`` / ``power_iteration`` / ``lanczos_smallest`` are exercised against a
  genuinely SPD / Hermitian operator so the extra Hermitian and
  positive-curvature probes enabled at ``check_level='strict'`` pass.
* ``lsqr`` is exercised against a small rectangular operator (its natural
  least-squares shape) across the same check levels.
* Where it is cheap, results are compared to a direct NumPy reference
  (``solve`` / ``lstsq`` / ``eigvalsh``).
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc

from tests._helpers import to_numpy

# A genuinely SPD (real symmetric, positive eigenvalues) operator matrix.
# Used by cg / power_iteration / lanczos_smallest so that the strict-level
# Hermitian and positive-curvature probes pass.
_SPD = np.array(
    [[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]],
    dtype=np.float64,
)
# A small rectangular operator (4x3) for lsqr.
_RECT = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    dtype=np.float64,
)


def _ctx(check_level: str):
    return sc.Context(sc.NumpyOps(), dtype=np.float64, check_level=check_level)


# ===========================================================================
# Conjugate gradients across every check level
# ===========================================================================
class TestCGCheckLevels:
    @pytest.mark.parametrize("check_level", sc.CHECK_LEVELS)
    def test_cg_solves_spd_at_each_check_level(self, check_level):
        ctx = _ctx(check_level)
        space = sc.DenseCoordinateSpace((3,), ctx)
        operator = sc.DenseLinOp(ctx.asarray(_SPD), space, space, ctx)
        b = ctx.asarray([1.0, -2.0, 3.0])

        result = sc.cg(operator, b, tol=1e-12, maxiter=20, check_every=1)

        expected = np.linalg.solve(_SPD, np.array([1.0, -2.0, 3.0]))
        assert bool(to_numpy(result.converged))
        np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            to_numpy(operator.apply(result.x) - b), 0.0, atol=1e-9
        )


# ===========================================================================
# LSQR across every check level
# ===========================================================================
class TestLSQRCheckLevels:
    @pytest.mark.parametrize("check_level", sc.CHECK_LEVELS)
    def test_lsqr_solves_rectangular_at_each_check_level(self, check_level):
        ctx = _ctx(check_level)
        domain = sc.DenseCoordinateSpace((3,), ctx)
        codomain = sc.DenseCoordinateSpace((4,), ctx)
        operator = sc.DenseLinOp(ctx.asarray(_RECT), domain, codomain, ctx)
        b = ctx.asarray([1.0, 2.0, 3.0, 6.0])

        result = sc.lsqr(operator, b, tol=1e-12, maxiter=50, check_every=1)

        expected, *_ = np.linalg.lstsq(
            _RECT, np.array([1.0, 2.0, 3.0, 6.0]), rcond=None
        )
        assert bool(to_numpy(result.converged))
        np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(
            to_numpy(result.normal_residual_norm), 0.0, atol=1e-7
        )


# ===========================================================================
# Lanczos (smallest eigenvalue) across every check level
# ===========================================================================
class TestLanczosCheckLevels:
    @pytest.mark.parametrize("check_level", sc.CHECK_LEVELS)
    def test_lanczos_smallest_eigenvalue_at_each_check_level(self, check_level):
        ctx = _ctx(check_level)
        space = sc.DenseCoordinateSpace((3,), ctx)
        operator = sc.DenseLinOp(ctx.asarray(_SPD), space, space, ctx)
        initial_vector = ctx.asarray([1.0, 1.0, 1.0])

        result = sc.lanczos_smallest(
            operator, initial_vector, max_iter=20, tol=1e-10, check_every=1
        )

        expected = float(np.linalg.eigvalsh(_SPD).min())
        assert bool(to_numpy(result.converged))
        np.testing.assert_allclose(
            float(to_numpy(result.eigenvalue)), expected, rtol=1e-8, atol=1e-8
        )


# ===========================================================================
# Power iteration (largest eigenvalue) across every check level
# ===========================================================================
class TestPowerIterationCheckLevels:
    @pytest.mark.parametrize("check_level", sc.CHECK_LEVELS)
    def test_power_iteration_largest_eigenvalue_at_each_check_level(self, check_level):
        ctx = _ctx(check_level)
        space = sc.DenseCoordinateSpace((3,), ctx)
        operator = sc.DenseLinOp(ctx.asarray(_SPD), space, space, ctx)
        x0 = ctx.asarray([1.0, 1.0, 1.0])

        result = sc.power_iteration(operator, x0=x0, tol=1e-10, maxiter=500)

        expected = float(np.linalg.eigvalsh(_SPD).max())
        assert bool(to_numpy(result.converged))
        np.testing.assert_allclose(
            float(to_numpy(result.eigenvalue)), expected, rtol=1e-6, atol=1e-6
        )
