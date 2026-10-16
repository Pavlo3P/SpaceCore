"""Tests for :class:`spacecore.LinalgResult`, the one result type of every linalg solver.

* Fields are reachable by name and position; solver-specific fields default to ``None``.
* ``__repr__`` is compact: it lists only the fields that are set and summarizes array
  fields as ``<array shape=...>`` rather than dumping them.
* Every solver returns it.
"""
from __future__ import annotations

import numpy as np

import spacecore as sc


def test_fields_by_name_and_position():
    x = np.array([1.0, 2.0, 3.0])
    r = sc.LinalgResult(x, True, 4, 1e-7)
    assert r.x is x
    assert (r.converged, r.num_iters, r.residual_norm) == (True, 4, 1e-7)
    assert r.eigenvalue is None and r.normal_residual_norm is None
    assert tuple(r) == (x, True, 4, 1e-7, None, None)


def test_repr_lists_only_set_fields_and_summarizes_arrays():
    text = repr(sc.LinalgResult(np.zeros(100), True, 4, 1e-7, eigenvalue=2.0))
    assert text.startswith("LinalgResult(")
    for field in ("x", "converged", "num_iters", "residual_norm", "eigenvalue"):
        assert field in text
    assert "normal_residual_norm" not in text
    assert "x=<array shape=(100,)" in text
    assert "0.0, 0.0" not in text  # full array not dumped


def test_every_solver_returns_linalg_result():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    V = sc.DenseVectorSpace((3,), ctx=ctx)
    A = sc.DenseLinOp(ctx.asarray(np.diag([1.0, 2.0, 3.0])), V, V, ctx)
    b = ctx.asarray([1.0, 1.0, 1.0])
    results = {
        "cg": sc.cg(A, b),
        "lsqr": sc.lsqr(A, b),
        "power_iteration": sc.power_iteration(A),
        "lanczos_smallest": sc.lanczos_smallest(A, b, max_iter=3),
        "expm_multiply": sc.expm_multiply(A, b, max_iter=3),
    }
    for name, r in results.items():
        assert type(r) is sc.LinalgResult, name
    assert results["lsqr"].normal_residual_norm is not None
    assert results["power_iteration"].eigenvalue is not None
    assert results["lanczos_smallest"].eigenvalue is not None
    assert results["cg"].eigenvalue is None and results["expm_multiply"].eigenvalue is None
