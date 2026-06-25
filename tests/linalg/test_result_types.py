"""Tests for the linalg result ``NamedTuple`` types.

Checklist section 8, Result types:

* ``CGResult``, ``LSQRResult``, ``LanczosResult``, ``PowerIterationResult`` and
  ``ExpmMultiplyResult`` expose their documented fields by name and position.
* ``__repr__`` is compact: it names the type and its scalar fields but
  summarizes array fields as ``<array shape=...>`` rather than dumping the full
  vector / solution.
"""
from __future__ import annotations

import numpy as np

import spacecore as sc


# ===========================================================================
# CGResult
# ===========================================================================
class TestCGResult:
    def test_fields_by_name_and_position(self):
        x = np.array([1.0, 2.0, 3.0])
        r = sc.CGResult(x, True, 4, 1e-7)
        assert r.x is x
        assert r.converged is True
        assert r.num_iters == 4
        assert r.residual_norm == 1e-7
        assert tuple(r) == (x, True, 4, 1e-7)

    def test_repr_is_compact(self):
        r = sc.CGResult(np.zeros(100), True, 4, 1e-7)
        text = repr(r)
        assert text.startswith("CGResult(")
        for field in ("converged", "num_iters", "residual_norm", "x"):
            assert field in text
        assert "x=<array shape=(100,)" in text
        assert "0.0, 0.0" not in text  # full array not dumped


# ===========================================================================
# LSQRResult
# ===========================================================================
class TestLSQRResult:
    def test_fields(self):
        x = np.array([1.0, 2.0])
        r = sc.LSQRResult(x, False, 9, 0.5, 0.25)
        assert r.x is x
        assert r.converged is False
        assert r.num_iters == 9
        assert r.residual_norm == 0.5
        assert r.normal_residual_norm == 0.25

    def test_repr_lists_both_residual_norms(self):
        text = repr(sc.LSQRResult(np.zeros(50), True, 3, 0.5, 0.25))
        assert text.startswith("LSQRResult(")
        assert "residual_norm" in text and "normal_residual_norm" in text
        assert "x=<array shape=(50,)" in text


# ===========================================================================
# LanczosResult
# ===========================================================================
class TestLanczosResult:
    def test_fields(self):
        vec = np.array([1.0, 0.0])
        r = sc.LanczosResult(2.0, vec, 1e-9, 2, True)
        assert r.eigenvalue == 2.0
        assert r.eigenvector is vec
        assert r.residual_norm == 1e-9
        assert r.krylov_dim == 2
        assert r.converged is True

    def test_repr_summarizes_eigenvector(self):
        text = repr(sc.LanczosResult(2.0, np.zeros(64), 1e-9, 2, True))
        assert text.startswith("LanczosResult(")
        assert "eigenvalue" in text and "krylov_dim" in text
        assert "eigenvector=<array shape=(64,)" in text


# ===========================================================================
# PowerIterationResult
# ===========================================================================
class TestPowerIterationResult:
    def test_fields(self):
        vec = np.array([0.0, 1.0])
        r = sc.PowerIterationResult(5.0, vec, True, 12, 1e-8)
        assert r.eigenvalue == 5.0
        assert r.eigenvector is vec
        assert r.converged is True
        assert r.num_iters == 12
        assert r.residual_norm == 1e-8

    def test_repr_summarizes_eigenvector(self):
        text = repr(sc.PowerIterationResult(5.0, np.zeros(32), True, 12, 1e-8))
        assert text.startswith("PowerIterationResult(")
        assert "eigenvalue" in text and "num_iters" in text
        assert "eigenvector=<array shape=(32,)" in text


# ===========================================================================
# ExpmMultiplyResult
# ===========================================================================
class TestExpmMultiplyResult:
    def test_fields(self):
        vec = np.array([1.0, 2.0])
        r = sc.ExpmMultiplyResult(vec, 3, 1e-11, True)
        assert r.result is vec
        assert r.krylov_dim == 3
        assert r.residual_estimate == 1e-11
        assert r.converged is True

    def test_repr_summarizes_result_vector(self):
        text = repr(sc.ExpmMultiplyResult(np.zeros(16), 3, 1e-11, True))
        assert text.startswith("ExpmMultiplyResult(")
        assert "krylov_dim" in text and "residual_estimate" in text
        assert "result=<array shape=(16,)" in text
