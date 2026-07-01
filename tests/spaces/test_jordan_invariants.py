"""Jordan spectral primitives: ``trace`` / ``determinant`` / ``unit`` (0.4.2 W2).

Parametrized invariants across every Jordan-space family (elementwise, Hermitian,
Euclidean-elementwise, stacked, tree; real and complex) via the shared generator
registry, plus direct tests for batch-axis preservation, multi-dimensional
elementwise reduction, and the direct-sum (stacked) semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy
from tests.generators import jordan_space_cases
from tests.spaces._generated_helpers import assert_allclose, case_params, tolerances

CASES = jordan_space_cases()


def _numpy_ctx(dtype=np.float64):
    return sc.Context(sc.NumpyOps(), dtype=dtype, check_level="none")


# ===========================================================================
# Registry-wide invariants (every Jordan family)
# ===========================================================================
@pytest.mark.parametrize("case", case_params(CASES))
def test_trace_matches_spectrum_sum(case):
    space = case.obj
    x = case.reference["x"]
    rtol, atol = tolerances(space.dtype)
    # trace(x) == sum of the Jordan spectrum (additive over direct sums, since a
    # tree/stacked spectrum is the concatenation of leaf/copy spectra).
    np.testing.assert_allclose(
        to_numpy(space.trace(x)), np.sum(to_numpy(space.spectrum(x))), rtol=rtol, atol=atol
    )
    if case.reference["kind"] == "hermitian":
        # Independent oracle: the matrix trace of a Hermitian element is real.
        np.testing.assert_allclose(
            to_numpy(space.trace(x)), np.real(np.trace(to_numpy(x))), rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("case", case_params(CASES))
def test_determinant_matches_spectrum_prod(case):
    space = case.obj
    x = case.reference["x"]
    rtol, atol = tolerances(space.dtype)
    # determinant(x) == product of the Jordan spectrum (multiplicative over direct
    # sums, since np.prod over the concatenated spectrum multiplies leaf/copy dets).
    np.testing.assert_allclose(
        to_numpy(space.determinant(x)),
        np.prod(to_numpy(space.spectrum(x))),
        rtol=rtol,
        atol=atol,
    )
    if case.reference["kind"] == "hermitian":
        np.testing.assert_allclose(
            to_numpy(space.determinant(x)),
            np.real(np.linalg.det(to_numpy(x))),
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("case", case_params(CASES))
def test_unit_is_jordan_identity(case):
    space = case.obj
    x = case.reference["x"]
    # e ∘ x == x on every Jordan algebra.
    assert_allclose(space, space.jordan(space.unit(), x), x)


@pytest.mark.parametrize("case", case_params(CASES))
def test_trace_inner_oracle(case):
    space = case.obj
    if not isinstance(space, sc.EuclideanJordanAlgebraSpace):
        pytest.skip("trace(x) == <unit, x> holds only on Euclidean Jordan algebras")
    x = case.reference["x"]
    rtol, atol = tolerances(space.dtype)
    # Compare real parts: for complex Hermitian both sides are real-valued but the
    # inner product returns a complex-typed scalar.
    np.testing.assert_allclose(
        np.real(to_numpy(space.trace(x))),
        np.real(to_numpy(space.inner(space.unit(), x))),
        rtol=rtol,
        atol=atol,
    )


# ===========================================================================
# Batch-axis preservation (Batched EJS / HMS keep the leading batch axis)
# ===========================================================================
def test_hermitian_trace_determinant_preserve_batch_axis():
    ctx = _numpy_ctx()
    X = sc.HermitianSpace(3, ctx=ctx)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 3, 3))
    A = 0.5 * (A + np.swapaxes(A, -1, -2))  # symmetric batch
    arr = ctx.asarray(A)

    tr = to_numpy(X.trace(arr))
    assert tr.shape == (4,)
    np.testing.assert_allclose(tr, [np.trace(a) for a in A], rtol=1e-10, atol=1e-11)

    det = to_numpy(X.determinant(arr))
    assert det.shape == (4,)
    np.testing.assert_allclose(det, [np.linalg.det(a) for a in A], rtol=1e-6, atol=1e-8)


def test_elementwise_multidim_reduces_all_element_axes():
    ctx = _numpy_ctx()
    X = sc.ElementwiseJordanSpace((2, 3), ctx)  # matrix-shaped elementwise algebra
    rng = np.random.default_rng(1)
    A = rng.standard_normal((2, 3))
    arr = ctx.asarray(A)

    np.testing.assert_allclose(float(to_numpy(X.trace(arr))), A.sum(), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(
        float(to_numpy(X.determinant(arr))), A.prod(), rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(to_numpy(X.unit()), np.ones((2, 3)), rtol=1e-10, atol=1e-11)

    # Batched input keeps the leading batch axis: (B, 2, 3) -> (B,).
    B = rng.standard_normal((4, 2, 3))
    tr = to_numpy(X.trace(ctx.asarray(B)))
    assert tr.shape == (4,)
    np.testing.assert_allclose(tr, B.sum(axis=(-2, -1)), rtol=1e-10, atol=1e-11)


# ===========================================================================
# Direct-sum semantics: stacked = additive trace, multiplicative det, tiled unit
# ===========================================================================
def test_stacked_is_direct_sum_of_copies():
    ctx = _numpy_ctx()
    base = sc.HermitianSpace(2, ctx=ctx)
    X = sc.StackedSpace(base, 3, ctx)
    rng = np.random.default_rng(2)
    copies = []
    for _ in range(3):
        m = rng.standard_normal((2, 2))
        copies.append(0.5 * (m + m.T))
    arr = ctx.asarray(np.stack(copies))

    np.testing.assert_allclose(
        float(to_numpy(X.trace(arr))), sum(np.trace(m) for m in copies), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        float(to_numpy(X.determinant(arr))),
        float(np.prod([np.linalg.det(m) for m in copies])),
        rtol=1e-6,
        atol=1e-8,
    )
    # unit is I_2 replicated across all three copies.
    u = to_numpy(X.unit())
    assert u.shape == (3, 2, 2)
    for i in range(3):
        np.testing.assert_allclose(u[i], np.eye(2), rtol=1e-10, atol=1e-11)


def test_stacked_trace_determinant_preserve_batch_axis():
    # A batch of stacked elements (B, count, n, n) must reduce only the copy axis,
    # yielding (B,) — the stacked reduction must not collapse the leading batch axis.
    ctx = _numpy_ctx()
    X = sc.StackedSpace(sc.HermitianSpace(2, ctx=ctx), 3, ctx)
    rng = np.random.default_rng(4)
    A = rng.standard_normal((4, 3, 2, 2))
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    arr = ctx.asarray(A)

    tr = to_numpy(X.trace(arr))
    assert tr.shape == (4,)
    np.testing.assert_allclose(
        tr, [sum(np.trace(A[b, i]) for i in range(3)) for b in range(4)], rtol=1e-10, atol=1e-11
    )

    det = to_numpy(X.determinant(arr))
    assert det.shape == (4,)
    np.testing.assert_allclose(
        det,
        [float(np.prod([np.linalg.det(A[b, i]) for i in range(3)])) for b in range(4)],
        rtol=1e-6,
        atol=1e-8,
    )
