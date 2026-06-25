from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy
from tests.generators import (
    backend_linop_cases,
    context_params,
    linop_cases,
    sparse_linop_case,
)


STANDARD_CASES = linop_cases()
CHECK_LEVEL_CASES = linop_cases(
    dtypes=(np.float64,),
    check_levels=sc.CHECK_LEVELS,
    include_weighted=False,
)


def _params(cases):
    return tuple(pytest.param(case, marks=case.marks, id=case.id) for case in cases)


def _assert_allclose(actual: Any, expected: Any, *, rtol: float = 1e-6) -> None:
    actual = to_numpy(actual)
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_leaf, expected_leaf in zip(actual, expected):
            _assert_allclose(actual_leaf, expected_leaf, rtol=rtol)
        return
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol)


def _convert_value(value: Any, ctx: sc.Context) -> Any:
    if isinstance(value, tuple):
        return tuple(_convert_value(leaf, ctx) for leaf in value)
    return ctx.asarray(value)


@pytest.mark.parametrize("case", _params(STANDARD_CASES))
def test_generated_apply_and_rapply_match_references(case):
    op = case.obj
    reference = case.reference

    _assert_allclose(op.apply(reference["x"]), reference["expected_apply"])
    _assert_allclose(op.rapply(reference["y"]), reference["expected_rapply"])


@pytest.mark.parametrize("case", _params(STANDARD_CASES))
def test_generated_metric_adjoint_and_structural_adjoint_laws(case):
    op = case.obj
    x = case.reference["x"]
    y = case.reference["y"]

    lhs = op.codomain.inner(op.apply(x), y)
    rhs = op.domain.inner(x, op.rapply(y))
    _assert_allclose(lhs, rhs)
    _assert_allclose(op.H.apply(y), op.rapply(y))
    _assert_allclose(op.H.H.apply(x), op.apply(x))
    assert op.H.H is op


@pytest.mark.parametrize("case", _params(STANDARD_CASES))
def test_generated_materialization_and_conversion_preserve_behavior(case):
    op = case.obj
    reference = case.reference
    matrix = reference["reference_matrix"]
    if matrix is not None:
        _assert_allclose(op.to_matrix(), matrix)

    target_ctx = reference["target_ctx"]
    if not reference["supports_conversion"] or target_ctx is None:
        pytest.skip("this generated case does not claim conversion coverage")
    converted = op.convert(target_ctx)
    converted_x = _convert_value(reference["x"], target_ctx)
    converted_y = _convert_value(reference["y"], target_ctx)
    _assert_allclose(converted.apply(converted_x), reference["expected_apply"], rtol=2e-5)
    _assert_allclose(converted.rapply(converted_y), reference["expected_rapply"], rtol=2e-5)


@pytest.mark.parametrize("case", _params(STANDARD_CASES))
def test_generated_algebraic_identity_laws(case):
    op = case.obj
    x = case.reference["x"]
    expected = case.reference["expected_apply"]

    _assert_allclose((sc.IdentityLinOp(op.codomain) @ op).apply(x), expected)
    _assert_allclose((op @ sc.IdentityLinOp(op.domain)).apply(x), expected)
    _assert_allclose((op + sc.ZeroLinOp(op.domain, op.codomain)).apply(x), expected)
    _assert_allclose((1 * op).apply(x), expected)
    _assert_allclose((op + op).apply(x), op.codomain.scale(2, op.apply(x)))
    _assert_allclose((2 * op).apply(x), op.codomain.scale(2, op.apply(x)))


@pytest.mark.parametrize("case", _params(CHECK_LEVEL_CASES))
def test_generated_batching_matches_references_at_every_check_level(case):
    reference = case.reference
    assert reference["supports_batching"]
    _assert_allclose(case.obj.apply(reference["x"]), reference["expected_apply"])
    _assert_allclose(case.obj.rapply(reference["y"]), reference["expected_rapply"])
    _assert_allclose(case.obj.vapply(reference["batch_x"]), reference["expected_vapply"])
    _assert_allclose(case.obj.rvapply(reference["batch_y"]), reference["expected_rvapply"])


@pytest.mark.parametrize("ctx", context_params())
def test_generated_portable_families_run_on_supported_backends(ctx):
    for case in backend_linop_cases(ctx):
        reference = case.reference
        _assert_allclose(case.obj.apply(reference["x"]), reference["expected_apply"], rtol=2e-5)
        _assert_allclose(case.obj.rapply(reference["y"]), reference["expected_rapply"], rtol=2e-5)
        _assert_allclose(case.obj.vapply(reference["batch_x"]), reference["expected_vapply"], rtol=2e-5)
        _assert_allclose(case.obj.rvapply(reference["batch_y"]), reference["expected_rvapply"], rtol=2e-5)


@pytest.mark.parametrize("ctx", context_params())
def test_generated_sparse_family_skips_when_backend_sparse_is_unavailable(ctx):
    try:
        case = sparse_linop_case(ctx)
    except (AttributeError, ImportError, NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
        pytest.skip(f"{ctx.ops.family} sparse arrays are unavailable: {exc}")
    reference = case.reference
    _assert_allclose(case.obj.apply(reference["x"]), reference["expected_apply"], rtol=2e-5)
    _assert_allclose(case.obj.rapply(reference["y"]), reference["expected_rapply"], rtol=2e-5)


def test_generated_inventory_covers_every_concrete_public_family():
    covered = {type(case.obj) for case in STANDARD_CASES}
    assert covered == {
        sc.DenseLinOp,
        sc.SparseLinOp,
        sc.DiagonalLinOp,
        sc.IdentityLinOp,
        sc.ZeroLinOp,
        sc.MatrixFreeLinOp,
        sc.ComposedLinOp,
        sc.SumLinOp,
        sc.ScaledLinOp,
        sc.StackedLinOp,
        sc.SumToSingleLinOp,
        sc.BlockDiagonalLinOp,
        sc.BlockMatrixLinOp,
    }
