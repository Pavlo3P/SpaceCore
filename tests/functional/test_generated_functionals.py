from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy
from tests.generators import check_level_params, functional_cases


CASES = functional_cases()


def _assert_element_allclose(space, actual, expected, *, rtol=1e-12, atol=1e-12):
    np.testing.assert_allclose(
        to_numpy(space.flatten(actual)),
        to_numpy(space.flatten(expected)),
        rtol=rtol,
        atol=atol,
    )


def _tolerances(case_or_functional):
    functional = getattr(case_or_functional, "obj", case_or_functional)
    dtype = np.dtype(functional.dtype)
    if dtype in (np.dtype(np.float32), np.dtype(np.complex64)):
        return 2e-5, 2e-5
    return 1e-12, 1e-12


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_generated_functional_values_match_direct_references(case):
    rtol, atol = _tolerances(case)

    np.testing.assert_allclose(
        to_numpy(case.obj.value(case.reference["x"])),
        case.reference["value"],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(case.obj(case.reference["x"])),
        case.reference["value"],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if "gradient" in case.capabilities],
    ids=lambda case: case.id,
)
def test_generated_functional_gradients_match_analytic_references(case):
    rtol, atol = _tolerances(case)
    actual = case.obj.grad(case.reference["x"])
    _assert_element_allclose(
        case.reference["domain"],
        actual,
        case.reference["gradient"],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in CASES
        if "gradient" in case.capabilities
        and "real" in case.capabilities
        and case.reference["kind"] != "zero"
    ],
    ids=lambda case: case.id,
)
def test_real_generated_gradients_satisfy_directional_derivative_identity(case):
    functional = case.obj
    domain = functional.domain
    x = case.reference["x"]
    direction = domain.unflatten(functional.ctx.asarray([0.25, -0.5, 0.75]))
    eps = 1e-6
    plus = domain.axpy(eps, direction, x)
    minus = domain.axpy(-eps, direction, x)
    finite_difference = (functional.value(plus) - functional.value(minus)) / (2.0 * eps)
    metric_derivative = domain.inner(functional.grad(x), direction)

    np.testing.assert_allclose(
        to_numpy(finite_difference), to_numpy(metric_derivative), rtol=2e-8, atol=2e-8
    )


@pytest.mark.parametrize(
    "case",
    [
        case
        for case in CASES
        if "weighted" in case.capabilities
        and case.reference["kind"] in {"linear", "quadratic"}
    ],
    ids=lambda case: case.id,
)
def test_weighted_functional_gradient_is_metric_not_coordinate_gradient(case):
    gradient = to_numpy(case.obj.domain.flatten(case.reference["gradient"]))
    coordinate_gradient = to_numpy(
        case.obj.domain.flatten(case.reference["coordinate_gradient"])
    )

    assert not np.allclose(gradient, coordinate_gradient)
    _assert_element_allclose(
        case.obj.domain,
        case.obj.grad(case.reference["x"]),
        case.reference["gradient"],
    )


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if "pullback" in case.reference],
    ids=lambda case: case.id,
)
def test_generated_pullbacks_match_value_and_gradient_identities(case):
    pullback = case.reference["pullback"]
    operator = case.reference["operator"]
    x = case.reference["pullback_x"]
    rtol, atol = _tolerances(case)

    np.testing.assert_allclose(
        to_numpy(pullback.value(x)),
        to_numpy(case.obj.value(operator.apply(x))),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(pullback.value(x)),
        case.reference["pullback_value"],
        rtol=rtol,
        atol=atol,
    )
    _assert_element_allclose(
        pullback.domain,
        pullback.grad(x),
        case.reference["pullback_gradient"],
        rtol=rtol,
        atol=atol,
    )

    target_ctx = case.reference["target_ctx"]
    converted = pullback.convert(target_ctx)
    converted_x = converted.domain.unflatten(
        target_ctx.asarray(pullback.domain.flatten(x))
    )
    converted_rtol, converted_atol = _tolerances(converted)
    np.testing.assert_allclose(
        to_numpy(converted.value(converted_x)),
        case.reference["pullback_value"],
        rtol=converted_rtol,
        atol=converted_atol,
    )


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case.reference["kind"] == "composed"],
    ids=lambda case: case.id,
)
def test_generic_composed_functional_is_a_value_pullback(case):
    x = case.reference["x"]
    source = case.reference["source_functional"]
    operator = case.reference["operator"]

    np.testing.assert_allclose(
        to_numpy(case.obj.value(x)),
        to_numpy(source.value(operator.apply(x))),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if "conversion" in case.capabilities],
    ids=lambda case: case.id,
)
def test_generated_functional_conversion_preserves_reference_behavior(case):
    target_ctx = case.reference["target_ctx"]
    converted = case.obj.convert(target_ctx)
    source_flat = case.obj.domain.flatten(case.reference["x"])
    converted_x = converted.domain.unflatten(target_ctx.asarray(source_flat))
    rtol, atol = _tolerances(converted)

    np.testing.assert_allclose(
        to_numpy(converted.value(converted_x)),
        case.reference["value"],
        rtol=rtol,
        atol=atol,
    )
    assert converted.ctx == target_ctx
    assert converted.domain.dtype == target_ctx.dtype


def test_functional_generator_records_check_level_and_required_reference_fields():
    cases = functional_cases(dtypes=(np.float64,), check_levels=sc.CHECK_LEVELS)

    assert {case.reference["check_level"] for case in cases} == set(sc.CHECK_LEVELS)
    for case in cases:
        assert case.reference["domain"] == case.obj.domain
        assert "x" in case.reference
        assert "value" in case.reference
        assert "gradient" in case.reference


def test_none_skips_optional_functional_input_membership_checks():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
    domain = sc.DenseCoordinateSpace((2,), ctx)
    functional = sc.MatrixFreeLinearFunctional(lambda _x: ctx.asarray(3.0), domain, ctx)

    np.testing.assert_allclose(functional.value(ctx.asarray([1.0, 2.0, 3.0])), 3.0)


@pytest.mark.parametrize("check_level", check_level_params(("cheap", "standard", "strict")))
def test_checked_functional_levels_reject_wrong_input_shape(check_level):
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level=check_level)
    domain = sc.DenseCoordinateSpace((2,), ctx)
    functional = sc.MatrixFreeLinearFunctional(lambda _x: ctx.asarray(3.0), domain, ctx)

    with pytest.raises(TypeError, match="Expected shape"):
        functional.value(ctx.asarray([1.0, 2.0, 3.0]))


def test_cheap_functional_checks_reject_field_and_dtype_mismatch():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="cheap")
    domain = sc.DenseCoordinateSpace((2,), ctx)
    functional = sc.InnerProductFunctional(ctx.asarray([1.0, 2.0]), domain, ctx)

    with pytest.raises(TypeError, match="real scalar field"):
        functional.value(np.asarray([1.0 + 1.0j, 2.0], dtype=np.complex128))
    with pytest.raises(TypeError, match="Expected dtype"):
        functional.value(np.asarray([1.0, 2.0], dtype=np.float32))


@pytest.mark.parametrize("check_level", check_level_params(("standard", "strict")))
def test_standard_functional_checks_reject_nonscalar_output(check_level):
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level=check_level)
    domain = sc.DenseCoordinateSpace((2,), ctx)
    functional = sc.MatrixFreeLinearFunctional(lambda x: x, domain, ctx)

    with pytest.raises(ValueError, match="Expected scalar batch output"):
        functional.value(ctx.asarray([1.0, 2.0]))


@pytest.mark.parametrize("check_level", check_level_params())
def test_pullback_domain_codomain_mismatch_is_always_rejected(check_level):
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level=check_level)
    domain = sc.DenseCoordinateSpace((2,), ctx)
    wrong_codomain = sc.DenseCoordinateSpace((3,), ctx)
    functional = sc.InnerProductFunctional(ctx.asarray([1.0, 2.0]), domain, ctx)
    operator = sc.DenseLinOp(
        ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        domain,
        wrong_codomain,
        ctx,
    )

    with pytest.raises(ValueError, match="A.codomain == F.domain"):
        functional.compose(operator)
