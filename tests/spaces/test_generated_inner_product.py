from __future__ import annotations

import numpy as np
import pytest

from spacecore._batching import _batched_inner
from tests._helpers import to_numpy
from tests.generators import MatrixInnerProduct, inner_product_space_cases
from tests.spaces._generated_helpers import batch_item, case_params, convert_element, tolerances


CASES = inner_product_space_cases()


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_inner_product_axioms_and_norm(case):
    space = case.obj
    ref = case.reference
    x, y, a = ref["x"], ref["y"], ref["a"]
    rtol, atol = tolerances(space.dtype)

    inner_xy = to_numpy(space.inner(x, y))
    assert np.shape(inner_xy) == ()
    np.testing.assert_allclose(inner_xy, np.conj(to_numpy(space.inner(y, x))), rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        to_numpy(space.inner(space.scale(a, x), y)),
        np.conj(to_numpy(a)) * inner_xy,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(space.inner(x, space.scale(a, y))),
        to_numpy(a) * inner_xy,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(space.norm(x)) ** 2,
        np.real(to_numpy(space.inner(x, x))),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(space.riesz_inverse(space.riesz(x))),
        to_numpy(x),
        rtol=rtol,
        atol=atol,
    )
    if ref["geometry"] == "spd":
        metric = ref["metric"]
        assert np.all(np.linalg.eigvalsh(metric["matrix"]) > 0)
        assert metric["condition_estimate"] == pytest.approx(8.0, rel=2e-5)


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_inner_product_conversion_preserves_geometry(case):
    space = case.obj
    ref = case.reference
    target = space.convert(ref["target_ctx"])
    x = convert_element(space, ref["x"], target)
    y = convert_element(space, ref["y"], target)
    source_tolerance = tolerances(space.dtype)
    target_tolerance = tolerances(target.dtype)
    rtol = max(source_tolerance[0], target_tolerance[0])
    atol = max(source_tolerance[1], target_tolerance[1])

    assert target.is_euclidean == space.is_euclidean
    if ref["geometry"] == "spd":
        assert isinstance(target.geometry, MatrixInnerProduct)
    np.testing.assert_allclose(
        to_numpy(target.inner(x, y)),
        to_numpy(space.inner(ref["x"], ref["y"])),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_batched_inner_product_matches_loop(case):
    space = case.obj
    ref = case.reference
    actual = _batched_inner(space, ref["batch_x"], ref["batch_y"])
    expected = np.asarray(
        [
            to_numpy(
                space.inner(
                    batch_item(space, ref["batch_x"], index),
                    batch_item(space, ref["batch_y"], index),
                )
            )
            for index in range(2)
        ]
    )
    rtol, atol = tolerances(space.dtype)

    np.testing.assert_allclose(to_numpy(actual), expected, rtol=rtol, atol=atol)
