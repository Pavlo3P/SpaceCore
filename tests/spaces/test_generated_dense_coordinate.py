from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from spacecore._batching import _check_batched
from tests.generators import dense_coordinate_space_cases
from tests.spaces._generated_helpers import (
    assert_allclose,
    assert_batch_allclose,
    batch_item,
    case_params,
    convert_element,
)


CASES = dense_coordinate_space_cases()


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_dense_coordinate_operations_and_flattening(case):
    space = case.obj
    ref = case.reference

    assert space.ctx == ref["ctx"]
    assert space.dtype == ref["dtype"]
    assert space.field == ref["field"]
    assert space.check_level == ref["check_level"]
    assert_allclose(space, space.zeros(), space.ctx.asarray(np.zeros(space.shape)))
    assert_allclose(space, space.add(ref["x"], ref["y"]), ref["x"] + ref["y"])
    assert_allclose(space, space.scale(ref["a"], ref["x"]), ref["a"] * ref["x"])

    flattened = space.flatten(ref["x"])
    assert flattened.shape == (space.size,)
    assert_allclose(space, space.unflatten(flattened), ref["x"])


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_dense_coordinate_batching_matches_pointwise(case):
    space = case.obj
    ref = case.reference

    _check_batched(space, ref["batch_x"])
    added = space.add_batch(ref["batch_x"], ref["batch_y"])
    scaled = space.scale_batch(ref["a"], ref["batch_x"])
    zero_batch = space.ops.stack((space.zeros(), space.zeros()), axis=0)
    assert_batch_allclose(
        space,
        space.add_batch(ref["batch_x"], zero_batch),
        ref["batch_x"],
    )
    for index in range(ref["batch_shape"][0]):
        assert_allclose(
            space,
            batch_item(space, added, index),
            space.add(batch_item(space, ref["batch_x"], index), batch_item(space, ref["batch_y"], index)),
        )
        assert_allclose(
            space,
            batch_item(space, scaled, index),
            space.scale(ref["a"], batch_item(space, ref["batch_x"], index)),
        )

    flat = space.flatten_batch(ref["batch_x"])
    assert flat.shape == (2, space.size)
    assert_batch_allclose(space, space.unflatten_batch(flat), ref["batch_x"])


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_dense_coordinate_membership_by_check_level(case):
    space = case.obj
    ref = case.reference

    space.check_member(ref["x"])
    invalid_values = (
        ref["invalid_shape"],
        ref["invalid_dtype"],
        ref["invalid_field"],
        ref["invalid_backend"],
    )
    if space.check_level == "none":
        for value in invalid_values:
            space.check_member(value)
        _check_batched(space, ref["invalid_batch"])
    else:
        for value in invalid_values:
            with pytest.raises(sc.SpaceValidationError):
                space.check_member(value)
        if space.shape:
            with pytest.raises(sc.SpaceValidationError, match="trailing shape"):
                _check_batched(space, ref["invalid_batch"])


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_dense_coordinate_conversion_preserves_structure(case):
    space = case.obj
    ref = case.reference
    converted = space.convert(ref["target_ctx"])
    value = convert_element(space, ref["x"], converted)

    assert type(converted) is type(space)
    assert converted.shape == space.shape
    assert converted.field == space.field
    assert converted.dtype == ref["target_ctx"].dtype
    converted.check_member(value)
    np.testing.assert_allclose(
        converted.flatten(value),
        np.asarray(ref["x_numpy"]).reshape((-1,)),
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "dtype, field",
    [(np.float32, "real"), (np.float64, "real"), (np.complex64, "complex"), (np.complex128, "complex")],
)
def test_generated_dense_coordinate_field_and_exact_dtype_are_distinct(dtype, field):
    ctx = sc.Context(sc.NumpyOps(), dtype=dtype, check_level="cheap")
    space = sc.DenseCoordinateSpace((2,), ctx)

    assert space.field == field
    if field == "real":
        with pytest.raises(sc.SpaceValidationError, match="real scalar field"):
            space.check_member(np.asarray([1.0 + 1.0j, 2.0], dtype=np.complex64))
    else:
        raw_real = np.asarray([1.0, 2.0], dtype=ctx.ops.real_dtype(ctx.dtype))
        assert sc.FieldCheck().is_valid(space, raw_real)
        assert not sc.DTypeCheck().is_valid(space, raw_real)
        converted_real = ctx.asarray(raw_real)
        space.check_member(converted_real)
