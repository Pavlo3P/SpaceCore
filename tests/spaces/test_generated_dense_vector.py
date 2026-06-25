from __future__ import annotations

import pytest

import spacecore as sc
from tests.generators import dense_vector_space_cases
from tests.spaces._generated_helpers import (
    assert_allclose,
    assert_batch_allclose,
    case_params,
    convert_element,
)


CASES = dense_vector_space_cases()


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_dense_vector_shape_operations_and_field(case):
    space = case.obj
    ref = case.reference

    assert isinstance(space, sc.DenseVectorSpace)
    assert len(space.shape) == 1
    assert space.field == ref["field"]
    assert space.dtype == ref["dtype"]
    assert_allclose(space, space.add(ref["x"], ref["y"]), ref["x"] + ref["y"])
    assert_allclose(space, space.scale(ref["a"], ref["x"]), ref["a"] * ref["x"])
    assert_allclose(space, space.star(space.star(ref["x"])), ref["x"])
    assert_batch_allclose(
        space,
        space.add_batch(ref["batch_x"], ref["batch_y"]),
        ref["batch_x"] + ref["batch_y"],
    )


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_dense_vector_conversion_and_batch_roundtrip(case):
    space = case.obj
    ref = case.reference
    converted = space.convert(ref["target_ctx"])
    value = convert_element(space, ref["x"], converted)

    assert isinstance(converted, sc.DenseVectorSpace)
    assert converted.shape == space.shape
    assert converted.field == space.field
    converted.check_member(value)
    assert_batch_allclose(
        space,
        space.unflatten_batch(space.flatten_batch(ref["batch_x"])),
        ref["batch_x"],
    )


def test_dense_vector_rejects_non_vector_shapes():
    ctx = sc.Context(sc.NumpyOps())

    for shape in ((), (2, 2), (1, 2, 3)):
        with pytest.raises(ValueError, match="one-dimensional shape"):
            sc.DenseVectorSpace(shape, ctx)
