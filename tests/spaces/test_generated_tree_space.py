from __future__ import annotations

import pytest

import spacecore as sc
from tests.generators import tree_space_generated_cases
from tests.spaces._generated_helpers import (
    assert_allclose,
    assert_batch_allclose,
    batch_item,
    case_params,
)


CASES = tree_space_generated_cases()


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_tree_structure_arithmetic_and_paths(case):
    space = case.obj
    ref = case.reference

    space.check(ref["x"])
    assert space.leaf_paths == ref["leaf_paths"]
    assert_allclose(space, space.zeros(), space.zero())
    assert_allclose(space, space.add(ref["x"], ref["y"]), space.unflatten(space.flatten(ref["x"]) + space.flatten(ref["y"])))
    assert_allclose(space, space.scale(ref["a"], ref["x"]), space.unflatten(ref["a"] * space.flatten(ref["x"])))

    with pytest.raises(sc.SpaceValidationError, match="structure mismatch"):
        space.check_member(ref["mismatch"])
    with pytest.raises(sc.SpaceValidationError, match=r"Invalid leaf at \$"):
        space.check_member(ref["invalid_leaf"])
    with pytest.raises(ValueError, match="expected.*leaves"):
        sc.TreeElement(space, space.flatten_tree(ref["x"])[:-1])


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_tree_conversion_preserves_structure_and_capabilities(case):
    space = case.obj
    ref = case.reference
    converted = space.convert(ref["target_ctx"])
    value = space.convert_element(ref["x"], ref["target_ctx"])

    assert converted.treedef == space.treedef
    assert converted.leaf_paths == space.leaf_paths
    assert converted.field == space.field
    converted.check_member(value)
    if ref["profile"] == "inner":
        assert isinstance(space, sc.InnerProductSpace)
        assert isinstance(converted, sc.InnerProductSpace)
    else:
        assert not isinstance(space, sc.InnerProductSpace)
        assert not isinstance(converted, sc.InnerProductSpace)


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_tree_batching_preserves_structure_and_matches_pointwise(case):
    space = case.obj
    ref = case.reference
    added = space.add_batch(ref["batch_x"], ref["batch_y"])
    scaled = space.scale_batch(ref["a"], ref["batch_x"])
    zero_batch = space.stacked(2).zeros()

    assert len(space.flatten_tree(added)) == space.arity
    assert_batch_allclose(
        space,
        space.add_batch(ref["batch_x"], zero_batch),
        ref["batch_x"],
    )
    for index in range(2):
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
    assert_batch_allclose(
        space,
        space.unflatten_batch(space.flatten_batch(ref["batch_x"])),
        ref["batch_x"],
    )


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_tree_inner_is_available_only_for_homogeneous_inner_leaves(case):
    space = case.obj
    ref = case.reference

    if ref["profile"] == "inner":
        expected = sum(
            leaf_space.inner(x, y)
            for leaf_space, x, y in zip(
                space.leaf_spaces,
                space.flatten_tree(ref["x"]),
                space.flatten_tree(ref["y"]),
            )
        )
        assert space.inner(ref["x"], ref["y"]) == pytest.approx(expected)
    else:
        assert not isinstance(space, sc.InnerProductSpace)
        assert not hasattr(space, "inner")
