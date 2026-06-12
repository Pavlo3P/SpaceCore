from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy
from tests.generators import jordan_space_cases, mixed_jordan_tree_case
from tests.spaces._generated_helpers import assert_allclose, case_params, tolerances


CASES = jordan_space_cases()
HERMITIAN_POLICY_CASES = tuple(
    case
    for level in sc.CHECK_LEVELS
    for case in jordan_space_cases(check_level=level)
    if case.reference["kind"] == "hermitian"
)


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_jordan_closure_star_and_spectrum(case):
    space = case.obj
    ref = case.reference
    product = space.jordan(ref["x"], ref["y"])
    spectrum = space.spectrum(ref["x"])

    space.check_member(product)
    assert_allclose(space, space.star(space.star(ref["x"])), ref["x"])
    if ref["kind"] == "hermitian":
        assert spectrum.shape == (space.n,)
        assert space.ops.get_dtype(spectrum) == space.ops.real_dtype(space.dtype)
    elif ref["kind"] == "tree":
        expected_size = sum(leaf.size for leaf in space.leaf_spaces)
        assert spectrum.shape == (expected_size,)
    else:
        assert spectrum.shape == space.shape
        assert space.ops.get_dtype(spectrum) == space.dtype


@pytest.mark.parametrize("case", case_params(CASES))
def test_generated_jordan_spectral_roundtrip(case):
    space = case.obj
    ref = case.reference
    decomposition = space.spectral_decompose(ref["x"])
    if isinstance(space, sc.TreeSpace):
        rebuilt = space.from_spectrum(decomposition)
    else:
        eigvals, frame = decomposition
        rebuilt = space.from_spectrum(eigvals, frame)

    assert_allclose(space, rebuilt, ref["x"])


@pytest.mark.parametrize(
    "case",
    case_params(tuple(case for case in CASES if case.reference["kind"] == "hermitian")),
)
def test_generated_strict_hermitian_spectral_input_is_validated(case):
    space = case.obj

    with pytest.raises(sc.SpaceValidationError, match="not Hermitian"):
        space.spectrum(case.reference["invalid_spectral"])


@pytest.mark.parametrize("case", case_params(HERMITIAN_POLICY_CASES))
def test_generated_hermitian_membership_follows_check_level(case):
    space = case.obj
    invalid = case.reference["invalid_spectral"]

    if space.check_level in {"none", "cheap"}:
        space.check_member(invalid)
    else:
        with pytest.raises(sc.SpaceValidationError, match="not Hermitian"):
            space.check_member(invalid)


def test_generated_mixed_tree_does_not_advertise_jordan_or_star():
    case = mixed_jordan_tree_case()
    space = case.obj

    assert isinstance(space, sc.InnerProductSpace)
    assert not isinstance(space, sc.JordanAlgebraSpace)
    assert not isinstance(space, sc.StarSpace)
    assert not hasattr(space, "jordan")
    assert not hasattr(space, "spectrum")


def test_jordan_capabilities_are_recomputed_on_real_to_complex_conversion():
    real = sc.Context(sc.NumpyOps(), dtype=np.float64)
    complex_ = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    space = sc.EuclideanElementwiseJordanSpace((2,), real)
    converted = space.convert(complex_)
    rtol, atol = tolerances(converted.dtype)

    assert type(converted) is sc.ElementwiseJordanSpace
    assert isinstance(converted, sc.JordanAlgebraSpace)
    assert not isinstance(converted, sc.EuclideanJordanAlgebraSpace)
    np.testing.assert_allclose(
        to_numpy(converted.jordan(complex_.asarray([1.0, 2.0]), complex_.asarray([3.0, 4.0]))),
        [3.0, 8.0],
        rtol=rtol,
        atol=atol,
    )

    tree = sc.TreeSpace.from_leaf_spaces((space, space), real)
    converted_tree = tree.convert(complex_)
    assert isinstance(converted_tree, sc.JordanAlgebraSpace)
    assert not isinstance(converted_tree, sc.EuclideanJordanAlgebraSpace)
    assert converted_tree.treedef == tree.treedef
    assert converted_tree.leaf_paths == tree.leaf_paths
