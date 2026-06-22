from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests.generators import (
    BatchCase,
    batch_cases,
    check_level_params,
    context_cases,
    dense_array_case,
    dense_array_cases,
    hermitian_case,
    seeded_rng,
    spd_metric_case,
    tree_space_cases,
)


def _ctx(dtype=np.float64):
    return sc.Context(sc.NumpyOps(), dtype=dtype)


def test_package_imports_and_seed_policy_are_deterministic():
    first = seeded_rng(17).standard_normal((3, 2))
    second = seeded_rng(17).standard_normal((3, 2))

    np.testing.assert_array_equal(first, second)
    with pytest.raises(TypeError, match="either seed or rng"):
        dense_array_case(_ctx(), (2,), seed=1, rng=seeded_rng(2))


def test_context_check_level_and_batch_parameters_cover_core_cases():
    cases = context_cases(include_optional=False)

    assert [case.id for case in cases] == ["numpy-float64", "numpy-complex128"]
    assert all(isinstance(case.obj, sc.Context) for case in cases)
    assert len(check_level_params()) == 4
    with pytest.raises(ValueError, match="Unknown check level"):
        check_level_params(("invalid",))
    assert batch_cases() == (BatchCase(False, ()), BatchCase(True, (2,)))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_dense_arrays_match_requested_shape_field_and_dtype(dtype):
    ctx = _ctx(dtype)
    cases = dense_array_cases(ctx, batch_shape=(2,), seed=9)

    assert [case.reference["shape"] for case in cases] == [(), (3,), (2, 3), (2, 2, 2)]
    for case in cases:
        reference = case.reference["array"]
        assert case.obj.shape == (2,) + case.reference["shape"]
        assert case.obj.dtype == ctx.dtype
        np.testing.assert_array_equal(case.obj, reference)


def test_dense_generator_rejects_complex_values_for_real_context():
    with pytest.raises(TypeError, match="real context"):
        dense_array_case(_ctx(), (2,), field="complex")


def test_hermitian_generator_supports_complex_batches():
    case = hermitian_case(_ctx(np.complex128), 3, batch_shape=(2,), seed=4)
    matrix = case.reference["matrix"]

    np.testing.assert_allclose(matrix, np.swapaxes(matrix.conj(), -1, -2), atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_spd_metric_generator_has_positive_eigenvalues(dtype):
    case = spd_metric_case(_ctx(dtype), 4, condition_number=20.0, seed=5)
    matrix = case.reference["matrix"]

    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(matrix) > 0)
    np.testing.assert_allclose(matrix @ case.reference["inverse"], np.eye(4), atol=1e-10)


def test_tree_generators_produce_valid_values_paths_and_mismatches():
    for case in tree_space_cases(_ctx(), seed=6):
        space = case.obj
        element = case.reference["element"]

        space.check_member(element)
        assert space.flatten_tree(element) == case.reference["leaves"]
        assert space.leaf_paths == case.reference["leaf_paths"]
        with pytest.raises(TypeError, match="structure mismatch"):
            space.check_member(case.reference["mismatch"])
