from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_space_field_is_independent_of_precision(dtype):
    space = sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), dtype=dtype))

    assert space.field == "real"


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_space_field_is_independent_of_precision(dtype):
    space = sc.DenseCoordinateSpace((2,), sc.Context(sc.NumpyOps(), dtype=dtype))

    assert space.field == "complex"


def test_product_and_stacked_spaces_expose_the_derived_field():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex64)
    base = sc.DenseCoordinateSpace((2,), ctx)

    assert base.stacked(3).field == "complex"
    assert sc.TreeSpace.from_leaf_spaces((base, base), ctx).field == "complex"


def test_space_conversion_recomputes_field_from_target_representation_dtype():
    real = sc.Context(sc.NumpyOps(), dtype=np.float32)
    complex_ = sc.Context(sc.NumpyOps(), dtype=np.complex64)
    space = sc.DenseCoordinateSpace((2,), real)

    assert space.convert(complex_).field == "complex"


def test_euclidean_elementwise_jordan_guard_uses_scalar_field():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex64)

    with pytest.raises(ValueError, match="requires a real scalar field"):
        sc.EuclideanElementwiseJordanSpace((2,), ctx)

    space = sc.ElementwiseJordanSpace((2,), ctx)
    assert space.field == "complex"
    assert not isinstance(space, sc.EuclideanElementwiseJordanSpace)


def test_field_check_and_exact_dtype_check_are_distinct():
    real_space = sc.DenseCoordinateSpace(
        (2,), sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="cheap")
    )
    complex_value = np.asarray([1.0, 2.0], dtype=np.complex64)

    assert not sc.FieldCheck().is_valid(real_space, complex_value)
    with pytest.raises(ValueError, match="real scalar field"):
        sc.FieldCheck()(real_space, complex_value)

    real_other_precision = np.asarray([1.0, 2.0], dtype=np.float64)
    assert sc.FieldCheck().is_valid(real_space, real_other_precision)
    assert not sc.DTypeCheck().is_valid(real_space, real_other_precision)


def test_complex_space_field_accepts_real_values_before_representation_check():
    complex_space = sc.DenseCoordinateSpace(
        (2,), sc.Context(sc.NumpyOps(), dtype=np.complex64, check_level="cheap")
    )
    real_value = np.asarray([1.0, 2.0], dtype=np.float32)

    assert sc.FieldCheck().is_valid(complex_space, real_value)
    assert not sc.DTypeCheck().is_valid(complex_space, real_value)

