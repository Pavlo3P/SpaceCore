from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import spacecore as sc


class FiniteSetSpace(sc.Space):
    def __init__(self, values: set[Any], ctx=None):
        super().__init__(ctx)
        self.values = values

    def _check_member(self, x: Any) -> None:
        if x not in self.values:
            raise ValueError("not a member")

    def _convert(self, new_ctx):
        return FiniteSetSpace(self.values, new_ctx)


class PairVectorSpace(sc.VectorSpace):
    def __init__(self, ctx=None):
        super().__init__(ctx)

    def zeros(self):
        return (0.0, 0.0)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def scale(self, a, x):
        return (a * x[0], a * x[1])

    def _convert(self, new_ctx):
        return PairVectorSpace(new_ctx)


class NonCoordinateInnerProductSpace(sc.InnerProductSpace):
    def __init__(self, ctx=None):
        super().__init__(ctx)
        self.geometry = sc.EuclideanInnerProduct()

    def zeros(self):
        return (0.0, 0.0)

    def add(self, x, y):
        return (x[0] + y[0], x[1] + y[1])

    def scale(self, a, x):
        return (a * x[0], a * x[1])

    def inner(self, x, y):
        return x[0] * y[0] + x[1] * y[1]

    def _convert(self, new_ctx):
        return NonCoordinateInnerProductSpace(new_ctx)


def test_minimal_space_membership_only():
    space = FiniteSetSpace({"a", "b"}, sc.Context(sc.NumpyOps(), enable_checks=True))

    space.check_member("a")
    with pytest.raises(ValueError, match="not a member"):
        space.check_member("c")

    assert not isinstance(space, sc.VectorSpace)
    assert not hasattr(space, "shape")


def test_vector_space_is_abstract_but_linear_capability_is_non_coordinate():
    with pytest.raises(TypeError):
        sc.VectorSpace()

    space = PairVectorSpace()

    assert isinstance(space, sc.VectorSpace)
    assert not isinstance(space, sc.CoordinateSpace)
    assert space.axpy(2.0, (1.0, 3.0), (-1.0, 4.0)) == (1.0, 10.0)


def test_coordinate_space_dense_vectors_matrices_tensors_and_stacking():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    for shape in ((3,), (2, 3), (2, 1, 3)):
        space = sc.DenseCoordinateSpace(shape, ctx)
        x = ctx.asarray(np.arange(space.size, dtype=float).reshape(shape))
        flat = space.flatten(x)
        batch = ctx.asarray(np.stack([np.asarray(x), np.asarray(x) + 1.0]))

        assert isinstance(space, sc.CoordinateSpace)
        assert tuple(flat.shape) == (space.size,)
        np.testing.assert_allclose(space.unflatten(flat), x)
        np.testing.assert_allclose(space.flatten_batch(batch), np.asarray(batch).reshape((2, -1)))
        assert space.stacked(2).shape == (2,) + shape


def test_dense_vector_space_is_only_one_dimensional():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseVectorSpace((4,), ctx)

    assert isinstance(space, sc.DenseVectorSpace)
    assert isinstance(space, sc.CoordinateSpace)
    assert isinstance(space, sc.InnerProductSpace)
    assert isinstance(space, sc.StarSpace)
    assert isinstance(space, sc.EuclideanJordanAlgebraSpace)

    with pytest.raises(ValueError, match="one-dimensional"):
        sc.DenseVectorSpace((2, 2), ctx)


def test_non_coordinate_inner_product_space_norm():
    space = NonCoordinateInnerProductSpace()

    assert isinstance(space, sc.InnerProductSpace)
    assert not isinstance(space, sc.CoordinateSpace)
    assert space.inner((1.0, 2.0), (3.0, 4.0)) == 11.0
    assert float(space.norm((3.0, 4.0))) == 5.0


def test_star_involution_and_conjugate_linearity():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    space = sc.DenseVectorSpace((2,), ctx)
    x = ctx.asarray([1.0 + 2.0j, -3.0 + 0.5j])
    alpha = 2.0 - 3.0j

    np.testing.assert_allclose(space.star(space.star(x)), x)
    np.testing.assert_allclose(space.star(alpha * x), np.conj(alpha) * np.asarray(space.star(x)))

    herm = sc.HermitianSpace(2, ctx=ctx)
    h = ctx.asarray([[1.0 + 0j, 2.0 - 1.0j], [2.0 + 1.0j, 3.0 + 0j]])
    np.testing.assert_allclose(herm.star(herm.star(h)), h)


def test_jordan_identity_for_dense_vectors_and_hermitian_space():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    vector = sc.DenseVectorSpace((3,), ctx)
    x = ctx.asarray([1.0, 2.0, -1.0])
    y = ctx.asarray([0.5, -3.0, 4.0])
    z = ctx.asarray([2.0, 1.0, 0.25])

    lhs = vector.inner(vector.jordan(x, y), z)
    rhs = vector.inner(y, vector.jordan(x, z))
    np.testing.assert_allclose(lhs, rhs)
    np.testing.assert_allclose(vector.spectral_apply(x, lambda t: t * t), x * x)

    herm = sc.HermitianSpace(2, ctx=ctx)
    a = ctx.asarray([[1.0, 0.25], [0.25, 2.0]])
    b = ctx.asarray([[0.5, -0.75], [-0.75, 3.0]])
    c = ctx.asarray([[2.0, 1.0], [1.0, -1.0]])
    np.testing.assert_allclose(herm.inner(herm.jordan(a, b), c), herm.inner(b, herm.jordan(a, c)))
