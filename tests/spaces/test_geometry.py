import importlib

import numpy as np

from tests._helpers import to_numpy


sc = importlib.import_module("spacecore")


class WeightedInnerProduct(sc.InnerProduct):
    def __init__(self, weights, convert_log=None):
        self.weights = weights
        self.convert_log = [] if convert_log is None else convert_log

    def inner(self, ops, x, y):
        return ops.vdot(x, self.weights * y)

    def riesz(self, ops, x):
        return self.weights * x

    def riesz_inverse(self, ops, x):
        return x / self.weights

    def convert(self, ctx):
        self.convert_log.append(ctx)
        return type(self)(ctx.asarray(self.weights), self.convert_log)

    @property
    def is_euclidean(self):
        return False

    def __eq__(self, other):
        return type(other) is type(self) and np.allclose(
            to_numpy(self.weights), to_numpy(other.weights)
        )


def _weighted_geometry(weights, ctx, convert_log=None):
    return WeightedInnerProduct(ctx.asarray(weights), convert_log)


def test_vector_space_default_geometry_is_euclidean():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.DenseCoordinateSpace((3,), ctx)
    x = ctx.asarray([1.0, 2.0, 3.0])
    y = ctx.asarray([4.0, 5.0, 6.0])

    assert space.is_euclidean is True
    assert np.allclose(space.inner(x, y), np.vdot(to_numpy(x), to_numpy(y)))
    assert space.riesz(x) is x
    assert space.riesz_inverse(x) is x


def test_hermitian_space_default_geometry_remains_frobenius_euclidean():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.HermitianSpace(2, ctx=ctx)
    x = ctx.asarray([[1.0, 2.0], [2.0, 3.0]])
    y = ctx.asarray([[4.0, -1.0], [-1.0, 2.0]])
    converted = space.convert(sc.Context(sc.NumpyOps(), dtype=np.float32))

    assert space.is_euclidean is True
    assert np.allclose(space.inner(x, y), np.vdot(to_numpy(x), to_numpy(y)))
    assert converted.is_euclidean is True
    assert type(converted.geometry) is sc.EuclideanInnerProduct


def test_vector_space_accepts_custom_geometry():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    geometry = _weighted_geometry([2.0, 3.0], ctx)
    space = sc.DenseCoordinateSpace((2,), ctx, geometry=geometry)
    x = ctx.asarray([1.0, 4.0])
    y = ctx.asarray([5.0, 6.0])

    assert space.is_euclidean is False
    assert np.allclose(space.inner(x, y), np.vdot(to_numpy(x), [10.0, 18.0]))
    assert np.allclose(space.riesz(x), [2.0, 12.0])
    assert np.allclose(space.riesz_inverse(space.riesz(x)), x)


def test_weighted_inner_product_geometry_is_shipped_and_converts():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    geometry = sc.WeightedInnerProduct(ctx.asarray([2.0, 3.0]))
    space = sc.DenseCoordinateSpace((2,), ctx, geometry=geometry)
    x = ctx.asarray([1.0, 4.0])
    y = ctx.asarray([5.0, 6.0])
    xb = ctx.asarray([[1.0, 4.0], [2.0, 3.0]])

    assert space.is_euclidean is False
    assert np.allclose(space.inner(x, y), np.vdot(to_numpy(x), [10.0, 18.0]))
    assert np.allclose(space.riesz(x), [2.0, 12.0])
    assert np.allclose(space.riesz_inverse(space.riesz(x)), x)
    assert np.allclose(space.riesz(xb), [[2.0, 12.0], [4.0, 9.0]])

    converted = geometry.convert(new_ctx)
    assert isinstance(converted, sc.WeightedInnerProduct)
    assert converted == sc.WeightedInnerProduct(new_ctx.asarray([2.0, 3.0]))


def test_space_equality_includes_geometry():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    euclidean_a = sc.DenseCoordinateSpace((2,), ctx)
    euclidean_b = sc.DenseCoordinateSpace((2,), ctx)
    weighted_a = sc.DenseCoordinateSpace((2,), ctx, geometry=_weighted_geometry([2.0, 3.0], ctx))
    weighted_b = sc.DenseCoordinateSpace((2,), ctx, geometry=_weighted_geometry([2.0, 3.0], ctx))
    weighted_c = sc.DenseCoordinateSpace((2,), ctx, geometry=_weighted_geometry([2.0, 4.0], ctx))

    assert euclidean_a == euclidean_b
    assert euclidean_a != weighted_a
    assert weighted_a == weighted_b
    assert weighted_a != weighted_c


def test_space_equality_is_symmetric_and_requires_exact_space_type():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    vector = sc.DenseCoordinateSpace((2, 2), ctx)
    hermitian_a = sc.HermitianSpace(2, ctx=ctx)
    hermitian_b = sc.HermitianSpace(2, ctx=ctx)
    weighted = sc.DenseCoordinateSpace((2, 2), ctx, geometry=_weighted_geometry([1.0, 2.0, 3.0, 4.0], ctx))
    product_a = sc.ProductSpace((sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx)
    product_b = sc.ProductSpace((sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx)
    product_reordered = sc.ProductSpace((sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((2,), ctx)), ctx)

    assert vector != hermitian_a
    assert hermitian_a != vector
    assert (vector == hermitian_a) == (hermitian_a == vector)
    assert hermitian_a == hermitian_b
    assert (hermitian_a == hermitian_b) == (hermitian_b == hermitian_a)
    assert vector != weighted
    assert (vector == weighted) == (weighted == vector)
    assert product_a == product_b
    assert (product_a == product_b) == (product_b == product_a)
    assert product_a != product_reordered


def test_vector_space_conversion_preserves_and_converts_geometry():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    convert_log = []
    geometry = _weighted_geometry([2.0, 3.0], src, convert_log)
    space = sc.DenseCoordinateSpace((2,), src, geometry=geometry)

    converted = space.convert(dst)

    assert converted.ctx == dst
    assert converted.geometry == _weighted_geometry([2.0, 3.0], dst)
    assert converted.geometry is not geometry
    assert convert_log == [dst]
    assert converted.ops.get_dtype(converted.geometry.weights) == dst.dtype


def test_product_space_geometry_is_componentwise():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    euclidean = sc.DenseCoordinateSpace((2,), ctx)
    weighted = sc.DenseCoordinateSpace((2,), ctx, geometry=_weighted_geometry([2.0, 3.0], ctx))
    product = sc.ProductSpace((euclidean, weighted), ctx)
    x = (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0, 4.0]))
    y = (ctx.asarray([5.0, 6.0]), ctx.asarray([7.0, 8.0]))

    assert sc.ProductSpace((euclidean, euclidean), ctx).is_euclidean is True
    assert product.is_euclidean is False
    expected_inner = euclidean.inner(x[0], y[0]) + weighted.inner(x[1], y[1])
    assert np.allclose(product.inner(x, y), expected_inner)

    dual = product.riesz(x)
    assert np.allclose(dual[0], x[0])
    assert np.allclose(dual[1], [6.0, 12.0])
    roundtrip = product.riesz_inverse(dual)
    assert np.allclose(roundtrip[0], x[0])
    assert np.allclose(roundtrip[1], x[1])


def test_product_space_equality_distinguishes_component_geometry():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    euclidean = sc.DenseCoordinateSpace((2,), ctx)
    weighted_a = sc.DenseCoordinateSpace((2,), ctx, geometry=_weighted_geometry([2.0, 3.0], ctx))
    weighted_b = sc.DenseCoordinateSpace((2,), ctx, geometry=_weighted_geometry([2.0, 4.0], ctx))

    assert sc.ProductSpace((euclidean, weighted_a), ctx) == sc.ProductSpace(
        (euclidean, weighted_a), ctx
    )
    assert sc.ProductSpace((euclidean, weighted_a), ctx) != sc.ProductSpace(
        (euclidean, weighted_b), ctx
    )
    assert sc.ProductSpace((euclidean, weighted_a), ctx) != sc.ProductSpace(
        (weighted_a, euclidean), ctx
    )
