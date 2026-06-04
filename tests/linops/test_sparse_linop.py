import importlib

import numpy as np
import pytest
import scipy.sparse as sps
from tests._helpers import to_numpy
from test_generators.linop._sparse import (
    bare_sparse_linop,
    check_sparse_linop,
    make_sparse_linop_data,
)


class TransposeCountingCSR(sps.csr_matrix):
    def __init__(self, *args, counter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = counter

    @property
    def T(self):
        if self.counter is not None:
            self.counter["calls"] += 1
        return super().T


def test_sparse_linop_construct_apply_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    dense = np.array([[1., 2.], [3., 4.], [5., 6.]])
    op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)

    x = ctx.asarray([7., 8.])
    y = ctx.asarray([1., -1., 2.])

    assert np.allclose(op.apply(x), dense @ np.asarray(x))
    assert np.allclose(op.rapply(y), dense.T @ np.asarray(y))
    assert np.allclose(op.to_dense(), dense)
    assert np.allclose(op.to_matrix(), dense)


def test_sparse_linop_rectangular_batched_apply_and_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    dense = np.array([[1.0, -2.0], [3.0, 0.5], [0.25, 4.0]])
    op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)

    xs = ctx.asarray([[1.0, 2.0], [-3.0, 4.0], [0.5, -1.5]])
    ys = ctx.asarray([[2.0, -1.0, 0.5], [1.5, 3.0, -2.0]])

    assert op.to_dense().shape == (3, 2)
    assert op.to_matrix().shape == (3, 2)
    assert np.allclose(op.apply(xs[0]), dense @ np.asarray(xs[0]))
    assert np.allclose(op.rapply(ys[0]), dense.T @ np.asarray(ys[0]))
    for i in range(xs.shape[0]):
        assert np.allclose(op.vapply(xs)[i], op.apply(xs[i]))
    for i in range(ys.shape[0]):
        assert np.allclose(op.rvapply(ys)[i], op.rapply(ys[i]))


def test_sparse_linop_tensor_shaped_euclidean_behavior():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2, 2), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    dense = np.array(
        [
            [1.0, -2.0, 0.5, 3.0],
            [0.25, 4.0, -1.0, 2.0],
            [3.0, 0.5, 2.0, -0.75],
        ]
    )
    op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)
    x = ctx.asarray([[1.0, 2.0], [-3.0, 0.5]])
    y = ctx.asarray([2.0, -1.0, 0.25])
    xs = ctx.asarray([[[1.0, 0.0], [2.0, -1.0]], [[0.5, 3.0], [-2.0, 1.5]]])
    ys = ctx.asarray([[2.0, -1.0, 0.25], [0.5, 3.0, -2.0]])

    np.testing.assert_allclose(op.apply(x), dense @ np.asarray(x).reshape(-1))
    np.testing.assert_allclose(op.rapply(y), (dense.T @ np.asarray(y)).reshape((2, 2)))
    for i in range(xs.shape[0]):
        np.testing.assert_allclose(op.vapply(xs)[i], op.apply(xs[i]))
    for i in range(ys.shape[0]):
        np.testing.assert_allclose(op.rvapply(ys)[i], op.rapply(ys[i]))


def test_sparse_linop_complex_rapply_uses_conjugate_transpose():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((2,), ctx)
    dense = np.array(
        [[1.0 + 2.0j, 3.0 - 1.0j], [-2.0j, 4.0 + 0.5j]],
        dtype=np.complex128,
    )
    op = sc.SparseLinOp(ctx.assparse(dense), dom, cod, ctx)
    y = ctx.asarray([1.0 - 1.0j, 2.0 + 0.25j])

    assert np.allclose(op.rapply(y), dense.conj().T @ np.asarray(y))


def test_sparse_linop_accepts_euclidean_vector_space_subclass():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    class WeightedVectorSpace(sc.DenseCoordinateSpace):
        pass

    space = WeightedVectorSpace((2,), ctx)
    matrix = ctx.assparse([[1.0, 0.0], [0.0, 1.0]])
    op = sc.SparseLinOp(matrix, space, space, ctx)
    x = ctx.asarray([2.0, -1.0])

    assert type(op.domain) is WeightedVectorSpace
    assert np.allclose(op.apply(x), x)
    assert np.allclose(op.rapply(x), x)


def test_sparse_linop_accepts_coordinate_product_spaces():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    vector = sc.DenseCoordinateSpace((2,), ctx)
    product = sc.ProductSpace((sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx)
    matrix = ctx.assparse(np.eye(2))
    x_product = (ctx.asarray([1.0]), ctx.asarray([-2.0]))
    x_vector = ctx.asarray([0.5, 3.0])

    to_vector = sc.SparseLinOp(matrix, product, vector, ctx)
    to_product = sc.SparseLinOp(matrix, vector, product, ctx)

    np.testing.assert_allclose(to_numpy(to_vector.apply(x_product)), [1.0, -2.0])
    y = to_product.apply(x_vector)
    np.testing.assert_allclose(to_numpy(y[0]), [0.5])
    np.testing.assert_allclose(to_numpy(y[1]), [3.0])


def test_sparse_linop_weighted_metric_adjoint_identity():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    domain = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0])))
    codomain = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([3.0, 7.0, 11.0])))
    dense = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    op = sc.SparseLinOp(ctx.assparse(dense), domain, codomain, ctx)
    x = ctx.asarray([0.25, -1.5])
    y = ctx.asarray([2.0, -0.5, 1.25])

    lhs = op.codomain.inner(op.apply(x), y)
    rhs = op.domain.inner(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs))


def test_sparse_linop_general_metric_adjoint_identity():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    class ScalingInnerProduct(sc.InnerProduct):
        def __init__(self, weights):
            self.weights = weights

        def inner(self, ops, x, y):
            return ops.vdot(x, self.weights * y)

        def riesz(self, ops, x):
            return self.weights * x

        def riesz_inverse(self, ops, x):
            return x / self.weights

        @property
        def is_euclidean(self):
            return False

    domain = sc.DenseCoordinateSpace((2,), ctx, geometry=ScalingInnerProduct(ctx.asarray([2.0, 5.0])))
    codomain = sc.DenseCoordinateSpace((3,), ctx, geometry=ScalingInnerProduct(ctx.asarray([3.0, 7.0, 11.0])))
    dense = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    op = sc.SparseLinOp(ctx.assparse(dense), domain, codomain, ctx)
    x = ctx.asarray([0.25, -1.5])
    y = ctx.asarray([2.0, -0.5, 1.25])

    lhs = op.codomain.inner(op.apply(x), y)
    rhs = op.domain.inner(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs))


def test_sparse_linop_to_sparse_returns_stored_object():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    matrix = sps.csr_matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.SparseLinOp(matrix, dom, cod, ctx)

    assert op.A is matrix
    assert op.to_sparse() is matrix


def test_sparse_linop_convert_preserves_action_and_converts_sparse_storage():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), src)
    cod = sc.DenseCoordinateSpace((3,), src)
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    op = sc.SparseLinOp(src.assparse(dense), dom, cod, src)

    op2 = op.convert(dst)
    x = op2.ctx.asarray([7.0, 8.0])

    assert op2 is not op
    assert type(op2.dom) is sc.DenseCoordinateSpace
    assert type(op2.cod) is sc.DenseCoordinateSpace
    assert op2.ops.get_dtype(op2.A) == dst.dtype
    assert np.allclose(op2.apply(x), dense.astype(np.float64) @ np.asarray(x))


def test_sparse_linop_reuses_cached_transpose():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    cod = sc.DenseCoordinateSpace((3,), ctx)
    counter = {"calls": 0}
    A = TransposeCountingCSR([[1., 2.], [3., 4.], [5., 6.]], counter=counter)

    op = sc.SparseLinOp(A, dom, cod, ctx)
    transpose_calls = counter["calls"]

    op.rapply(ctx.asarray([1., -1., 2.]))
    op.rapply(ctx.asarray([3., -2., 1.]))

    assert transpose_calls == 1
    assert counter["calls"] == transpose_calls


def test_sparse_linop_test_data_bare_reference_tensor_euclidean():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    data = make_sparse_linop_data(
        ctx,
        domain_shape=(2, 3),
        codomain_shape=(4, 2),
        batch=5,
        weighted=False,
        seed=12,
    )
    op = sc.SparseLinOp(data.operator, data.domain, data.codomain, ctx)

    assert np.allclose(bare_sparse_linop(ctx.ops, data, "apply"), op.apply(data.x))
    assert np.allclose(bare_sparse_linop(ctx.ops, data, "rapply"), op.rapply(data.y))
    assert np.allclose(bare_sparse_linop(ctx.ops, data, "vapply"), op.vapply(data.xs))
    assert np.allclose(bare_sparse_linop(ctx.ops, data, "rvapply"), op.rvapply(data.ys))


def test_sparse_linop_test_data_bare_reference_flat_weighted_and_timing():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    data = make_sparse_linop_data(
        ctx,
        domain_shape=(6,),
        codomain_shape=(4,),
        batch=3,
        weighted=True,
        seed=14,
    )
    op = sc.SparseLinOp(data.operator, data.domain, data.codomain, ctx)

    assert data.domain_weights is not None
    assert data.codomain_weights is not None
    assert np.allclose(bare_sparse_linop(ctx.ops, data, "apply"), op.apply(data.x))
    assert np.allclose(bare_sparse_linop(ctx.ops, data, "rapply"), op.rapply(data.y))
    assert np.allclose(bare_sparse_linop(ctx.ops, data, "vapply"), op.vapply(data.xs))
    assert np.allclose(bare_sparse_linop(ctx.ops, data, "rvapply"), op.rvapply(data.ys))

    timed = bare_sparse_linop(ctx.ops, data, "rapply", time=True)
    assert np.allclose(timed, op.rapply(data.y))
    assert data.bare_time_s["rapply"] >= 0.0


@pytest.mark.parametrize("batch", [None, 4])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("domain_shape", [(5,), (2, 3)])
@pytest.mark.parametrize("codomain_shape", [(7,), (3, 2)])
@pytest.mark.parametrize("kind", ["apply", "rapply", "vapply", "rvapply"])
def test_check_sparse_linop_covers_shapes_batches_and_geometry(
    batch,
    weighted,
    domain_shape,
    codomain_shape,
    kind,
):
    if batch is None and kind in {"vapply", "rvapply"}:
        pytest.skip("batched checks require generated batch data")
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    data = make_sparse_linop_data(
        ctx,
        domain_shape=domain_shape,
        codomain_shape=codomain_shape,
        batch=batch,
        weighted=weighted,
        seed=(
            (0 if batch is None else batch)
            + 10 * int(weighted)
            + 100 * (1 if domain_shape == (2, 3) else 0)
            + 1_000 * (1 if codomain_shape == (3, 2) else 0)
            + 10_000 * ["apply", "rapply", "vapply", "rvapply"].index(kind)
        ),
    )

    assert check_sparse_linop(data, kind)
    assert kind in data.bare_outputs
    assert kind in data.spacecore_outputs


def test_check_sparse_linop_records_bare_and_spacecore_timing():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    data = make_sparse_linop_data(
        ctx,
        domain_shape=(5,),
        codomain_shape=(7,),
        batch=4,
        weighted=True,
        seed=22,
    )

    assert check_sparse_linop(data, "rvapply", time=True)
    assert "rvapply" in data.bare_time_s
    assert "rvapply" in data.spacecore_time_s
    assert "rvapply" in data.bare_outputs
    assert "rvapply" in data.spacecore_outputs
    assert data.bare_time_s["rvapply"] >= 0.0
    assert data.spacecore_time_s["rvapply"] >= 0.0
