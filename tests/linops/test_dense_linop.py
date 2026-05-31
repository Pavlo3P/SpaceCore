import importlib
import numpy as np
import pytest
from tests._helpers import has_jax, jax_real_dtype, to_numpy


class ReshapeCountingArray(np.ndarray):
    def __new__(cls, data, counter):
        obj = np.asarray(data).view(cls)
        obj.counter = counter
        obj._track_reshape = True
        return obj

    def __array_finalize__(self, obj):
        self.counter = getattr(obj, "counter", None)
        self._track_reshape = False

    def reshape(self, *shape, **kwargs):
        if self.counter is not None and self._track_reshape:
            self.counter["calls"] += 1
        return super().reshape(*shape, **kwargs)


def test_dense_linop_construct_apply_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    A = ctx.asarray([[1.,2.],[3.,4.],[5.,6.]])
    op = sc.DenseLinOp(A, dom, cod, ctx)
    x = ctx.asarray([7.,8.])
    y = ctx.asarray([1.,-1.,2.])
    assert np.allclose(op.apply(x), np.array([[1.,2.],[3.,4.],[5.,6.]]) @ np.array([7.,8.]))
    assert np.allclose(op.rapply(y), np.array([[1.,2.],[3.,4.],[5.,6.]]).T @ np.array([1.,-1.,2.]))
    assert np.allclose(op.to_dense(), np.array([[1., 2.], [3., 4.], [5., 6.]]))


def test_dense_linop_rectangular_batched_apply_and_rapply():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    matrix = np.array([[1.0, -2.0], [3.0, 0.5], [0.25, 4.0]])
    op = sc.DenseLinOp(ctx.asarray(matrix), dom, cod, ctx)

    xs = ctx.asarray([[1.0, 2.0], [-3.0, 4.0], [0.5, -1.5]])
    ys = ctx.asarray([[2.0, -1.0, 0.5], [1.5, 3.0, -2.0]])

    assert op.to_dense().shape == (3, 2)
    assert op.to_matrix().shape == (3, 2)
    assert np.allclose(op.apply(xs[0]), matrix @ np.asarray(xs[0]))
    assert np.allclose(op.rapply(ys[0]), matrix.T @ np.asarray(ys[0]))
    assert np.allclose(op.vapply(xs), np.asarray(xs) @ matrix.T)
    assert np.allclose(op.rvapply(ys), np.asarray(ys) @ matrix)


def test_dense_linop_complex_rapply_uses_conjugate_transpose():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((2,), ctx)
    matrix = np.array(
        [[1.0 + 2.0j, 3.0 - 1.0j], [-2.0j, 4.0 + 0.5j]],
        dtype=np.complex128,
    )
    op = sc.DenseLinOp(ctx.asarray(matrix), dom, cod, ctx)
    y = ctx.asarray([1.0 - 1.0j, 2.0 + 0.25j])

    assert np.allclose(op.rapply(y), matrix.conj().T @ np.asarray(y))


def test_dense_linop_is_hermitian_uses_euclidean_matrix():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    space = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)

    hermitian = sc.DenseLinOp(
        ctx.asarray([[2.0, 1.0 + 2.0j], [1.0 - 2.0j, 5.0]]),
        space,
        space,
        ctx,
    )
    non_hermitian = sc.DenseLinOp(
        ctx.asarray([[1.0, 2.0], [3.0, 4.0]]),
        space,
        space,
        ctx,
    )
    rectangular = sc.DenseLinOp(ctx.asarray(np.ones((3, 2))), space, cod, ctx)

    assert hermitian.is_hermitian() is True
    assert non_hermitian.is_hermitian() is False
    assert rectangular.is_hermitian() is False


def test_dense_linop_rejects_non_plain_vector_space():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    class WeightedVectorSpace(sc.VectorSpace):
        pass

    custom = WeightedVectorSpace((2,), ctx)
    plain = sc.VectorSpace((2,), ctx)
    matrix = ctx.asarray([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(TypeError, match="plain VectorSpace"):
        sc.DenseLinOp(matrix, custom, plain, ctx)
    with pytest.raises(TypeError, match="metric-aware"):
        sc.DenseLinOp(matrix, plain, custom, ctx)


def test_dense_linop_reuses_cached_matrix_reshape():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    counter = {"calls": 0}
    A = ReshapeCountingArray([[1., 2.], [3., 4.], [5., 6.]], counter)

    op = sc.DenseLinOp(A, dom, cod, ctx)
    matrix_reshape_calls = counter["calls"]

    op.apply(ctx.asarray([7., 8.]))
    op.rapply(ctx.asarray([1., -1., 2.]))
    op.apply(ctx.asarray([9., 10.]))
    op.rapply(ctx.asarray([3., -2., 1.]))

    assert matrix_reshape_calls == 1
    assert counter["calls"] == matrix_reshape_calls


def test_dense_linop_bad_shape_raises():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    with pytest.raises(Exception):
        sc.DenseLinOp(ctx.asarray([[1.,2.],[3.,4.]]), sc.VectorSpace((2,), ctx), sc.VectorSpace((3,), ctx), ctx)


def test_dense_linop_convert_preserves_action():
    sc = importlib.import_module("spacecore")
    src = sc.Context(sc.NumpyOps(), dtype=np.float32)
    dst = sc.Context(sc.NumpyOps(), dtype=np.float64)
    op = sc.DenseLinOp(src.asarray([[1.,2.],[3.,4.],[5.,6.]]), sc.VectorSpace((2,), src), sc.VectorSpace((3,), src), src)
    op2 = op.convert(dst)
    x = op2.ctx.asarray([7.,8.])
    assert type(op2.dom) is sc.VectorSpace
    assert type(op2.cod) is sc.VectorSpace
    assert op2.ops.get_dtype(op2.A) == dst.dtype
    assert np.allclose(to_numpy(op2.apply(x)), [23.,53.,83.])


def test_dense_linop_tree_flatten_unflatten_round_trip():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    op = sc.DenseLinOp(matrix, dom, cod, ctx)

    children, aux = op.tree_flatten()
    restored = sc.DenseLinOp.tree_unflatten(aux, children)

    assert type(restored.dom) is sc.VectorSpace
    assert type(restored.cod) is sc.VectorSpace
    assert np.allclose(restored.to_dense(), matrix)
    assert np.allclose(restored.apply(ctx.asarray([7.0, 8.0])), op.apply(ctx.asarray([7.0, 8.0])))


def test_dense_linop_convert_to_jax_if_supported():
    if not has_jax():
        return
    sc = importlib.import_module("spacecore")
    dt = jax_real_dtype()
    src = sc.Context(sc.NumpyOps(), dtype=dt)
    dst = sc.Context(sc.JaxOps(), dtype=dt)
    op = sc.DenseLinOp(src.asarray([[1.,2.],[3.,4.],[5.,6.]]), sc.VectorSpace((2,), src), sc.VectorSpace((3,), src), src)
    op2 = op.convert(dst)
    assert op2.ctx.ops.family == "jax"
