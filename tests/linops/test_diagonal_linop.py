import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, to_numpy


def _ctx(dtype=np.float64, enable_checks=True):
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=enable_checks)


def test_apply_and_rapply_flat_diagonal():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)
    x = ctx.asarray([4.0, -1.0, 0.5])

    np.testing.assert_allclose(op.apply(x), [4.0, -2.0, 1.5])
    np.testing.assert_allclose(op.rapply(x), [4.0, -2.0, 1.5])


def test_apply_and_rapply_tensor_shaped_diagonal():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2, 2), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0]]), space, ctx)
    x = ctx.asarray([[2.0, -1.0], [0.5, 3.0]])

    np.testing.assert_allclose(op.apply(x), [[2.0, -2.0], [1.5, 12.0]])
    np.testing.assert_allclose(op.rapply(x), [[2.0, -2.0], [1.5, 12.0]])


def test_complex_diagonal_satisfies_adjoint_identity_and_hermitian_predicate():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(np.complex128)
    space = sc.VectorSpace((2,), ctx)
    hermitian = sc.DiagonalLinOp(ctx.asarray([1.0 + 0.0j, 2.0 + 0.0j]), space, ctx)
    non_hermitian = sc.DiagonalLinOp(ctx.asarray([1.0 + 2.0j, 3.0 - 1.0j]), space, ctx)
    u = ctx.asarray([2.0 - 1.0j, -0.5 + 0.25j])
    v = ctx.asarray([1.5 + 0.5j, -2.0j])

    lhs = space.inner(non_hermitian.apply(u), v)
    rhs = space.inner(u, non_hermitian.rapply(v))

    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs))
    assert hermitian.is_hermitian() is True
    assert non_hermitian.is_hermitian() is False


def test_vapply_and_rvapply_with_leading_batch_space():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)
    xs = ctx.asarray([[1.0, 2.0, 3.0], [4.0, -1.0, 0.5]])
    batch_space = space.batch((2,), (0,))

    expected = ctx.asarray([[1.0, 4.0, 9.0], [4.0, -2.0, 1.5]])
    np.testing.assert_allclose(op.vapply(xs, batch_space), expected)
    np.testing.assert_allclose(op.rvapply(xs, batch_space), expected)


def test_vapply_and_rvapply_with_non_leading_batch_space():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)
    xs = ctx.asarray([[1.0, 4.0], [2.0, -1.0], [3.0, 0.5]])
    batch_space = space.batch((2,), (1,))

    expected = ctx.asarray([[1.0, 4.0], [4.0, -2.0], [9.0, 1.5]])
    np.testing.assert_allclose(op.vapply(xs, batch_space), expected)
    np.testing.assert_allclose(op.rvapply(xs, batch_space), expected)


def test_to_dense_matches_numpy_diagonal_for_tensor_space():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2, 2), ctx)
    diagonal = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])
    op = sc.DiagonalLinOp(diagonal, space, ctx)

    expected = np.diag(np.asarray(diagonal).reshape((4,))).reshape((2, 2, 2, 2))

    np.testing.assert_allclose(op.to_dense(), expected)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_pytree_flatten_unflatten_round_trip():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), space, ctx)

    leaves, treedef = jax.tree_util.tree_flatten(op)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert restored == op
    np.testing.assert_allclose(restored.apply(ctx.asarray([1.0, 1.0, 1.0])), [1.0, 2.0, 3.0])


def test_convert_changes_context_dtype():
    sc = importlib.import_module("spacecore")
    ctx64 = _ctx(np.float64)
    ctx32 = _ctx(np.float32)
    space = sc.VectorSpace((3,), ctx64)
    op = sc.DiagonalLinOp(ctx64.asarray([1.0, 2.0, 3.0]), space, ctx64)

    converted = op._convert(ctx32)

    assert converted.ctx == ctx32
    assert converted.domain.ctx == ctx32
    assert converted.diagonal.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(converted.apply(ctx32.asarray([1.0, 1.0, 1.0])), [1.0, 2.0, 3.0])
