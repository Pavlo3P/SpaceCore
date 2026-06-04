import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, to_numpy


def _contexts():
    sc = importlib.import_module("spacecore")
    yield sc.Context(sc.NumpyOps(), dtype=np.float64)
    if has_jax():
        yield sc.Context(sc.JaxOps(), dtype=np.float32)


def _assert_vapply_loop(op, xs):
    actual = to_numpy(op.vapply(xs))
    expected = np.stack([to_numpy(op.apply(x)) for x in xs], axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def _assert_rvapply_loop(op, ys):
    actual = to_numpy(op.rvapply(ys))
    expected = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("ctx", list(_contexts()))
def test_dense_linop_batched_apply_matches_loop(ctx):
    sc = importlib.import_module("spacecore")
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    op = sc.DenseLinOp(matrix, domain, codomain, ctx)

    _assert_vapply_loop(op, ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]))
    _assert_rvapply_loop(op, ctx.asarray([[1.0, 0.0, -2.0], [3.0, -1.0, 0.5]]))


@pytest.mark.parametrize("ctx", list(_contexts()))
def test_diagonal_linop_batched_apply_matches_loop(ctx):
    sc = importlib.import_module("spacecore")
    space = sc.DenseCoordinateSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([2.0, -1.0, 0.5]), space, ctx)
    xs = ctx.asarray([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]])

    _assert_vapply_loop(op, xs)
    _assert_rvapply_loop(op, xs)


@pytest.mark.parametrize("ctx", list(_contexts()))
def test_matrix_free_linop_batched_apply_matches_loop(ctx):
    sc = importlib.import_module("spacecore")
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    op = sc.MatrixFreeLinOp(
        lambda x: matrix @ x,
        lambda y: matrix.T @ y,
        domain,
        codomain,
        ctx,
    )

    _assert_vapply_loop(op, ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]))
    _assert_rvapply_loop(op, ctx.asarray([[1.0, 0.0, -2.0], [3.0, -1.0, 0.5]]))


@pytest.mark.parametrize("ctx", list(_contexts()))
def test_composed_linop_batched_apply_matches_loop(ctx):
    sc = importlib.import_module("spacecore")
    space = sc.DenseCoordinateSpace((2,), ctx)
    left = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, -1.0]]), space, space, ctx)
    right = sc.DiagonalLinOp(ctx.asarray([2.0, -0.5]), space, ctx)
    op = left @ right
    xs = ctx.asarray([[1.0, 2.0], [-1.0, 0.5]])

    _assert_vapply_loop(op, xs)
    _assert_rvapply_loop(op, xs)
