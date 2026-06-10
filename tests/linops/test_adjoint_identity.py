import importlib
import warnings

import numpy as np
import pytest

from tests._helpers import has_jax, jax_complex_dtype, jax_real_dtype, to_numpy


def _contexts(complex_values=False):
    sc = importlib.import_module("spacecore")
    yield sc.Context(sc.NumpyOps(), dtype=np.complex128 if complex_values else np.float64)
    if has_jax():
        yield sc.Context(
            sc.JaxOps(),
            dtype=jax_complex_dtype() if complex_values else jax_real_dtype(),
            enable_checks=False,
        )


def _assert_adjoint_identity(op, x, y, rtol=1e-6, atol=1e-6):
    lhs = op.codomain.inner(op.apply(x), y)
    rhs = op.domain.inner(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=rtol, atol=atol)


def _weighted_space_class():
    sc = importlib.import_module("spacecore")

    class WeightedVectorSpace(sc.DenseCoordinateSpace):
        def __init__(self, weights, ctx):
            self.weights = ctx.asarray(weights)
            super().__init__(
                tuple(self.weights.shape), ctx, geometry=sc.WeightedInnerProduct(self.weights)
            )

        def _convert(self, new_ctx):
            return WeightedVectorSpace(new_ctx.asarray(self.weights), new_ctx)

        def __eq__(self, other):
            return (
                type(other) is type(self)
                and super().__eq__(other)
                and np.allclose(to_numpy(self.weights), to_numpy(other.weights))
            )

    return WeightedVectorSpace


def _assert_vapply_loop(op, xs, rtol=1e-6, atol=1e-6):
    actual = to_numpy(op.vapply(xs))
    expected = (
        to_numpy(tuple(op.apply(tuple(xi[i] for xi in xs)) for i in range(xs[0].shape[0])))
        if isinstance(xs, tuple)
        else np.stack([to_numpy(op.apply(x)) for x in xs], axis=0)
    )
    if isinstance(actual, tuple):
        for actual_part, expected_rows in zip(actual, zip(*expected)):
            np.testing.assert_allclose(
                actual_part, np.stack(expected_rows, axis=0), rtol=rtol, atol=atol
            )
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def _assert_rvapply_loop(op, ys, rtol=1e-6, atol=1e-6):
    actual = to_numpy(op.rvapply(ys))
    expected = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ctx", list(_contexts(False)))
def test_euclidean_real_adjoint_identity_for_matrix_backed_ops(ctx):
    sc = importlib.import_module("spacecore")
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    x = ctx.asarray([0.25, -1.5])
    y = ctx.asarray([2.0, -0.5, 1.25])

    dense = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    sparse = sc.SparseLinOp(ctx.assparse(matrix), domain, codomain, ctx)
    diagonal_space = sc.DenseCoordinateSpace((3,), ctx)
    diagonal = sc.DiagonalLinOp(ctx.asarray([2.0, -1.0, 0.5]), diagonal_space, ctx)
    z = ctx.asarray([1.0, -2.0, 0.75])
    w = ctx.asarray([-0.5, 3.0, 1.25])

    _assert_adjoint_identity(dense, x, y)
    _assert_adjoint_identity(sparse, x, y)
    _assert_adjoint_identity(diagonal, z, w)


@pytest.mark.parametrize("ctx", list(_contexts(True)))
def test_euclidean_complex_adjoint_identity_for_matrix_backed_ops(ctx):
    sc = importlib.import_module("spacecore")
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = np.array([[1.0 + 0.5j, -2.0j], [0.5 - 1.0j, 3.0], [4.0, -1.0 + 2.0j]])
    x = ctx.asarray([0.25 + 1.0j, -1.5 + 0.5j])
    y = ctx.asarray([2.0 - 0.25j, -0.5 + 1.0j, 1.25j])

    dense = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    sparse = sc.SparseLinOp(ctx.assparse(matrix), domain, codomain, ctx)
    diagonal_space = sc.DenseCoordinateSpace((3,), ctx)
    diagonal = sc.DiagonalLinOp(ctx.asarray([2.0 + 1.0j, -1.0j, 0.5 - 0.25j]), diagonal_space, ctx)
    z = ctx.asarray([1.0 + 1.0j, -2.0, 0.75 - 0.5j])
    w = ctx.asarray([-0.5j, 3.0 + 0.25j, 1.25])

    _assert_adjoint_identity(dense, x, y)
    _assert_adjoint_identity(sparse, x, y)
    _assert_adjoint_identity(diagonal, z, w)


def test_dense_linop_accepts_hermitian_space_and_satisfies_adjoint_identity():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.HermitianSpace(2, ctx=ctx)
    identity_tensor = ctx.asarray(np.eye(4).reshape((2, 2, 2, 2)))
    op = sc.DenseLinOp(identity_tensor, space, space, ctx)
    x = ctx.asarray([[1.0, 2.0], [2.0, 3.0]])
    y = ctx.asarray([[4.0, -1.0], [-1.0, 2.0]])

    _assert_adjoint_identity(op, x, y)


@pytest.mark.parametrize("ctx", list(_contexts(False)))
def test_weighted_scalar_metric_adjoint_counterexample(ctx):
    sc = importlib.import_module("spacecore")
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0], ctx)
    codomain = WeightedVectorSpace([3.0], ctx)
    op = sc.DenseLinOp(ctx.asarray([[5.0]]), domain, codomain, ctx)
    y = ctx.asarray([4.0])

    np.testing.assert_allclose(to_numpy(op.rapply(y)), [30.0], rtol=1e-6, atol=1e-6)
    _assert_adjoint_identity(op, ctx.asarray([1.25]), y)


@pytest.mark.parametrize("ctx", list(_contexts(False)))
def test_weighted_vector_metric_adjoint_for_dense_sparse_and_diagonal(ctx):
    sc = importlib.import_module("spacecore")
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    codomain = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    matrix = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    x = ctx.asarray([0.25, -1.5])
    y = ctx.asarray([2.0, -0.5, 1.25])

    _assert_adjoint_identity(sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx), x, y)
    _assert_adjoint_identity(sc.SparseLinOp(ctx.assparse(matrix), domain, codomain, ctx), x, y)

    diagonal_space = WeightedVectorSpace([2.0, 3.0], ctx)
    diagonal = sc.DiagonalLinOp(ctx.asarray([2.0, -0.5]), diagonal_space, ctx)
    _assert_adjoint_identity(diagonal, ctx.asarray([1.0, -2.0]), ctx.asarray([3.0, 0.25]))


def test_weighted_dense_fused_adjoint_matches_generic_metric_formula():
    sc = importlib.import_module("spacecore")
    from spacecore.linop._metric import metric_rapply

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    domain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0]))
    )
    codomain = sc.DenseCoordinateSpace(
        (3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([3.0, 7.0, 11.0]))
    )
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    op = sc.DenseLinOp(matrix, domain, codomain, ctx)
    y = ctx.asarray([2.0, -0.5, 1.25])

    assert op._mode.name == "WEIGHTED_FUSED"
    np.testing.assert_allclose(to_numpy(op.rapply(y)), to_numpy(op._weighted_A2H @ y))
    np.testing.assert_allclose(
        to_numpy(op.rapply(y)),
        to_numpy(metric_rapply(op.domain, op.codomain, op._euclidean_rapply_core, y)),
    )


@pytest.mark.parametrize("ctx", list(_contexts(False)))
def test_weighted_composed_sum_scaled_and_adjoint_view_identity(ctx):
    sc = importlib.import_module("spacecore")
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    middle = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    codomain = WeightedVectorSpace([13.0, 17.0], ctx)
    A = sc.DenseLinOp(
        ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]]),
        domain,
        middle,
        ctx,
    )
    B = sc.DenseLinOp(
        ctx.asarray([[2.0, -0.5, 1.0], [-1.5, 0.25, 3.0]]),
        middle,
        codomain,
        ctx,
    )
    C = sc.DenseLinOp(
        ctx.asarray([[-0.25, 2.0], [1.0, 0.5], [2.5, -1.0]]),
        domain,
        middle,
        ctx,
    )
    x = ctx.asarray([0.25, -1.5])
    y = ctx.asarray([2.0, -0.5, 1.25])
    z = ctx.asarray([1.5, -0.75])

    _assert_adjoint_identity(B @ A, x, z)
    _assert_adjoint_identity(A + C, x, y)
    _assert_adjoint_identity(2.5 * A, x, y)

    adjoint = A.H
    _assert_adjoint_identity(adjoint, y, x)
    assert adjoint.H is A


@pytest.mark.parametrize("ctx", list(_contexts(True)))
def test_weighted_complex_scaled_lazily_adjoint_identity(ctx):
    sc = importlib.import_module("spacecore")
    domain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0]))
    )
    codomain = sc.DenseCoordinateSpace(
        (3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([3.0, 7.0, 11.0]))
    )
    op = sc.DenseLinOp(
        ctx.asarray([[1.0 + 0.5j, -2.0j], [0.5 - 1.0j, 3.0], [4.0, -1.0 + 2.0j]]),
        domain,
        codomain,
        ctx,
    )
    x = ctx.asarray([0.25 + 1.0j, -1.5 + 0.5j])
    y = ctx.asarray([2.0 - 0.25j, -0.5 + 1.0j, 1.25j])

    _assert_adjoint_identity((1.25 - 0.5j) * op, x, y)


@pytest.mark.parametrize("ctx", list(_contexts(False)))
def test_weighted_stacked_linop_adjoint_identity(ctx):
    sc = importlib.import_module("spacecore")
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    cod0 = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    cod1 = WeightedVectorSpace([13.0], ctx)
    A0 = sc.DenseLinOp(
        ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]]),
        domain,
        cod0,
        ctx,
    )
    A1 = sc.DenseLinOp(ctx.asarray([[2.0, -0.25]]), domain, cod1, ctx)
    stacked = sc.StackedLinOp.from_operators((A0, A1))
    x = ctx.asarray([0.25, -1.5])
    y = (ctx.asarray([2.0, -0.5, 1.25]), ctx.asarray([-0.75]))

    _assert_adjoint_identity(stacked, x, y)


def test_weighted_matrix_backed_mode_recomputed_after_convert():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float32, enable_checks=False)
    domain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0]))
    )
    codomain = sc.DenseCoordinateSpace(
        (3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([3.0, 7.0, 11.0]))
    )
    op = sc.DenseLinOp(
        ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]]),
        domain,
        codomain,
        ctx,
    )

    converted = op.convert(new_ctx)

    assert op._mode.name == "WEIGHTED_FUSED"
    assert converted._mode.name == "WEIGHTED_FUSED"
    _assert_adjoint_identity(
        converted,
        new_ctx.asarray([0.25, -1.5]),
        new_ctx.asarray([2.0, -0.5, 1.25]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_matrix_free_coordinate_adjoint_constructor_matches_direct_euclidean_adjoint():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    x = ctx.asarray([0.25, -1.5])
    y = ctx.asarray([2.0, -0.5, 1.25])

    direct = sc.MatrixFreeLinOp(lambda z: matrix @ z, lambda w: matrix.T @ w, domain, codomain, ctx)
    coordinate = sc.MatrixFreeLinOp.from_coordinate_adjoint(
        lambda z: matrix @ z, lambda w: matrix.T @ w, domain, codomain, ctx
    )

    np.testing.assert_allclose(to_numpy(coordinate.apply(x)), to_numpy(direct.apply(x)))
    np.testing.assert_allclose(to_numpy(coordinate.rapply(y)), to_numpy(direct.rapply(y)))
    _assert_adjoint_identity(coordinate, x, y)


def test_matrix_free_coordinate_adjoint_constructor_corrects_weighted_spaces():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    codomain = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    x = ctx.asarray([0.25, -1.5])
    y = ctx.asarray([2.0, -0.5, 1.25])

    direct_coordinate = sc.MatrixFreeLinOp(
        lambda z: matrix @ z,
        lambda w: matrix.T @ w,
        domain,
        codomain,
        ctx,
    )
    lhs = codomain.inner(direct_coordinate.apply(x), y)
    rhs = domain.inner(x, direct_coordinate.rapply(y))
    assert not np.allclose(to_numpy(lhs), to_numpy(rhs))

    corrected = sc.MatrixFreeLinOp.from_coordinate_adjoint(
        lambda z: matrix @ z,
        lambda w: matrix.T @ w,
        domain,
        codomain,
        ctx,
    )
    _assert_adjoint_identity(corrected, x, y)


def test_matrix_free_coordinate_adjoint_batched_rvapply_matches_loop():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    codomain = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    ys = ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]])

    op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
        lambda z: matrix @ z,
        lambda w: matrix.T @ w,
        domain,
        codomain,
        ctx,
        coordinate_rvapply=lambda ws: ws @ matrix,
    )
    expected = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)

    np.testing.assert_allclose(to_numpy(op.rvapply(ys)), expected)


def test_matrix_free_coordinate_adjoint_supports_weighted_product_space():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    WeightedVectorSpace = _weighted_space_class()
    domain = sc.ProductSpace(
        (WeightedVectorSpace([2.0, 5.0], ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx
    )
    codomain = sc.ProductSpace(
        (sc.DenseCoordinateSpace((1,), ctx), WeightedVectorSpace([3.0, 7.0], ctx)), ctx
    )
    matrix = ctx.asarray([[1.0, 2.0, -1.0], [0.5, 3.0, 2.0], [4.0, -1.0, 0.25]])

    def apply(x):
        return codomain.unflatten(matrix @ domain.flatten(x))

    def coordinate_rapply(y):
        return domain.unflatten(matrix.T @ codomain.flatten(y))

    op = sc.MatrixFreeLinOp.from_coordinate_adjoint(apply, coordinate_rapply, domain, codomain, ctx)
    x = (ctx.asarray([0.25, -1.5]), ctx.asarray([2.0]))
    y = (ctx.asarray([-0.5]), ctx.asarray([2.0, 1.25]))

    _assert_adjoint_identity(op, x, y)


def test_matrix_free_coordinate_adjoint_conversion_uses_converted_riesz_maps():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)

    class CountingWeightedInnerProduct(sc.InnerProduct):
        def __init__(self, weights, counter, converted_counter):
            self.weights = weights
            self.counter = counter
            self.converted_counter = converted_counter

        def inner(self, ops, x, y):
            return ops.vdot(x, self.weights * y)

        def riesz(self, ops, x):
            self.counter["riesz"] += 1
            return self.weights * x

        def riesz_inverse(self, ops, x):
            self.counter["inverse"] += 1
            return x / self.weights

        def convert(self, new_ctx):
            return CountingWeightedInnerProduct(
                new_ctx.asarray(self.weights), self.converted_counter, self.converted_counter
            )

        @property
        def is_euclidean(self):
            return False

    old_dom = {"riesz": 0, "inverse": 0}
    old_cod = {"riesz": 0, "inverse": 0}
    new_dom = {"riesz": 0, "inverse": 0}
    new_cod = {"riesz": 0, "inverse": 0}
    domain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=CountingWeightedInnerProduct(ctx.asarray([2.0, 5.0]), old_dom, new_dom)
    )
    codomain = sc.DenseCoordinateSpace(
        (3,),
        ctx,
        geometry=CountingWeightedInnerProduct(ctx.asarray([3.0, 7.0, 11.0]), old_cod, new_cod),
    )
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
        lambda z: matrix @ z,
        lambda w: matrix.T @ w,
        domain,
        codomain,
        ctx,
    )
    y = ctx.asarray([2.0, -0.5, 1.25])

    op.rapply(y)
    assert old_cod["riesz"] == 1
    assert old_dom["inverse"] == 1

    converted = op.convert(new_ctx)
    converted.rapply(new_ctx.asarray(y))
    assert new_cod["riesz"] == 1
    assert new_dom["inverse"] == 1
    assert old_cod["riesz"] == 1
    assert old_dom["inverse"] == 1

    _assert_adjoint_identity(
        converted,
        new_ctx.asarray([0.25, -1.5]),
        new_ctx.asarray([2.0, -0.5, 1.25]),
    )


def test_matrix_free_coordinate_rvapply_is_preserved_under_conversion():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    codomain = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    calls = {"coordinate_rvapply": 0}

    def coordinate_rvapply(ws):
        calls["coordinate_rvapply"] += 1
        return ws @ matrix

    op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
        lambda z: matrix @ z,
        lambda w: matrix.T @ w,
        domain,
        codomain,
        ctx,
        coordinate_rvapply=coordinate_rvapply,
    )
    converted = op.convert(new_ctx)
    ys = new_ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]])
    expected = np.stack([to_numpy(converted.rapply(y)) for y in ys], axis=0)

    np.testing.assert_allclose(to_numpy(converted.rvapply(ys)), expected)
    assert calls["coordinate_rvapply"] == 1


def test_matrix_free_coordinate_adjoint_conversion_without_coordinate_rvapply_uses_fallback():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    codomain = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
        lambda z: matrix @ z,
        lambda w: matrix.T @ w,
        domain,
        codomain,
        ctx,
    )
    converted = op.convert(new_ctx)
    ys = new_ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]])
    expected = np.stack([to_numpy(converted.rapply(y)) for y in ys], axis=0)

    assert converted.rvapply_fn is None
    np.testing.assert_allclose(to_numpy(converted.rvapply(ys)), expected)


def test_direct_matrix_free_conversion_remains_backward_compatible():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    new_ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])

    def apply(z):
        return matrix @ z

    def rapply(w):
        return matrix.T @ w

    op = sc.MatrixFreeLinOp(apply, rapply, domain, codomain, ctx)
    converted = op.convert(new_ctx)
    y = new_ctx.asarray([2.0, -0.5, 1.25])

    assert converted.rapply_fn is rapply
    assert converted._uses_coordinate_adjoint is False
    np.testing.assert_allclose(to_numpy(converted.rapply(y)), to_numpy(rapply(y)))


def test_matrix_free_internal_coordinate_metadata_invariants():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    domain = sc.DenseCoordinateSpace((2,), ctx)

    with pytest.raises(ValueError, match="_coordinate_rapply_fn"):
        sc.MatrixFreeLinOp(
            lambda x: x,
            lambda y: y,
            domain,
            domain,
            ctx,
            _uses_coordinate_adjoint=True,
        )

    with pytest.raises(ValueError, match="direct-adjoint"):
        sc.MatrixFreeLinOp(
            lambda x: x,
            lambda y: y,
            domain,
            domain,
            ctx,
            _coordinate_rapply_fn=lambda y: y,
        )


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_direct_matrix_free_pytree_remains_backward_compatible():
    import jax

    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    domain = sc.DenseCoordinateSpace((2,), ctx)
    codomain = sc.DenseCoordinateSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])

    def apply(z):
        return matrix @ z

    def rapply(w):
        return matrix.T @ w

    op = sc.MatrixFreeLinOp(apply, rapply, domain, codomain, ctx)
    leaves, treedef = jax.tree.flatten(op)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert rebuilt == op
    assert rebuilt._uses_coordinate_adjoint is False
    np.testing.assert_allclose(
        to_numpy(rebuilt.rapply(ctx.asarray([2.0, -0.5, 1.25]))),
        to_numpy(op.rapply(ctx.asarray([2.0, -0.5, 1.25]))),
    )


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_coordinate_adjoint_matrix_free_pytree_preserves_construction_mode():
    import jax

    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    codomain = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    matrix = ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    op = sc.MatrixFreeLinOp.from_coordinate_adjoint(
        lambda z: matrix @ z,
        lambda w: matrix.T @ w,
        domain,
        codomain,
        ctx,
    )

    leaves, treedef = jax.tree.flatten(op)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert rebuilt == op
    assert rebuilt._uses_coordinate_adjoint is True
    _assert_adjoint_identity(
        rebuilt,
        ctx.asarray([0.25, -1.5]),
        ctx.asarray([2.0, -0.5, 1.25]),
    )


def test_method_based_riesz_space_is_accepted():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    class MethodInnerProduct(sc.InnerProduct):
        def inner(self, ops, x, y):
            return ops.vdot(x, ctx.asarray([2.0, 5.0]) * y)

    class MethodRieszSpace(sc.DenseCoordinateSpace):
        def __init__(self, ctx):
            super().__init__((2,), ctx, geometry=MethodInnerProduct())
            self.weights = ctx.asarray([2.0, 5.0])

        def riesz(self, x):
            return self.weights * x

        def riesz_inverse(self, x):
            return x / self.weights

    space = MethodRieszSpace(ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, -1.0]]), space, space, ctx)

    _assert_adjoint_identity(op, ctx.asarray([0.25, -1.5]), ctx.asarray([2.0, -0.5]))


@pytest.mark.parametrize("ctx", list(_contexts(False)))
def test_weighted_batched_apply_and_adjoint_match_loops(ctx):
    sc = importlib.import_module("spacecore")
    WeightedVectorSpace = _weighted_space_class()
    domain = WeightedVectorSpace([2.0, 5.0], ctx)
    codomain = WeightedVectorSpace([3.0, 7.0, 11.0], ctx)
    matrix = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    xs = ctx.asarray([[0.25, -1.5], [2.0, 0.5]])
    ys = ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]])

    dense = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    sparse = sc.SparseLinOp(ctx.assparse(matrix), domain, codomain, ctx)

    _assert_vapply_loop(dense, xs)
    _assert_rvapply_loop(dense, ys)
    _assert_vapply_loop(sparse, xs)
    _assert_rvapply_loop(sparse, ys)


def test_weighted_batched_adjoint_uses_broadcast_riesz_without_fallback_warning():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    domain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0]))
    )
    codomain = sc.DenseCoordinateSpace(
        (3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([3.0, 7.0, 11.0]))
    )
    matrix = np.array([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]])
    op = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    ys = ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        actual = op.rvapply(ys)

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    expected = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)
    np.testing.assert_allclose(to_numpy(actual), expected)


def test_product_space_batched_forward_uses_space_batch_helpers():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    domain = sc.ProductSpace(
        (sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx
    )
    codomain = sc.ProductSpace(
        (sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((2,), ctx)), ctx
    )
    matrix = np.array([[1.0, 2.0, -1.0], [0.5, 3.0, 2.0], [4.0, -1.0, 0.25]])
    xs = (ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]), ctx.asarray([[3.0], [4.0]]))

    dense = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)

    sparse = sc.SparseLinOp(ctx.assparse(matrix), domain, codomain, ctx)

    _assert_vapply_loop(dense, xs)
    _assert_vapply_loop(sparse, xs)


def test_matrix_backed_ops_accept_euclidean_product_space():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    space = sc.ProductSpace(
        (sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx
    )
    matrix = ctx.asarray(np.eye(3))
    diagonal = ctx.asarray([2.0, -1.0, 0.5])
    x = (ctx.asarray([1.0, -2.0]), ctx.asarray([0.5]))
    y = (ctx.asarray([3.0, 0.25]), ctx.asarray([-1.5]))

    dense = sc.DenseLinOp(matrix, space, space, ctx)
    diagonal_op = sc.DiagonalLinOp(diagonal, space, ctx)

    sparse = sc.SparseLinOp(ctx.assparse(np.eye(3)), space, space, ctx)

    _assert_adjoint_identity(dense, x, y)
    _assert_adjoint_identity(diagonal_op, x, y)
    _assert_adjoint_identity(sparse, x, y)


def test_matrix_backed_ops_accept_weighted_product_space_and_satisfy_adjoint_identity():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    WeightedVectorSpace = _weighted_space_class()
    domain = sc.ProductSpace(
        (WeightedVectorSpace([2.0, 5.0], ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx
    )
    codomain = sc.ProductSpace(
        (sc.DenseCoordinateSpace((1,), ctx), WeightedVectorSpace([3.0, 7.0], ctx)), ctx
    )
    matrix = np.array([[1.0, 2.0, -1.0], [0.5, 3.0, 2.0], [4.0, -1.0, 0.25]])
    x = (ctx.asarray([0.25, -1.5]), ctx.asarray([2.0]))
    y = (ctx.asarray([-0.5]), ctx.asarray([2.0, 1.25]))

    dense = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)

    sparse = sc.SparseLinOp(ctx.assparse(matrix), domain, codomain, ctx)

    _assert_adjoint_identity(dense, x, y)
    _assert_adjoint_identity(sparse, x, y)

    diagonal_space = sc.ProductSpace(
        (WeightedVectorSpace([11.0, 13.0], ctx), sc.DenseCoordinateSpace((1,), ctx)), ctx
    )
    diagonal = sc.DiagonalLinOp(ctx.asarray([2.0, -1.0, 0.5]), diagonal_space, ctx)
    z = (ctx.asarray([1.0, -2.0]), ctx.asarray([0.75]))
    w = (ctx.asarray([-0.5, 3.0]), ctx.asarray([1.25]))
    _assert_adjoint_identity(diagonal, z, w)


def test_metric_rvapply_warns_and_falls_back_when_batched_riesz_unavailable():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    class NoBatchWeightedInnerProduct(sc.InnerProduct):
        def __init__(self, weights):
            self.weights = weights

        def inner(self, ops, x, y):
            return ops.vdot(x, self.weights * y)

        def riesz(self, ops, x):
            if len(tuple(getattr(x, "shape", ()))) != 1:
                raise NotImplementedError("batched Riesz is intentionally unavailable")
            return self.weights * x

        def riesz_inverse(self, ops, x):
            if len(tuple(getattr(x, "shape", ()))) != 1:
                raise NotImplementedError("batched inverse Riesz is intentionally unavailable")
            return x / self.weights

        @property
        def is_euclidean(self):
            return False

    domain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=NoBatchWeightedInnerProduct(ctx.asarray([2.0, 5.0]))
    )
    codomain = sc.DenseCoordinateSpace(
        (3,), ctx, geometry=NoBatchWeightedInnerProduct(ctx.asarray([3.0, 7.0, 11.0]))
    )
    op = sc.DenseLinOp(
        ctx.asarray([[1.0, -2.0], [0.5, 3.0], [4.0, -1.0]]),
        domain,
        codomain,
        ctx,
    )
    ys = ctx.asarray([[2.0, -0.5, 1.25], [-1.0, 3.0, 0.75]])

    with pytest.warns(RuntimeWarning, match="falling back to vmap"):
        actual = op.rvapply(ys)
    expected = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)
    np.testing.assert_allclose(to_numpy(actual), expected)


def test_non_euclidean_space_without_riesz_maps_is_rejected():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    class BrokenInnerProduct(sc.InnerProduct):
        def inner(self, ops, x, y):
            return ops.vdot(x, 2.0 * y)

    class BrokenSpace(sc.DenseCoordinateSpace):
        def __init__(self, shape, ctx):
            super().__init__(shape, ctx)
            self.geometry = BrokenInnerProduct()

    space = BrokenSpace((2,), ctx)

    with pytest.raises(TypeError, match="MatrixFreeLinOp"):
        sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), space, space, ctx)
    with pytest.raises(TypeError, match="riesz/riesz_inverse"):
        sc.DiagonalLinOp(ctx.asarray([1.0, 2.0]), space, ctx)


def test_product_space_with_component_missing_riesz_maps_is_rejected():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    class BrokenInnerProduct(sc.InnerProduct):
        def inner(self, ops, x, y):
            return ops.vdot(x, 2.0 * y)

    broken = sc.DenseCoordinateSpace((2,), ctx, geometry=BrokenInnerProduct())
    euclidean = sc.DenseCoordinateSpace((1,), ctx)
    product = sc.ProductSpace((broken, euclidean), ctx)

    with pytest.raises(TypeError, match="Riesz maps"):
        sc.DenseLinOp(ctx.asarray(np.eye(3)), product, product, ctx)
    with pytest.raises(TypeError, match="Riesz maps"):
        sc.SparseLinOp(ctx.assparse(np.eye(3)), product, product, ctx)
    with pytest.raises(TypeError, match="Riesz maps"):
        sc.DiagonalLinOp(ctx.asarray([1.0, 2.0, 3.0]), product, ctx)


def test_metric_hermitian_detection_uses_weighted_adjoint():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    WeightedVectorSpace = _weighted_space_class()
    space = WeightedVectorSpace([2.0, 3.0], ctx)

    weighted_self_adjoint = sc.DenseLinOp(ctx.asarray([[1.0, 3.0], [2.0, 4.0]]), space, space, ctx)
    coordinate_symmetric_only = sc.DenseLinOp(
        ctx.asarray([[1.0, 1.0], [1.0, 4.0]]), space, space, ctx
    )

    assert weighted_self_adjoint.is_hermitian() is True
    assert coordinate_symmetric_only.is_hermitian() is False


def test_large_metric_hermitian_check_returns_unknown_without_applying_operator():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    n = 1025
    space = sc.DenseCoordinateSpace(
        (n,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray(np.ones(n)))
    )
    op = sc.DiagonalLinOp(ctx.asarray(np.ones(n)), space, ctx)

    def fail_apply(_x):
        raise AssertionError("is_hermitian should not apply large metric operators")

    op.apply = fail_apply

    assert op.is_hermitian() is None
