import importlib

import numpy as np
import pytest

from tests._helpers import has_jax, has_torch, jax_complex_dtype, jax_real_dtype
from tests._helpers import to_numpy, torch_complex_dtype


def _backend_params():
    params = [pytest.param("numpy", np.complex128, id="numpy")]
    params.append(
        pytest.param(
            "jax",
            jax_complex_dtype(),
            marks=pytest.mark.skipif(not has_jax(), reason="jax is not installed"),
            id="jax",
        )
    )
    params.append(
        pytest.param(
            "torch",
            torch_complex_dtype(),
            marks=pytest.mark.skipif(not has_torch(), reason="torch is not installed"),
            id="torch",
        )
    )
    return params


def _ops_for_backend(name):
    sc = importlib.import_module("spacecore")
    if name == "numpy":
        return sc.NumpyOps()
    if name == "jax":
        return sc.JaxOps()
    if name == "torch":
        return sc.TorchOps()
    raise ValueError(f"Unknown backend {name!r}.")


def _ctx(dtype=np.complex128, enable_checks=True):
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=enable_checks)


def _spaces(ctx):
    sc = importlib.import_module("spacecore")
    return sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((3,), ctx)


def _matrix():
    return np.array(
        [
            [1.0 + 2.0j, 3.0 - 1.0j],
            [-2.0 + 0.5j, 0.25 + 4.0j],
            [1.5 - 3.0j, -0.75 + 2.0j],
        ]
    )


def _square_matrix():
    return np.array([[2.0 - 1.0j, -0.5 + 0.25j], [1.25 + 2.0j, -3.0 + 0.5j]])


def _dense_linop(ctx):
    sc = importlib.import_module("spacecore")
    dom, cod = _spaces(ctx)
    return sc.DenseLinOp(ctx.asarray(_matrix()), dom, cod, ctx)


def _dense_same_shape(ctx, scale=1.0):
    sc = importlib.import_module("spacecore")
    dom, cod = _spaces(ctx)
    return sc.DenseLinOp(ctx.asarray(scale * _matrix()), dom, cod, ctx)


def _dense_square(ctx):
    sc = importlib.import_module("spacecore")
    dom = sc.DenseCoordinateSpace((2,), ctx)
    return sc.DenseLinOp(ctx.asarray(_square_matrix()), dom, dom, ctx)


def _xy(ctx):
    x = ctx.asarray([2.0 - 1.0j, -0.5 + 0.25j])
    y = ctx.asarray([1.0 + 0.5j, -2.0j, 0.75 - 1.25j])
    return x, y


def _assert_adjoint_identity(op, x, y, ctx):
    lhs = ctx.ops.vdot(op.apply(x), y)
    rhs = ctx.ops.vdot(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)


def _slice_batch(value, i):
    if isinstance(value, tuple):
        return tuple(_slice_batch(part, i) for part in value)
    return value[i]


def _stack_rows(rows):
    if isinstance(rows[0], tuple):
        return tuple(_stack_rows(tuple(row[i] for row in rows)) for i in range(len(rows[0])))
    return np.stack([to_numpy(row) for row in rows], axis=0)


def _assert_nested_allclose(actual, expected, *, rtol=1e-7, atol=1e-7):
    actual_np = to_numpy(actual)
    expected_np = to_numpy(expected)
    if isinstance(actual_np, tuple):
        assert isinstance(expected_np, tuple)
        assert len(actual_np) == len(expected_np)
        for a, e in zip(actual_np, expected_np):
            _assert_nested_allclose(a, e, rtol=rtol, atol=atol)
        return
    np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol)


def _adjoint_cases(ctx):
    sc = importlib.import_module("spacecore")
    A = _dense_linop(ctx)
    B = _dense_same_shape(ctx, scale=0.5 - 0.25j)
    C = _dense_square(ctx)
    dom, cod = _spaces(ctx)
    x, y = _xy(ctx)
    z = ctx.asarray([-1.0 + 0.5j, 2.0 - 0.25j])

    matrix = ctx.asarray(_matrix())
    matrix_free = sc.MatrixFreeLinOp(
        lambda v: matrix @ v,
        lambda w: ctx.ops.conj(ctx.ops.transpose(matrix)) @ w,
        dom,
        cod,
        ctx,
    )

    return [
        ((2.0 + 3.0j) * A, x, y),
        (A + B, x, y),
        (A @ C, z, y),
        (sc.ZeroLinOp(dom, cod, ctx), x, y),
        (sc.IdentityLinOp(dom, ctx), x, x),
        (matrix_free, x, y),
        (A.H, y, x),
    ]


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
@pytest.mark.parametrize("case_index", range(7))
def test_complex_adjoint_identity_for_algebra_classes(backend_name, dtype, case_index):
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(_ops_for_backend(backend_name), dtype=dtype)
    op, x, y = _adjoint_cases(ctx)[case_index]

    _assert_adjoint_identity(op, x, y, ctx)


def test_simplification_canonicalizations():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    A = _dense_linop(ctx)
    B = _dense_same_shape(ctx, scale=2.0)
    C = _dense_same_shape(ctx, scale=-1.0)
    Z = sc.ZeroLinOp(A.domain, A.codomain, ctx)

    assert sc.make_sum((A, Z)) is A
    assert isinstance(sc.make_sum((Z, Z)), sc.ZeroLinOp)
    assert sc.make_sum((A,)) is A
    flattened = sc.make_sum((sc.make_sum((A, B)), C))
    assert isinstance(flattened, sc.SumLinOp)
    assert flattened.parts == (A, B, C)

    scaled_zero = sc.make_scaled(0, A)
    assert isinstance(scaled_zero, sc.ZeroLinOp)
    assert scaled_zero.domain == A.domain
    assert scaled_zero.codomain == A.codomain
    assert sc.make_scaled(1, A) is A
    assert sc.make_scaled(7.0, Z) is Z
    folded = sc.make_scaled(2, sc.make_scaled(3, A))
    assert isinstance(folded, sc.ScaledLinOp)
    assert folded.scalar == 6
    assert folded.op is A

    I_dom = sc.IdentityLinOp(A.domain, ctx)
    I_cod = sc.IdentityLinOp(A.codomain, ctx)
    assert sc.make_composed(I_cod, A) is A
    assert sc.make_composed(A, I_dom) is A

    out = sc.DenseCoordinateSpace((4,), ctx)
    left_zero = sc.ZeroLinOp(A.codomain, out, ctx)
    composed_zero = sc.make_composed(left_zero, A)
    assert isinstance(composed_zero, sc.ZeroLinOp)
    assert composed_zero.domain == A.domain
    assert composed_zero.codomain == out


@pytest.mark.parametrize("case_index", range(7))
def test_double_adjoint_view_returns_literal_original(case_index):
    ctx = _ctx()
    op, _, _ = _adjoint_cases(ctx)[case_index]

    assert op.H.H is op


def test_identity_linop_apply_is_literal_input_when_checks_disabled():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(enable_checks=False)
    space = sc.DenseCoordinateSpace((2,), ctx)
    op = sc.IdentityLinOp(space, ctx)
    x = ctx.asarray([1.0 + 2.0j, 3.0 - 4.0j])

    assert op.apply(x) is x
    assert op.rapply(x) is x


def test_identity_linop_apply_equals_input_when_checks_enabled():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(enable_checks=True)
    space = sc.DenseCoordinateSpace((2,), ctx)
    op = sc.IdentityLinOp(space, ctx)
    x = ctx.asarray([1.0 + 2.0j, 3.0 - 4.0j])

    np.testing.assert_allclose(op.apply(x), x)
    np.testing.assert_allclose(op.rapply(x), x)


def test_python_sum_starts_from_zero_and_accumulates_linops():
    ctx = _ctx()
    A = _dense_same_shape(ctx, scale=1.0)
    B = _dense_same_shape(ctx, scale=0.5)
    C = _dense_same_shape(ctx, scale=-2.0)
    x, _ = _xy(ctx)

    op = sum([A, B, C])
    expected = A.apply(x) + B.apply(x) + C.apply(x)

    np.testing.assert_allclose(op.apply(x), expected)


def test_scaled_linop_uses_space_scale_for_vectors_and_complex_adjoint():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.complex128)
    A = _dense_linop(ctx)
    x, y = _xy(ctx)
    alpha = 2.0 + 3.0j
    op = alpha * A

    np.testing.assert_allclose(to_numpy(op.apply(x)), to_numpy(alpha * A.apply(x)))
    np.testing.assert_allclose(to_numpy(op.rapply(y)), to_numpy(np.conj(alpha) * A.rapply(y)))
    _assert_adjoint_identity(op, x, y, ctx)

    xs = ctx.asarray(np.stack([to_numpy(x), to_numpy(0.5 * x)], axis=0))
    ys = ctx.asarray(np.stack([to_numpy(y), to_numpy(-2.0 * y)], axis=0))
    np.testing.assert_allclose(to_numpy(op.vapply(xs)), to_numpy(alpha * A.vapply(xs)))
    np.testing.assert_allclose(to_numpy(op.rvapply(ys)), to_numpy(np.conj(alpha) * A.rvapply(ys)))


def test_scaled_linop_supports_product_space_elements_and_batches():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)
    x1, x2 = sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)
    y1, y2 = sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((2,), ctx)
    base = sc.BlockDiagonalLinOp.from_operators(
        (
            sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x1, y1, ctx),
            sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x2, y2, ctx),
        )
    )
    op = 2.5 * base
    x = (ctx.asarray([1.0, 2.0]), ctx.asarray([3.0]))
    y = (ctx.asarray([5.0]), ctx.asarray([1.0, 2.0]))
    xs = (ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]), ctx.asarray([[3.0], [4.0]]))
    ys = (ctx.asarray([[5.0], [-2.0]]), ctx.asarray([[1.0, 2.0], [0.5, -1.0]]))

    _assert_nested_allclose(op.apply(x), base.codomain.scale(2.5, base.apply(x)))
    _assert_nested_allclose(op.rapply(y), base.domain.scale(2.5, base.rapply(y)))

    expected_v = _stack_rows(tuple(op.apply(_slice_batch(xs, i)) for i in range(2)))
    expected_rv = _stack_rows(tuple(op.rapply(_slice_batch(ys, i)) for i in range(2)))
    _assert_nested_allclose(op.vapply(xs), expected_v)
    _assert_nested_allclose(op.rvapply(ys), expected_rv)


def test_scaled_linop_batched_paths_use_space_scale_batch():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)

    class CountingVectorSpace(sc.DenseCoordinateSpace):
        def __init__(self, shape, ctx, counter):
            self.counter = counter
            super().__init__(shape, ctx)

        def scale(self, a, x):
            self.counter["scale"] += 1
            return super().scale(a, x)

        def scale_batch(self, a, x):
            self.counter["scale_batch"] += 1
            return super().scale_batch(a, x)

        def _convert(self, new_ctx):
            return CountingVectorSpace(self.shape, new_ctx, self.counter)

    domain_counter = {"scale": 0, "scale_batch": 0}
    codomain_counter = {"scale": 0, "scale_batch": 0}
    X = CountingVectorSpace((2,), ctx, domain_counter)
    Y = CountingVectorSpace((2,), ctx, codomain_counter)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, -1.0]]), X, Y, ctx)
    op = 3.0 * A
    x = ctx.asarray([1.0, -2.0])
    y = ctx.asarray([0.5, 4.0])
    xs = ctx.asarray([[1.0, -2.0], [0.5, 4.0]])
    ys = ctx.asarray([[0.5, 4.0], [3.0, -1.0]])

    op.apply(x)
    assert codomain_counter["scale"] == 1
    assert domain_counter["scale"] == 0

    op.rapply(y)
    assert domain_counter["scale"] == 1

    op.vapply(xs)
    assert codomain_counter["scale_batch"] == 1

    op.rvapply(ys)
    assert domain_counter["scale_batch"] == 1


def test_sum_linop_vapply_and_rvapply_match_scalar_loop_for_dense_vectors():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y = sc.DenseCoordinateSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]]), X, Y, ctx)
    B = sc.DenseLinOp(ctx.asarray([[0.25, -2.0], [1.5, 0.5], [-1.0, 3.0]]), X, Y, ctx)
    op = A + B
    xs = ctx.asarray([[1.0, -2.0], [0.5, 4.0], [-3.0, 1.0]])
    ys = ctx.asarray([[2.0, -1.0, 0.5], [4.0, 3.0, -2.0], [0.25, -0.5, 1.5]])

    expected_v = np.stack([to_numpy(op.apply(x)) for x in xs], axis=0)
    expected_rv = np.stack([to_numpy(op.rapply(y)) for y in ys], axis=0)

    np.testing.assert_allclose(to_numpy(op.vapply(xs)), expected_v)
    np.testing.assert_allclose(to_numpy(op.rvapply(ys)), expected_rv)


def test_sum_linop_vapply_and_rvapply_work_for_product_space_batches():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)
    x1, x2 = sc.DenseCoordinateSpace((2,), ctx), sc.DenseCoordinateSpace((1,), ctx)
    y1, y2 = sc.DenseCoordinateSpace((1,), ctx), sc.DenseCoordinateSpace((2,), ctx)
    A = sc.BlockDiagonalLinOp.from_operators(
        (
            sc.DenseLinOp(ctx.asarray([[1.0, 2.0]]), x1, y1, ctx),
            sc.DenseLinOp(ctx.asarray([[3.0], [-1.0]]), x2, y2, ctx),
        )
    )
    B = sc.BlockDiagonalLinOp.from_operators(
        (
            sc.DenseLinOp(ctx.asarray([[0.5, -4.0]]), x1, y1, ctx),
            sc.DenseLinOp(ctx.asarray([[2.0], [5.0]]), x2, y2, ctx),
        )
    )
    op = A + B
    xs = (ctx.asarray([[1.0, 2.0], [-1.0, 0.5]]), ctx.asarray([[3.0], [4.0]]))
    ys = (ctx.asarray([[5.0], [-2.0]]), ctx.asarray([[1.0, 2.0], [0.5, -1.0]]))

    expected_v = _stack_rows(tuple(op.apply(_slice_batch(xs, i)) for i in range(2)))
    expected_rv = _stack_rows(tuple(op.rapply(_slice_batch(ys, i)) for i in range(2)))

    _assert_nested_allclose(op.vapply(xs), expected_v)
    _assert_nested_allclose(op.rvapply(ys), expected_rv)


def test_sum_linop_batched_accumulation_uses_space_add_batch():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)

    class CountingVectorSpace(sc.DenseCoordinateSpace):
        def __init__(self, shape, ctx, counter):
            self.counter = counter
            super().__init__(shape, ctx)

        def add_batch(self, x, y):
            self.counter["calls"] += 1
            return super().add_batch(x, y)

        def _convert(self, new_ctx):
            return CountingVectorSpace(self.shape, new_ctx, self.counter)

    domain_counter = {"calls": 0}
    codomain_counter = {"calls": 0}
    X = CountingVectorSpace((2,), ctx, domain_counter)
    Y = CountingVectorSpace((2,), ctx, codomain_counter)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, -1.0]]), X, Y, ctx)
    B = sc.DenseLinOp(ctx.asarray([[0.5, -4.0], [2.0, 5.0]]), X, Y, ctx)
    op = A + B
    xs = ctx.asarray([[1.0, 2.0], [-1.0, 0.5]])
    ys = ctx.asarray([[5.0, -2.0], [0.5, -1.0]])

    op.vapply(xs)
    assert codomain_counter["calls"] == 1
    assert domain_counter["calls"] == 0

    op.rvapply(ys)
    assert domain_counter["calls"] == 1


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
@pytest.mark.parametrize("case_index", range(7))
def test_jax_pytree_roundtrip_for_algebra_classes(case_index):
    import jax

    ctx = _ctx()
    op, _, _ = _adjoint_cases(ctx)[case_index]
    leaves, treedef = jax.tree.flatten(op)
    rebuilt = jax.tree.unflatten(treedef, leaves)

    assert rebuilt == op


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_jax_jit_algebra_expression_matches_eager():
    import jax

    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.JaxOps(), dtype=jax_real_dtype(), enable_checks=False)
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y = sc.DenseCoordinateSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, ctx)
    B = sc.DenseLinOp(ctx.asarray([[0.5, -1.0], [2.0, 1.0], [-0.5, 3.0]]), X, Y, ctx)
    C = sc.DenseLinOp(ctx.asarray([[2.0, -1.0], [0.25, 1.5]]), X, X, ctx)
    expr = (2 * A + B) @ C
    x = ctx.asarray([1.0, -2.0])

    apply_jit = jax.jit(lambda op, z: op.apply(z))

    np.testing.assert_allclose(to_numpy(apply_jit(expr, x)), to_numpy(expr.apply(x)))


def test_factories_enforce_same_context_dtype():
    sc = importlib.import_module("spacecore")
    ctx32 = sc.Context(sc.NumpyOps(), dtype=np.float32)
    ctx64 = sc.Context(sc.NumpyOps(), dtype=np.float64)
    X32 = sc.DenseCoordinateSpace((2,), ctx32)
    Y32 = sc.DenseCoordinateSpace((2,), ctx32)
    X64 = sc.DenseCoordinateSpace((2,), ctx64)
    Y64 = sc.DenseCoordinateSpace((2,), ctx64)
    A32 = sc.DenseLinOp(ctx32.asarray([[1.0, 2.0], [3.0, 4.0]]), X32, Y32, ctx32)
    A64 = sc.DenseLinOp(ctx64.asarray([[1.0, 2.0], [3.0, 4.0]]), X64, Y64, ctx64)

    with pytest.raises(ValueError, match="same ctx"):
        sc.make_sum((A32, A64))
    with pytest.raises(ValueError, match="same ctx"):
        sc.make_composed(A32, A64)


def test_factories_ignore_enable_checks_when_context_dtype_matches():
    sc = importlib.import_module("spacecore")
    checked = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
    unchecked = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=False)
    X_checked = sc.DenseCoordinateSpace((2,), checked)
    X_unchecked = sc.DenseCoordinateSpace((2,), unchecked)
    A = sc.DenseLinOp(checked.asarray([[1.0, 0.0], [0.0, 1.0]]), X_checked, X_checked, checked)
    B = sc.DenseLinOp(
        unchecked.asarray([[2.0, 0.0], [0.0, 3.0]]),
        X_unchecked,
        X_unchecked,
        unchecked,
    )

    summed = sc.make_sum((A, B))
    composed = sc.make_composed(A, B)

    assert isinstance(summed, sc.SumLinOp)
    assert isinstance(composed, sc.ComposedLinOp)


def test_factories_reject_matching_shapes_with_different_geometry():
    sc = importlib.import_module("spacecore")
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)

    euclidean = sc.DenseCoordinateSpace((2,), ctx)
    weighted_a = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 3.0])))
    weighted_b = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 3.0])))
    differently_weighted = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 4.0])))

    A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0]]), euclidean, euclidean, ctx)
    B = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), weighted_a, weighted_a, ctx)
    C = sc.DenseLinOp(ctx.asarray([[4.0, 0.0], [0.0, 5.0]]), differently_weighted, differently_weighted, ctx)
    B_same_geometry = sc.DenseLinOp(
        ctx.asarray([[0.5, 0.0], [0.0, 0.25]]), weighted_b, weighted_b, ctx
    )

    with pytest.raises(ValueError, match="same domain and codomain"):
        sc.make_sum((A, B))
    with pytest.raises(ValueError, match="right.codomain == left.domain"):
        sc.make_composed(A, B)
    with pytest.raises(ValueError, match="same domain and codomain"):
        sc.make_sum((B, C))

    assert isinstance(sc.make_sum((B, B_same_geometry)), sc.SumLinOp)
    assert isinstance(sc.make_composed(B, B_same_geometry), sc.ComposedLinOp)


def test_factories_enforce_domain_and_codomain_compatibility():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y = sc.DenseCoordinateSpace((3,), ctx)
    Z = sc.DenseCoordinateSpace((4,), ctx)
    A = sc.DenseLinOp(ctx.asarray(np.ones((3, 2))), X, Y, ctx)
    B = sc.DenseLinOp(ctx.asarray(np.ones((4, 2))), X, Z, ctx)

    with pytest.raises(ValueError, match="same domain and codomain"):
        sc.make_sum((A, B))
    with pytest.raises(ValueError, match="right.codomain == left.domain"):
        sc.make_composed(A, B)


def test_base_linop_equality_protocol_does_not_raise():
    A = _dense_linop(_ctx())

    assert (A == None) is False  # noqa: E711
    assert A in [A]
