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
    return sc.VectorSpace((2,), ctx), sc.VectorSpace((3,), ctx)


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
    dom = sc.VectorSpace((2,), ctx)
    return sc.DenseLinOp(ctx.asarray(_square_matrix()), dom, dom, ctx)


def _xy(ctx):
    x = ctx.asarray([2.0 - 1.0j, -0.5 + 0.25j])
    y = ctx.asarray([1.0 + 0.5j, -2.0j, 0.75 - 1.25j])
    return x, y


def _assert_adjoint_identity(op, x, y, ctx):
    lhs = ctx.ops.vdot(op.apply(x), y)
    rhs = ctx.ops.vdot(x, op.rapply(y))
    np.testing.assert_allclose(to_numpy(lhs), to_numpy(rhs), rtol=1e-6, atol=1e-6)


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

    out = sc.VectorSpace((4,), ctx)
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
    space = sc.VectorSpace((2,), ctx)
    op = sc.IdentityLinOp(space, ctx)
    x = ctx.asarray([1.0 + 2.0j, 3.0 - 4.0j])

    assert op.apply(x) is x
    assert op.rapply(x) is x


def test_identity_linop_apply_equals_input_when_checks_enabled():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(enable_checks=True)
    space = sc.VectorSpace((2,), ctx)
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
    X = sc.VectorSpace((2,), ctx)
    Y = sc.VectorSpace((3,), ctx)
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
    X32 = sc.VectorSpace((2,), ctx32)
    Y32 = sc.VectorSpace((2,), ctx32)
    X64 = sc.VectorSpace((2,), ctx64)
    Y64 = sc.VectorSpace((2,), ctx64)
    A32 = sc.DenseLinOp(ctx32.asarray([[1.0, 2.0], [3.0, 4.0]]), X32, Y32, ctx32)
    A64 = sc.DenseLinOp(ctx64.asarray([[1.0, 2.0], [3.0, 4.0]]), X64, Y64, ctx64)

    with pytest.raises(ValueError, match="same ctx"):
        sc.make_sum((A32, A64))
    with pytest.raises(ValueError, match="same ctx"):
        sc.make_composed(A32, A64)


def test_factories_enforce_domain_and_codomain_compatibility():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)
    X = sc.VectorSpace((2,), ctx)
    Y = sc.VectorSpace((3,), ctx)
    Z = sc.VectorSpace((4,), ctx)
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
