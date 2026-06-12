import importlib

import numpy as np
import pytest


def _ctx(dtype=np.float64, enable_checks=True):
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=enable_checks)


def _quadratic_problem(ctx):
    sc = importlib.import_module("spacecore")
    dom = sc.DenseCoordinateSpace((2,), ctx)
    Q = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 4.0]]), dom, dom, ctx)
    linear = sc.InnerProductFunctional(ctx.asarray([1.0, -1.0]), dom, ctx)
    return sc.LinOpQuadraticForm(Q, linear, 3.0, ctx)


def test_explicit_context_overrides_inferred_contexts():
    sc = importlib.import_module("spacecore")
    inferred = _ctx(np.float32, enable_checks=True)
    explicit = _ctx(np.float64, enable_checks=False)
    dom = sc.DenseCoordinateSpace((2,), inferred)
    Q = sc.DenseLinOp(inferred.asarray([[1.0, 0.0], [0.0, 1.0]]), dom, dom, inferred)
    linear = sc.InnerProductFunctional(inferred.asarray([1.0, 2.0]), dom)

    functional = sc.InnerProductFunctional(inferred.asarray([1.0, 2.0]), dom, explicit)
    quadratic = sc.LinOpQuadraticForm(Q, linear, 0.0, explicit)

    assert functional.ctx == explicit
    assert functional.dtype == np.dtype(np.float64)
    assert functional.domain.ctx == explicit
    assert quadratic.ctx == explicit
    assert quadratic.Q.ctx == explicit
    assert quadratic.linear.ctx == explicit


def test_domain_conversion_and_membership_checks_work():
    sc = importlib.import_module("spacecore")
    source = _ctx(np.float32, enable_checks=True)
    explicit = _ctx(np.float64, enable_checks=True)
    dom = sc.DenseCoordinateSpace((2,), source)
    functional = sc.InnerProductFunctional(source.asarray([1.0, 2.0]), dom, explicit)

    assert functional.ctx == explicit
    assert functional.dtype == np.dtype(np.float64)
    assert functional.domain.ctx == explicit
    assert functional.domain.ctx.enable_checks is True
    assert np.allclose(functional.value(functional.domain.ctx.asarray([3.0, 4.0])), 11.0)
    with pytest.raises(Exception):
        functional.value(explicit.asarray([1.0, 2.0, 3.0]))


def test_functional_construction_rejects_complex_data_for_real_space():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(np.float32)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    representer = np.asarray([1.0 + 1.0j, 2.0], dtype=np.complex64)

    with pytest.raises(TypeError, match="rejected complex-valued input.*x.real"):
        sc.InnerProductFunctional(representer, dom, ctx)


def test_call_matches_value():
    ctx = _ctx()
    q = _quadratic_problem(ctx)
    x = ctx.asarray([2.0, -1.0])
    assert np.allclose(q(x), q.value(x))


def test_inner_product_functional_matches_domain_inner():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.DenseCoordinateSpace((2,), ctx)
    c = ctx.asarray([1.0, -2.0])
    x = ctx.asarray([3.0, 4.0])
    functional = sc.InnerProductFunctional(c, dom, ctx)

    assert np.allclose(functional.value(x), dom.inner(c, x))


def test_matrix_free_linear_functional_has_no_representer():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.DenseCoordinateSpace((2,), ctx)
    c = ctx.asarray([2.0, 3.0])
    x = ctx.asarray([4.0, 5.0])
    functional = sc.MatrixFreeLinearFunctional(lambda y: dom.inner(c, y), dom, ctx)

    assert np.allclose(functional.value(x), 23.0)
    with pytest.raises(NotImplementedError):
        functional.representer


def test_linear_functional_compose_specializes_to_inner_product_functional():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y = sc.DenseCoordinateSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]]), X, Y, ctx)
    c = ctx.asarray([2.0, -1.0, 0.5])
    F = sc.InnerProductFunctional(c, Y, ctx)
    pullback = F.compose(A)
    x = ctx.asarray([4.0, -2.0])

    assert isinstance(pullback, sc.InnerProductFunctional)
    np.testing.assert_allclose(pullback.representer, A.H.apply(c))
    np.testing.assert_allclose(pullback.value(x), F.value(A.apply(x)))


def test_quadratic_form_compose_specializes_quadratic_and_linear_terms():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y = sc.DenseCoordinateSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]]), X, Y, ctx)
    Q = sc.IdentityLinOp(Y, ctx)
    linear = sc.InnerProductFunctional(ctx.asarray([1.0, -2.0, 0.5]), Y, ctx)
    F = sc.LinOpQuadraticForm(Q, linear, 1.25, ctx)
    pullback = F.compose(A)
    x = ctx.asarray([0.5, -1.5])

    assert isinstance(pullback, sc.LinOpQuadraticForm)
    np.testing.assert_allclose(pullback.value(x), F.value(A.apply(x)))
    np.testing.assert_allclose(pullback.grad(x), A.H.apply(F.grad(A.apply(x))))


def test_generic_functional_compose_forwards_value():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y = sc.DenseCoordinateSpace((2,), ctx)
    A = sc.DiagonalLinOp(ctx.asarray([2.0, -1.0]), X, ctx)

    class SumSquares(sc.Functional):
        def value(self, x):
            return self.ops.sum(x * x)

        def tree_flatten(self):
            return (), (self.domain, self.ctx)

        @classmethod
        def tree_unflatten(cls, aux, children):
            domain, ctx = aux
            return cls(domain, ctx)

        def _convert(self, new_ctx):
            return SumSquares(self.domain.convert(new_ctx), new_ctx)

    F = SumSquares(Y, ctx)
    pullback = F.compose(A)
    x = ctx.asarray([3.0, 4.0])

    assert isinstance(pullback, sc.ComposedFunctional)
    np.testing.assert_allclose(pullback.value(x), F.value(A.apply(x)))


def test_functional_compose_rejects_incompatible_codomain():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    X = sc.DenseCoordinateSpace((2,), ctx)
    Y = sc.DenseCoordinateSpace((3,), ctx)
    A = sc.IdentityLinOp(X, ctx)
    F = sc.InnerProductFunctional(ctx.asarray([1.0, 2.0, 3.0]), Y, ctx)

    with pytest.raises(ValueError, match="A.codomain == F.domain"):
        F.compose(A)


def test_linop_quadratic_value_and_gradient_match_euclidean_hand_computation():
    ctx = _ctx()
    q = _quadratic_problem(ctx)
    x = ctx.asarray([2.0, -1.0])

    assert np.allclose(q.value(x), 12.0)
    assert np.allclose(q.grad(x), [5.0, -5.0])
    assert np.allclose(q.hess_apply(x), [4.0, -4.0])


def test_linop_quadratic_form_hermitian_gradient_is_q_apply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2,), ctx)
    Q = sc.DenseLinOp(ctx.asarray([[2.0, 1.0], [1.0, 4.0]]), space, space, ctx)
    q = sc.LinOpQuadraticForm(Q, ctx=ctx)
    x = ctx.asarray([2.0, -1.0])

    np.testing.assert_allclose(q.grad(x), Q.apply(x))
    np.testing.assert_allclose(q.hess_apply(x), Q.apply(x))


def test_linop_quadratic_form_rejects_non_hermitian_dense_operator():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2,), ctx)
    Q = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, 3.0]]), space, space, ctx)

    with pytest.raises(ValueError, match="Hermitian"):
        sc.LinOpQuadraticForm(Q, ctx=ctx)


def test_linop_quadratic_form_does_not_validate_matrix_free_hermitian_assumption():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2,), ctx)

    def apply(x):
        return ctx.asarray([x[0] + 2.0 * x[1], 3.0 * x[1]])

    def rapply(y):
        return ctx.asarray([y[0], 2.0 * y[0] + 3.0 * y[1]])

    Q = sc.MatrixFreeLinOp(apply, rapply, space, space, ctx)
    q = sc.LinOpQuadraticForm(Q, ctx=ctx)
    x = ctx.asarray([1.0, 2.0])

    np.testing.assert_allclose(q.grad(x), Q.apply(x))


def test_linop_quadratic_form_always_rejects_nonscalar_constant():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(enable_checks=False)
    space = sc.DenseCoordinateSpace((2,), ctx)
    Q = sc.IdentityLinOp(space, ctx)

    with pytest.raises(ValueError, match="scalar batch"):
        sc.LinOpQuadraticForm(Q, a=ctx.asarray([0.0, 0.0]), ctx=ctx)


def test_vvalue_and_vgrad_match_elementwise_loops():
    ctx = _ctx()
    q = _quadratic_problem(ctx)
    xs = ctx.asarray([[2.0, -1.0], [0.0, 3.0], [1.5, 2.0]])

    expected_values = ctx.ops.stack(tuple(q.value(x) for x in xs), axis=0)
    expected_grads = ctx.ops.stack(tuple(q.grad(x) for x in xs), axis=0)

    assert np.allclose(q.vvalue(xs), expected_values)
    assert np.allclose(q.vgrad(xs), expected_grads)


def test_bad_shapes_raise_when_checks_are_enabled():
    ctx = _ctx(enable_checks=True)
    q = _quadratic_problem(ctx)
    bad = ctx.asarray([1.0, 2.0, 3.0])

    with pytest.raises(Exception):
        q.value(bad)
    with pytest.raises(Exception):
        q.grad(bad)
    with pytest.raises(Exception):
        q.vvalue(ctx.asarray([[1.0, 2.0, 3.0]]))
