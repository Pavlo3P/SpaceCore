"""Differentials agree with directional changes, independently of geometry."""
import numpy as np
import pytest

import spacecore as sc


def space(complex_=False, weighted=False):
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128 if complex_ else np.float64)
    geometry = sc.WeightedInnerProduct(ctx.asarray([2., 5.])) if weighted else None
    return sc.DenseCoordinateSpace((2,), ctx, geometry=geometry)


def check_direction(f, x, h):
    d = f.derivative_at(x)
    assert isinstance(d, sc.LinearFunctional)
    eps = 1e-6
    expected = (f(x + eps * h) - f(x - eps * h)) / (2 * eps)
    np.testing.assert_allclose(d(h), expected, rtol=1e-6, atol=1e-7)
    value, fused = f.value_and_derivative(x)
    np.testing.assert_allclose(value, f(x))
    np.testing.assert_allclose(fused(h), expected, rtol=1e-6, atol=1e-7)
    return d


@pytest.mark.parametrize('complex_', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_metric_and_complex_direction(complex_, weighted):
    X = space(complex_, weighted)
    x = X.ctx.asarray([1 + 2j, -3 + .5j] if complex_ else [1., -3.])
    h = X.ctx.asarray([.3 - 1j, 2 + .7j] if complex_ else [.3, 2.])
    f = sc.SquaredL2NormFunctional(X)
    d = check_direction(f, x, h)
    np.testing.assert_allclose(d.representer, f.grad(x))
    check_direction(2 * f + f * f + 3, x, h)
    if complex_:
        assert not np.allclose(d(1j * h), 1j * d(h))


class NativeDerivative(sc.Functional):
    """An objective with a callable differential and no gradient implementation."""

    def value(self, x):
        return x[0] ** 3 + x[1] ** 2

    def derivative_at(self, x):
        return sc.MatrixFreeLinearFunctional(
            lambda h: 3 * x[0] ** 2 * h[0] + 2 * x[1] * h[1], self.domain,
        )

    def tree_flatten(self):
        return (), (self.domain,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux[0])

    def _convert(self, ctx):
        return type(self)(self.domain.convert(ctx))


class RepresentedDerivative(NativeDerivative):
    def derivative_at(self, x):
        coordinates = self.ctx.asarray([3 * x[0] ** 2, 2 * x[1]])
        return sc.InnerProductFunctional(self.domain.riesz_inverse(coordinates), self.domain)


def test_native_derivative_algebra_and_forward_only_pullback():
    X = space(weighted=True)
    f = NativeDerivative(X)
    x, h = np.array([.4, -.7]), np.array([.3, 1.2])
    d = check_direction(f, x, h)
    assert d.derivative_at(x) is d
    with pytest.raises(NotImplementedError, match='representer'):
        f.grad(x)
    check_direction(2 * f + f * f + 4, x, h)
    A = sc.MatrixFreeLinOp(lambda v: 3 * v, lambda v: (_ for _ in ()).throw(
        AssertionError('directional pullback must not require an adjoint')),
        X, X, X.ctx, euclidean_adjoint=False, check_level='none')
    check_direction(f @ A, x, h)
    for expr in (d + d, 2 * d, 0 * d, d @ A):
        assert isinstance(expr, sc.LinearFunctional)
        assert expr.derivative_at(x) is expr
    g = RepresentedDerivative(X)
    check_direction(g, x, h)
    np.testing.assert_allclose(X.riesz(g.grad(x)), [3 * x[0] ** 2, 2 * x[1]])
    np.testing.assert_allclose((g + 2 * g).grad(x), 3 * g.grad(x))


def test_real_linear_maps_on_complex_domains():
    X = space(complex_=True)
    x, h = np.array([1+2j, 3-1j]), np.array([.5+1j, -2j])
    f = sc.InnerProductFunctional(np.array([2+1j, 3-4j]), X)
    assert check_direction(f, x, h) is f
    d = sc.MatrixFreeLinearFunctional(lambda v: np.real(v[0]), X)
    check_direction(d * d + f, x, h)
    q = sc.LinOpQuadraticForm(sc.IdentityLinOp(X), f)
    check_direction(q, x, h)
    check_direction(2 * q, x, h)


def test_real_pairing_roundtrip_batch_and_pullback():
    X = space(complex_=True, weighted=True)
    d = sc.InnerProductFunctional(np.array([1+2j, 3-1j]), X)
    h = np.array([.5+1j, -2j])
    children, aux = d.tree_flatten()
    restored = type(d).tree_unflatten(aux, children)
    A = sc.DenseLinOp(np.array([[1j, 2], [3, -1j]]), X, X)
    for expr in (d, restored, d + d, 2 * d, d @ A):
        assert isinstance(expr.codomain, sc.RealField)
        np.testing.assert_allclose(expr(h), np.real(X.inner(expr.representer, h)))
    np.testing.assert_allclose(d.vvalue(np.stack([h, 2*h])), [d(h), 2*d(h)])
    np.testing.assert_allclose((d @ A)(h), d(A(h)))
    converted = d.convert(sc.Context(sc.NumpyOps(), dtype=np.complex64))
    np.testing.assert_allclose(converted(h.astype(np.complex64)), d(h), rtol=1e-6)


def test_fused_evaluation_is_used_once():
    class Fused(RepresentedDerivative):
        calls = 0
        derivative_at = sc.Functional.derivative_at

        def value_and_grad(self, x, factor=1):
            self.calls += 1
            return factor * super().value(x), factor * self.ctx.asarray([3*x[0]**2, 2*x[1]])

        def grad(self, x, factor=1):
            return self.value_and_grad(x, factor)[1]

    f = Fused(space())
    value, d = f.value_and_derivative(np.array([2., 3.]), factor=2)
    assert f.calls == 1
    assert value == 34
    assert d(np.array([1., 1.])) == 36
    (2 * f + f).value_and_derivative(np.array([2., 3.]))
    assert f.calls == 3


def test_jax_differential_is_traceable():
    jax = pytest.importorskip('jax')
    import jax.numpy as jnp
    X = sc.DenseCoordinateSpace((2,), sc.Context(sc.JaxOps(), dtype=jnp.complex64),
                                check_level='none')
    f = sc.SquaredL2NormFunctional(X)
    x, h = jnp.array([1+2j, 3-1j]), jnp.array([.5+1j, -2j])
    d = jax.jit(f.derivative_at)(x)
    np.testing.assert_allclose(d(h), jax.jvp(f.value, (x,), (h,))[1])


def test_no_array_conversion_for_native_representation():
    from tests.linalg.test_native_krylov import Pair, PairSpace

    X = PairSpace(sc.Context(sc.NumpyOps(), dtype=np.float64))  # Pair forbids __array__
    x = Pair(np.asarray(1.), np.array([2., 0.]))
    h = Pair(np.asarray(3.), np.array([4., 5.]))
    d = sc.SquaredL2NormFunctional(X).derivative_at(x)
    assert d(h) == 14  # <x, h> = 2*1*3 + 2*4
    assert d.representer is x
    assert (2 * d + d)(h) == 42


def test_complex_scaled_real_differential_is_rejected():
    X = space(complex_=True)
    d = sc.SquaredL2NormFunctional(X).derivative_at(np.array([1+2j, 3-1j]))
    with pytest.raises(TypeError, match='real value'):
        1j * d
