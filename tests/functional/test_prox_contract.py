"""Proximal optimality, Hilbert conjugacy, extension, and capability boundaries."""
import numpy as np
import pytest

import spacecore as sc
from spacecore.mapping import register
from tests._helpers import to_numpy


def space(backend="numpy", weighted=True, level="standard"):
    if backend == "jax":
        pytest.importorskip("jax")
        ops, dtype = sc.JaxOps(), np.float32
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        ops, dtype = sc.TorchOps(), torch.float64
    else:
        ops, dtype = sc.NumpyOps(), np.float64
    ctx = sc.Context(ops, dtype)
    geometry = sc.WeightedInnerProduct(ctx.asarray([2., 5., 3.])) if weighted else None
    return sc.DenseCoordinateSpace((3,), ctx, geometry=geometry, check_level=level)


def array(X, value):
    return np.asarray(to_numpy(value))


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
@pytest.mark.parametrize("weighted", [False, True])
def test_proximal_kkt_and_hilbert_conjugate(backend, weighted):
    X = space(backend, weighted)
    x = X.ctx.asarray([-2., .05, 3.])
    weights = np.array([2., 5., 3.]) if weighted else np.ones(3)
    step, coefficient = .7, 1.3
    f = coefficient * sc.L1NormFunctional(X) + 2.0
    z = array(X, f.prox(x=x, step=step))
    subgrad = weights * (array(X, x) - z) / step
    active = np.abs(z) > 1e-7
    np.testing.assert_allclose(subgrad[active], coefficient * np.sign(z[active]), atol=2e-6)
    assert np.all(np.abs(subgrad[~active]) <= coefficient + 1e-6)

    # The Hilbert conjugate's feasible set is |w_i p_i| <= coefficient.
    p = array(X, f.conjugate_prox(x, step))
    np.testing.assert_allclose(p, np.clip(array(X, x), -coefficient / weights,
                                         coefficient / weights), atol=2e-6)
    quadratic = sc.SquaredL2NormFunctional(X)
    np.testing.assert_allclose(array(X, quadratic.prox(x, step)), array(X, x)/(1+step), atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_projection_indicator_polar_identity_and_conversion(backend):
    X = space(backend)
    C = sc.NonnegativeOrthant(X)
    f = sc.IndicatorFunctional(C)
    x = X.ctx.asarray([-2., .5, 3.])
    z = f.prox(x, .4)
    np.testing.assert_allclose(array(X, z), [0, .5, 3])
    assert bool(C.contains(z))
    assert float(f(z)) == 0
    assert np.isinf(float(f(x)))
    np.testing.assert_allclose(array(X, f.conjugate_prox(x, .4)), [-2, 0, 0], atol=1e-6)
    # Projection variational inequality over independent feasible points.
    rng = np.random.default_rng(2)
    for y in rng.uniform(0, 4, size=(10, 3)):
        assert float(X.inner(X.axpy(-1, z, x), X.axpy(-1, z, X.ctx.asarray(y)))) <= 1e-6
    converted = f.convert(sc.Context(sc.NumpyOps(), np.float64))
    assert float(converted(converted.domain.ctx.asarray([1., 2., 3.]))) == 0


@pytest.mark.parametrize("step", [0., -1., float("nan"), float("inf"), 1j, [1.]])
def test_invalid_steps(step):
    X = space()
    f = sc.SquaredL2NormFunctional(X)
    for method in (f.prox, f.conjugate_prox):
        with pytest.raises(ValueError, match="step"):
            method(X.zeros(), step)


def test_no_fictitious_rules_or_silent_geometry():
    X = space()
    x = X.zeros()
    f = sc.L1NormFunctional(X)
    for unsupported in (f + sc.SquaredL2NormFunctional(X), f @ sc.IdentityLinOp(X),
                        sc.LpNormFunctional(X, 3)):
        with pytest.raises(sc.CapabilityError):
            unsupported.prox(x, .5)
    with pytest.raises(ValueError):
        (-f).prox(x, .5)

    class UnspecifiedRiesz(sc.InnerProduct):
        def inner(self, ops, x, y):
            return 2 * ops.vdot(x, y)

    geometry = UnspecifiedRiesz()
    Y = sc.DenseCoordinateSpace((3,), X.ctx, geometry=geometry)
    for operation in (Y.riesz, Y.riesz_inverse):
        with pytest.raises(sc.CapabilityError, match="riesz"):
            operation(x)
    with pytest.raises(sc.CapabilityError):
        sc.L1NormFunctional(Y).prox(x, 1.)
    with pytest.raises(sc.CapabilityError):
        sc.NonnegativeOrthant(Y)
    # The squared *space* norm has a prox even for a full metric.
    np.testing.assert_allclose(sc.SquaredL2NormFunctional(Y).prox(x, 1.), x / 2)


def test_rule_specificity_and_missing_native_initialization():
    class Custom(sc.SquaredL2NormFunctional):
        pass

    X = space()
    register("prox", Custom, object, lambda f, g, x, t: f.domain.scale(1/(1+t), x))
    register("prox", Custom, sc.WeightedInnerProduct,
             lambda f, g, x, t: f.domain.scale(1/(1+t), x))
    np.testing.assert_allclose(Custom(X).prox(X.ctx.asarray([1., 2., 3.]), 1.), [.5, 1, 1.5])
    from tests.linalg.test_native_krylov import PairSpace
    N = PairSpace(X.ctx)  # native: no shape, no ones(), no flatten
    K = sc.MatrixFreeLinOp(lambda x: x, lambda x: x, N, N, N.ctx)
    with pytest.raises(sc.CapabilityError, match="x0"):
        sc.power_iteration(K.H @ K, maxiter=4)
    with pytest.raises(sc.CapabilityError, match="maxiter"):
        sc.cg(K.H @ K, N.zeros())


def test_jit_dynamic_steps_and_indicator_pytree():
    jax = pytest.importorskip("jax")
    X = space("jax", level="none")
    f = 1.3 * sc.L1NormFunctional(X)
    g = sc.IndicatorFunctional(sc.NonnegativeOrthant(X))
    x = X.ctx.asarray([-2., .1, 3.])

    def run(x, t):
        return f.prox(x, t), f.conjugate_prox(x, t), g.prox(x, t)

    for step in (.2, .7):
        eager = run(x, step)
        compiled = jax.jit(run)(x, X.ctx.asarray(step))
        for a, b in zip(eager, compiled):
            np.testing.assert_allclose(a, b, atol=1e-6)
    # Geometry and indicator survive being passed as compiled arguments too.
    out = jax.jit(lambda indicator, x, t: indicator.prox(x, t))(g, x, X.ctx.asarray(.3))
    np.testing.assert_allclose(out, [0, .1, 3], atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
def test_complex_l1_uses_real_metric_weights(backend):
    base = space(backend)
    dtype = base.ops.complex_dtype(base.dtype)
    ctx = sc.Context(base.ops, dtype)
    X = sc.DenseCoordinateSpace((3,), ctx,
                               geometry=sc.WeightedInnerProduct(ctx.asarray([2., 5., 3.])))
    x = ctx.asarray([3+4j, 0j, -1+2j])
    f, step = sc.L1NormFunctional(X), .7
    z = array(X, f.prox(x, step))
    weights, x_np = np.array([2., 5., 3.]), array(X, x)
    # Nonzero-coordinate subgradient optimality in the real Hilbert pairing.
    active = np.abs(z) > 0
    np.testing.assert_allclose((weights*(x_np-z)/step)[active], z[active]/np.abs(z[active]), atol=2e-6)


def test_checked_array_steps_and_missing_membership():
    X = space()
    f = sc.SquaredL2NormFunctional(X)
    for t in (-1., 0., np.nan):
        with pytest.raises(ValueError):
            f.prox(X.zeros(), X.ctx.asarray(t))

    class ProjectionOnly(sc.NonnegativeOrthant):
        def contains(self, x):
            return None

    indicator = sc.IndicatorFunctional(ProjectionOnly(X))
    np.testing.assert_allclose(indicator.prox(X.zeros(), 1.), X.zeros())
    with pytest.raises(sc.CapabilityError, match="membership"):
        indicator(X.zeros())
