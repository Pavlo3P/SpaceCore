"""Functional fields remain independent across binding, algebra and round trips."""
import numpy as np
import pytest

import spacecore as sc
from spacecore._checks import checked_method


@pytest.fixture
def real_valued():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    return sc.MatrixFreeLinearFunctional(
        lambda x: 2 * x.sum(), dom,
    )


def test_real_domain_real_codomain(real_valued):
    f = real_valued
    assert f.domain.dtype == np.dtype('float64')
    assert f.ctx.dtype == np.dtype('float64')
    assert isinstance(f.codomain, sc.RealField)
    assert f.codomain.dtype == np.dtype('float64')
    assert f.value(np.array([1., 2.])) == 6
    assert 'ℝ' in repr(f)


@pytest.mark.parametrize('build', [lambda f: f, lambda f: 2 * f, lambda f: f + f,
                                  lambda f: f + 1, lambda f: f * f,
                                  lambda f: 0 * f,
                                  lambda f: f.compose(sc.IdentityLinOp(f.domain))])
def test_codomain_survives_algebra_conversion_and_pytree(real_valued, build):
    f = build(real_valued)
    x = np.array([1., 2.])
    children, aux = f.tree_flatten()
    restored = type(f).tree_unflatten(aux, children)
    converted = f.convert(sc.Context(sc.NumpyOps(), dtype=np.float32))
    for obj in (f, restored, converted):
        assert isinstance(obj.codomain, sc.RealField)
        assert obj.domain.field == 'real'
        np.testing.assert_allclose(obj.value(obj.ctx.asarray(x)), f.value(x))
    assert converted.codomain.dtype == np.dtype('float32')
    assert converted.domain.dtype == np.dtype('float32')


def test_explicit_real_codomain_matches_default(real_valued):
    f = real_valued
    real = sc.MatrixFreeLinearFunctional(f.value_fn, f.domain)
    assert f == real


def test_default_hermitian_codomain_is_real():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.complex128)
    dom = sc.HermitianSpace(2, ctx=ctx)
    f = sc.InnerProductFunctional(dom.zeros(), dom)
    assert isinstance(f.codomain, sc.RealField)
    assert f.codomain.dtype == np.dtype('float64')
    assert f.domain.dtype == np.dtype('complex128')


@pytest.mark.parametrize('level,raises', [('none', False), ('cheap', False),
                                          ('standard', True), ('strict', True)])
def test_functional_policy_controls_field_output_check(level, raises):
    # The domain's policy differs deliberately; the functional's policy decides.
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), ctx, check_level='none')
    f = sc.MatrixFreeLinearFunctional(lambda x: x, dom, check_level=level)
    for obj in (f, f.convert(sc.Context(sc.NumpyOps(), dtype=np.float32))):
        if raises:
            with pytest.raises(sc.SpaceValidationError, match='Expected scalar output'):
                obj.value(obj.ctx.asarray([1., 2.]))
        else:
            obj.value(obj.ctx.asarray([1., 2.]))


def test_exact_batch_check_is_stronger_than_scalar_field_batch_membership():
    class InvalidBatch(sc.MatrixFreeLinearFunctional):
        def value(self, x):
            return x.sum()

        @checked_method(in_space='domain', in_batched=True, out_batched_scalar=True)
        def vvalue(self, xs):
            return self.ctx.asarray(np.ones((len(xs), 1)))

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    f = InvalidBatch(lambda x: x.sum(), sc.DenseCoordinateSpace((2,), ctx))
    from spacecore.space.checks import _run_checks

    _run_checks(f.codomain, np.ones((3, 1)), allow_leading=True)
    with pytest.raises(ValueError, match='scalar batch output'):
        f.vvalue(np.ones((3, 2)))


def test_real_constant_and_zero_round_trip():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    for f in (sc.ConstantFunctional(dom, 1.), sc.ZeroFunctional(dom)):
        children, aux = f.tree_flatten()
        restored = type(f).tree_unflatten(aux, children)
        assert restored == f
        assert isinstance(restored.codomain, sc.RealField)
        np.testing.assert_allclose(restored.value(dom.zeros()), f.value(dom.zeros()))
