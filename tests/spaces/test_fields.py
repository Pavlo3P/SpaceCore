"""The real field and declared scalar fields, independent of storage."""
import numpy as np
import pytest

import spacecore as sc
from spacecore.contextual import normalize_context
from spacecore._batching import _batched_inner
from spacecore.space._capabilities import probe_element, require


class RealCoefficients(sc.DenseCoordinateSpace):
    declared_scalar_field = 'real'


class ComplexCoefficients(sc.DenseCoordinateSpace):
    declared_scalar_field = 'complex'


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('declared', ['real', 'complex'])
def test_scalar_storage_table(dtype, declared):
    ctx = sc.Context(sc.NumpyOps(), dtype=dtype)
    cls = RealCoefficients if declared == 'real' else ComplexCoefficients
    if declared == 'complex' and not dtype.startswith('complex'):
        with pytest.raises(TypeError, match='complex floating storage'):
            cls((2,), ctx)
        return
    space = cls((2,), ctx)
    assert space.scalar_field == declared
    assert space.dtype == np.dtype(dtype)
    if declared == 'real':
        with pytest.raises(sc.SpaceValidationError):
            space.check_scalar(1j)


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'bool'])
@pytest.mark.parametrize('cls', [sc.RealField, sc.DenseCoordinateSpace])
def test_integer_boolean_storage_rejected(dtype, cls):
    ctx = sc.Context(sc.NumpyOps(), dtype=dtype)
    with pytest.raises(TypeError, match='floating storage'):
        cls((2,), ctx) if cls is sc.DenseCoordinateSpace else cls(ctx)


def test_field_laws_geometry_probe_and_missing_coordinates():
    field = sc.RealField(sc.Context(sc.NumpyOps(), dtype='float64'))
    assert isinstance(field, sc.VectorSpace)
    assert isinstance(field, sc.InnerProductSpace)
    assert not isinstance(field, sc.CoordinateSpace)
    for name in ('shape', 'size', 'flatten', 'stacked'):
        assert not hasattr(field, name)
    a, b = field.ctx.asarray(2.), field.ctx.asarray(4.)
    np.testing.assert_allclose(field.inner(a, b), a * b)
    assert field.riesz(a) is a
    assert field.riesz_inverse(a) is a
    assert field.is_euclidean
    assert field.zeros().shape == ()
    assert probe_element(field, field.ops, field.dtype) == 1
    np.testing.assert_allclose(field.axpy(2, a, b), 2 * a + b)
    with pytest.raises(sc.CapabilityError, match='CoordinateSpace'):
        require(field, sc.CoordinateSpace, 'flatten')
    with pytest.raises(sc.CapabilityError, match='CoordinateSpace'):
        _batched_inner(field, np.ones(2), np.ones(2))


@pytest.mark.parametrize('level,raises', [('none', False), ('cheap', False),
                                          ('standard', True), ('strict', True)])
def test_scalar_member_check_levels(level, raises):
    field = sc.RealField(check_level=level)
    field.check_member(2.)
    field.check_member(field.ones())
    if raises:
        with pytest.raises(sc.SpaceValidationError, match='Expected scalar output'):
            field.check_member(np.ones(2))
    else:
        field.check_member(np.ones(2))


def test_hermitian_space_is_real_over_complex_storage():
    space = sc.HermitianSpace(2, ctx=sc.Context(sc.NumpyOps(), dtype=np.complex128))
    assert space.dtype == np.dtype('complex128')
    assert space.field == 'complex' and space.scalar_field == 'real'


@pytest.mark.parametrize('family', ['numpy', 'torch', 'jax'])
def test_fields_preserve_backend_and_precision(family):
    if family != 'numpy':
        pytest.importorskip(family)
    ops = normalize_context(family).ops
    field = sc.RealField(sc.Context(ops, dtype='complex64'))
    assert field.ops == ops
    assert field.dtype == ops.sanitize_dtype('float32')
    assert field.ones().dtype == ops.sanitize_dtype('float32')


def test_default_fields_and_conversion_keep_mathematical_field():
    assert sc.RealField().field == 'real'
    assert sc.RealField().convert(sc.Context(sc.NumpyOps(), 'complex64')).dtype == np.dtype('float32')


def test_linop_rejects_field_codomain_at_construction():
    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    dom = sc.DenseCoordinateSpace((2,), ctx)
    with pytest.raises(TypeError, match='ADR-010'):
        sc.MatrixFreeLinOp(lambda x: x.sum(), lambda y: np.ones(2) * y, dom, sc.RealField(ctx))
