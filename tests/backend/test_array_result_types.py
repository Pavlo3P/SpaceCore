"""Array-valued methods must agree with their native result annotations."""
import pytest


@pytest.mark.parametrize(
    'name',
    ['sqrt', 'abs', 'conj', 'real', 'imag', 'sign', 'exp', 'log', 'isfinite', 'isnan'],
)
def test_scalar_elementwise_results_are_arrays(backend_ops, name):
    x = backend_ops.asarray(2.0 + 1.0j if name == "imag" else 2.0)
    result = getattr(backend_ops, name)(x)
    assert backend_ops.is_dense(result)
    assert backend_ops.shape(result) == ()
    assert backend_ops.get_dtype(result) is not None


def test_scalar_contractions_are_arrays(backend_ops):
    x = backend_ops.asarray([1.0, 2.0])
    for result in (backend_ops.matmul(x, x), backend_ops.einsum('i,i->', x, x)):
        assert backend_ops.is_dense(result)
        assert backend_ops.shape(result) == ()
        assert backend_ops.allclose(result, backend_ops.asarray(5.0))


def test_logsumexp_results_are_arrays(backend_ops):
    x = backend_ops.asarray([1.0, 2.0])
    value = backend_ops.logsumexp(x)
    signed_value, sign = backend_ops.logsumexp(x, return_sign=True)
    for result in (value, signed_value, sign):
        assert backend_ops.is_dense(result)
        assert backend_ops.shape(result) == ()
    assert backend_ops.allclose(value, signed_value)
    assert backend_ops.allclose(sign, backend_ops.asarray(1.0))
