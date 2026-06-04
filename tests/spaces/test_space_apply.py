import numpy as np
import pytest

from spacecore import Context, JaxOps, NumpyOps
from spacecore.space import ElementwiseJordanSpace, HermitianSpace, ProductSpace


def _np_ctx():
    return Context(ops=NumpyOps(), enable_checks=True)


def _jax_ctx():
    return Context(ops=JaxOps(), enable_checks=False)


def test_elementwise_spectral_apply_numpy_entrywise():
    sp = ElementwiseJordanSpace((3,), ctx=_np_ctx())
    x = np.asarray([1.0, 2.0, 3.0], dtype=sp.dtype)

    y = sp.spectral_apply(x, np.square)

    expected = np.asarray([1.0, 4.0, 9.0], dtype=sp.dtype)
    np.testing.assert_allclose(y, expected)
    assert not hasattr(sp, "apply")


def test_elementwise_spectral_apply_numpy_shape_change_raises():
    sp = ElementwiseJordanSpace((3,), ctx=_np_ctx())
    x = np.asarray([1.0, 2.0, 3.0], dtype=sp.dtype)

    def bad_f(z):
        return z[:2]

    with pytest.raises(ValueError, match="changed shape"):
        sp.spectral_apply(x, bad_f)


def test_product_spectral_apply_numpy_componentwise():
    sp1 = ElementwiseJordanSpace((2,), ctx=_np_ctx())
    sp2 = ElementwiseJordanSpace((3,), ctx=_np_ctx())
    psp = ProductSpace((sp1, sp2), ctx=_np_ctx())

    x = (
        np.asarray([1.0, 2.0], dtype=psp.spaces[0].dtype),
        np.asarray([3.0, 4.0, 5.0], dtype=psp.spaces[1].dtype),
    )

    y = psp.spectral_apply(x, np.square)

    np.testing.assert_allclose(y[0], np.asarray([1.0, 4.0], dtype=psp.spaces[0].dtype))
    np.testing.assert_allclose(
        y[1], np.asarray([9.0, 16.0, 25.0], dtype=psp.spaces[1].dtype)
    )
    assert not hasattr(psp, "apply")


def test_hermitian_spectral_apply_numpy_on_diagonal():
    sp = HermitianSpace(3, ctx=_np_ctx())
    x = np.diag(np.asarray([1.0, 2.0, 3.0], dtype=sp.dtype))

    y = sp.spectral_apply(x, np.exp)

    expected = np.diag(np.exp(np.asarray([1.0, 2.0, 3.0], dtype=sp.dtype)))
    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)
    assert not hasattr(sp, "apply")


def test_hermitian_spectral_apply_numpy_preserves_hermitian_structure():
    sp = HermitianSpace(2, ctx=_np_ctx())
    x = np.asarray([[2.0, 1.0], [1.0, 3.0]], dtype=sp.dtype)

    y = sp.spectral_apply(x, np.exp)

    np.testing.assert_allclose(y, y.T.conj(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "factory, expected",
    [
        (
            lambda sp: np.asarray([1.0, 2.0, 3.0], dtype=sp.dtype),
            np.asarray([1.0, 4.0, 9.0]),
        ),
    ],
)
def test_elementwise_spectral_apply_numpy_basic_regression(factory, expected):
    sp = ElementwiseJordanSpace((3,), ctx=_np_ctx())
    x = factory(sp)
    y = sp.spectral_apply(x, np.square)
    np.testing.assert_allclose(y, expected.astype(sp.dtype))


def test_elementwise_spectral_apply_jax_matches_eager_and_compiles():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    sp = ElementwiseJordanSpace((3,), ctx=_jax_ctx())
    x = jnp.asarray([1.0, 2.0, 3.0], dtype=sp.dtype)
    f = jnp.square

    y_eager = sp.spectral_apply(x, f)

    @jax.jit
    def compiled_apply(z):
        return sp.spectral_apply(z, f)

    y_jit = compiled_apply(x)

    np.testing.assert_allclose(np.asarray(y_eager), np.asarray([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(np.asarray(y_jit), np.asarray([1.0, 4.0, 9.0]))


def test_product_spectral_apply_jax_matches_eager_and_compiles():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    sp1 = ElementwiseJordanSpace((2,), ctx=_jax_ctx())
    sp2 = ElementwiseJordanSpace((3,), ctx=_jax_ctx())
    psp = ProductSpace((sp1, sp2), ctx=_jax_ctx())

    x = (
        jnp.asarray([1.0, 2.0], dtype=psp.spaces[0].dtype),
        jnp.asarray([3.0, 4.0, 5.0], dtype=psp.spaces[1].dtype),
    )
    f = jnp.square

    y_eager = psp.spectral_apply(x, f)

    @jax.jit
    def compiled_apply(a, b):
        return psp.spectral_apply((a, b), f)

    y_jit = compiled_apply(*x)

    np.testing.assert_allclose(np.asarray(y_eager[0]), np.asarray([1.0, 4.0]))
    np.testing.assert_allclose(np.asarray(y_eager[1]), np.asarray([9.0, 16.0, 25.0]))
    np.testing.assert_allclose(np.asarray(y_jit[0]), np.asarray([1.0, 4.0]))
    np.testing.assert_allclose(np.asarray(y_jit[1]), np.asarray([9.0, 16.0, 25.0]))


def test_hermitian_spectral_apply_jax_diagonal_matches_eager_and_compiles():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    sp = HermitianSpace(3, ctx=_jax_ctx())
    x = jnp.diag(jnp.asarray([1.0, 2.0, 3.0], dtype=sp.dtype))
    f = jnp.exp

    y_eager = sp.spectral_apply(x, f)

    @jax.jit
    def compiled_apply(z):
        return sp.spectral_apply(z, f)

    y_jit = compiled_apply(x)

    expected = np.diag(np.exp(np.asarray([1.0, 2.0, 3.0])))
    np.testing.assert_allclose(np.asarray(y_eager), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(y_jit), expected, rtol=1e-6, atol=1e-6)
