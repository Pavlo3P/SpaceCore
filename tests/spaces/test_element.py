"""Elements: checked once at the seam, arithmetic closed in the space, representation per subclass."""
import numpy as np
import pytest

import spacecore as sc

rng = np.random.default_rng(0)


def herm(n=3):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (a + a.conj().T) / 2


def factors(r):
    return rng.standard_normal((3, r)) + 1j * rng.standard_normal((3, r)), rng.standard_normal(r)


@pytest.fixture
def H():
    return sc.HermitianSpace(3, ctx=sc.Context(sc.NumpyOps(), dtype=np.complex128))


def test_dense_arithmetic_matches_numpy(H):
    a, b = herm(), herm()
    x, y = sc.DenseElement(H, a), sc.DenseElement(H, b)
    assert isinstance(x - 2.0 * y, sc.DenseElement)
    np.testing.assert_allclose((x - 2.0 * y).array, a - 2.0 * b)
    np.testing.assert_allclose((-x).array, -a)
    np.testing.assert_allclose(sum([x, y]).array, a + b)


def test_low_rank_stays_factored(H):
    p, q = sc.LowRankElement(H, *factors(1)), sc.LowRankElement(H, *factors(2))
    s = p + q
    assert isinstance(s, sc.LowRankElement) and s.U.shape == (3, 3)
    np.testing.assert_allclose(s.densify(), p.densify() + q.densify())
    assert isinstance(3.0 * s, sc.LowRankElement)
    np.testing.assert_allclose((3.0 * s).densify(), 3.0 * s.densify())


def test_mixed_representations_fall_back_to_dense(H):
    a, p = herm(), sc.LowRankElement(H, *factors(2))
    for s in (sc.DenseElement(H, a) + p, p + sc.DenseElement(H, a)):
        assert isinstance(s, sc.DenseElement)
        np.testing.assert_allclose(s.array, a + p.densify())


def test_scalar_field_guard_and_numpy_scalars(H):
    assert isinstance(np.float64(2.0) * sc.DenseElement(H, herm()), sc.DenseElement)
    for x in (sc.DenseElement(H, herm()), sc.LowRankElement(H, *factors(2))):
        with pytest.raises(sc.SpaceValidationError):
            1j * x


def test_unsupported_operations_raise(H):
    x = sc.DenseElement(H, herm())
    for undefined in (lambda: x * x, lambda: x + herm(), lambda: x @ x):
        with pytest.raises(TypeError):
            undefined()
    real = sc.HermitianSpace(3, ctx=sc.Context(sc.NumpyOps(), dtype=np.float64))
    with pytest.raises(sc.SpaceValidationError):
        x + sc.DenseElement(real, np.eye(3))  # a different space


def test_membership_checked_at_construction(H):
    with pytest.raises(sc.SpaceValidationError):
        sc.DenseElement(H, rng.standard_normal((3, 3)) + 1j)  # not Hermitian
    with pytest.raises(sc.SpaceValidationError):
        sc.LowRankElement(H, np.ones((3, 1), complex), np.array([1j]))  # complex weights
    with pytest.raises(sc.SpaceValidationError):
        sc.LowRankElement(sc.DenseCoordinateSpace((3, 3), H.ctx), *factors(1))


def test_context_is_the_space_context(H):
    x = sc.LowRankElement(H, *factors(2))
    assert x.ctx == H.ctx and x.check_level == H.check_level
    y = x.convert(sc.Context(sc.NumpyOps(), dtype=np.complex64))
    assert y.U.dtype == np.complex64 and y.d.dtype == np.float32  # real weights stay real
    assert y.space.dtype == np.complex64


def test_operator_application(H):
    A = sc.DenseLinOp(H.ctx.asarray(2.0 * np.eye(9).reshape(3, 3, 3, 3)), H, H, H.ctx)
    p = sc.LowRankElement(H, *factors(2))
    y = A @ p
    assert isinstance(y, sc.DenseElement) and y.space == H
    np.testing.assert_allclose(y.array, 2.0 * p.densify())


def test_jit_sees_through_elements():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    # DenseCoordinateSpace: its checks are structural, so they are trace-safe.
    S = sc.DenseCoordinateSpace((3, 3), ctx=sc.Context(sc.JaxOps(), dtype=jnp.float32))
    a = jnp.asarray(rng.standard_normal((3, 3)), dtype=jnp.float32)
    y = jax.jit(lambda u, v: (u + v) * 2.0)(sc.DenseElement(S, a), sc.DenseElement(S, a))
    assert isinstance(y, sc.DenseElement)
    np.testing.assert_allclose(np.asarray(y.array), 4.0 * np.asarray(a), rtol=1e-6)
