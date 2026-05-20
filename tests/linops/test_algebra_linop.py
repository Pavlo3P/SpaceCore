import importlib

import numpy as np
import pytest


def _ctx(dtype=np.float64, enable_checks=True):
    sc = importlib.import_module("spacecore")
    return sc.Context(sc.NumpyOps(), dtype=dtype, enable_checks=enable_checks)


def _op(matrix, dom_shape, cod_shape, ctx=None):
    sc = importlib.import_module("spacecore")
    ctx = ctx or _ctx()
    dom = sc.VectorSpace(dom_shape, ctx)
    cod = sc.VectorSpace(cod_shape, ctx)
    return sc.DenseLinOp(ctx.asarray(matrix), dom, cod, ctx)


def test_algebra_linops_inherit_from_linop():
    sc = importlib.import_module("spacecore")
    A = _op([[1.0, 2.0], [3.0, 4.0]], (2,), (2,))

    assert isinstance(2.0 * A, sc.LinOp)
    assert isinstance(A + A, sc.LinOp)
    assert isinstance(A @ A, sc.LinOp)
    assert isinstance(A.H, sc.LinOp)
    assert isinstance(sc.ZeroLinOp(A.domain, A.codomain, A.ctx), sc.LinOp)
    assert isinstance(sc.IdentityLinOp(A.domain, A.ctx), sc.LinOp)
    assert isinstance(sc.MatrixFreeLinOp(A.apply, A.rapply, A.domain, A.codomain, A.ctx), sc.LinOp)
    assert issubclass(sc.ScaledLinOp, sc.LinOp)
    assert issubclass(sc.SumLinOp, sc.LinOp)
    assert issubclass(sc.ComposedLinOp, sc.LinOp)
    assert issubclass(sc.ZeroLinOp, sc.LinOp)
    assert issubclass(sc.IdentityLinOp, sc.LinOp)
    assert issubclass(sc.MatrixFreeLinOp, sc.LinOp)
    assert not hasattr(sc, "AdjointLinOp")


def test_context_mismatch_raises_clear_error():
    A = _op([[1.0, 2.0], [3.0, 4.0]], (2,), (2,), _ctx(enable_checks=True))
    B = _op([[5.0, 6.0], [7.0, 8.0]], (2,), (2,), _ctx(enable_checks=False))

    with pytest.raises(ValueError, match="same ctx"):
        _ = A + B
    with pytest.raises(ValueError, match="same ctx"):
        _ = A @ B


def test_sum_requires_matching_domain_and_codomain():
    A = _op([[1.0, 2.0], [3.0, 4.0]], (2,), (2,))
    bad_cod = _op([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], (2,), (3,), A.ctx)
    bad_dom = _op([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (3,), (2,), A.ctx)

    with pytest.raises(ValueError, match="same domain and codomain"):
        _ = A + bad_cod
    with pytest.raises(ValueError, match="same domain and codomain"):
        _ = A + bad_dom


def test_composition_requires_matching_middle_space():
    A = _op([[1.0, 2.0], [3.0, 4.0]], (2,), (2,))
    B = _op([[1.0, 2.0], [3.0, 4.0]], (2,), (2,), A.ctx)
    C = _op([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], (2,), (3,), A.ctx)

    assert (A @ B).domain == B.domain
    assert (A @ B).codomain == A.codomain
    with pytest.raises(ValueError, match="right.codomain == left.domain"):
        _ = A @ C


def test_scaled_sum_subtraction_and_negation_are_numerically_correct():
    ctx = _ctx()
    A = _op([[1.0, 2.0], [3.0, 4.0]], (2,), (2,), ctx)
    B = _op([[5.0, 1.0], [-2.0, 3.0]], (2,), (2,), ctx)
    x = ctx.asarray([2.0, -1.0])
    y = ctx.asarray([1.0, 3.0])
    dense_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    dense_b = np.array([[5.0, 1.0], [-2.0, 3.0]])

    expr = 2.0 * A + B - (-A)

    assert expr.domain == A.domain
    assert expr.codomain == A.codomain
    assert np.allclose(expr.apply(x), (3.0 * dense_a + dense_b) @ np.asarray(x))
    assert np.allclose(expr.rapply(y), (3.0 * dense_a + dense_b).T @ np.asarray(y))
    assert np.allclose((-A).apply(x), -dense_a @ np.asarray(x))
    assert np.allclose((A * 3.0).apply(x), 3.0 * dense_a @ np.asarray(x))


def test_complex_scaled_adjoint_conjugates_scalar():
    ctx = _ctx(np.complex128)
    A = _op([[1.0 + 1.0j, 2.0], [3.0j, 4.0 - 2.0j]], (2,), (2,), ctx)
    y = ctx.asarray([1.0 - 1.0j, 2.0 + 3.0j])
    dense = np.array([[1.0 + 1.0j, 2.0], [3.0j, 4.0 - 2.0j]])
    alpha = 2.0 + 3.0j

    op = alpha * A

    assert np.allclose(op.rapply(y), np.conj(alpha) * dense.conj().T @ np.asarray(y))


def test_composition_apply_and_adjoint_are_numerically_correct():
    ctx = _ctx()
    A = _op([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], (2,), (3,), ctx)
    B = _op([[2.0, -1.0], [0.5, 3.0]], (2,), (2,), ctx)
    x = ctx.asarray([4.0, -2.0])
    z = ctx.asarray([1.0, -1.0, 2.0])
    dense_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dense_b = np.array([[2.0, -1.0], [0.5, 3.0]])

    op = A @ B

    assert op.domain == B.domain
    assert op.codomain == A.codomain
    assert np.allclose(op.apply(x), dense_a @ dense_b @ np.asarray(x))
    assert np.allclose(op.rapply(z), dense_b.T @ dense_a.T @ np.asarray(z))


def test_H_swaps_spaces_and_double_H_returns_original():
    ctx = _ctx()
    A = _op([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], (2,), (3,), ctx)
    x = ctx.asarray([7.0, 8.0])
    y = ctx.asarray([1.0, -1.0, 2.0])

    AH = A.H
    AHH = AH.H

    assert AH.ctx == A.ctx
    assert AH.domain == A.codomain
    assert AH.codomain == A.domain
    assert np.allclose(AH.apply(y), A.rapply(y))
    assert np.allclose(AH.rapply(x), A.apply(x))
    assert AHH is A
    assert np.allclose(AHH.apply(x), A.apply(x))
    assert np.allclose(AHH.rapply(y), A.rapply(y))


def test_zero_identity_and_matrix_free_rapply_are_numerically_correct():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    dom = sc.VectorSpace((2,), ctx)
    cod = sc.VectorSpace((3,), ctx)
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = ctx.asarray([7.0, 8.0])
    y = ctx.asarray([1.0, -1.0, 2.0])

    zero = sc.ZeroLinOp(dom, cod, ctx)
    identity = sc.IdentityLinOp(dom, ctx)
    matrix_free = sc.MatrixFreeLinOp(
        lambda v: ctx.asarray(dense @ np.asarray(v)),
        lambda w: ctx.asarray(dense.T @ np.asarray(w)),
        dom,
        cod,
        ctx,
    )

    assert np.allclose(zero.apply(x), np.zeros(3))
    assert np.allclose(zero.rapply(y), np.zeros(2))
    assert np.allclose(identity.apply(x), np.asarray(x))
    assert np.allclose(identity.rapply(x), np.asarray(x))
    assert np.allclose(matrix_free.apply(x), dense @ np.asarray(x))
    assert np.allclose(matrix_free.rapply(y), dense.T @ np.asarray(y))


def test_sum_factory_flattens_nested_sums_and_removes_zero_terms():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    A = _op([[1.0, 2.0], [3.0, 4.0]], (2,), (2,), ctx)
    B = _op([[5.0, 1.0], [-2.0, 3.0]], (2,), (2,), ctx)
    Z = sc.ZeroLinOp(A.domain, A.codomain, ctx)
    x = ctx.asarray([2.0, -1.0])
    y = ctx.asarray([1.0, 3.0])

    nested = sc.SumLinOp((A, B))
    simplified = nested + Z
    zero_sum = Z + Z

    assert isinstance(simplified, sc.SumLinOp)
    assert simplified.parts == (A, B)
    assert A + Z is A
    assert Z + A is A
    assert isinstance(zero_sum, sc.ZeroLinOp)
    assert zero_sum.domain == A.domain
    assert zero_sum.codomain == A.codomain

    unsimplified = sc.SumLinOp((nested, Z))
    assert np.allclose(simplified.apply(x), unsimplified.apply(x))
    assert np.allclose(simplified.rapply(y), unsimplified.rapply(y))


def test_scaling_factory_simplifies_zero_one_and_nested_scaling():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    A = _op([[1.0, 2.0], [3.0, 4.0]], (2,), (2,), ctx)
    x = ctx.asarray([2.0, -1.0])
    y = ctx.asarray([1.0, 3.0])
    dense = np.array([[1.0, 2.0], [3.0, 4.0]])

    zero = 0 * A
    unit = 1 * A
    nested = 2 * (3 * A)

    assert isinstance(zero, sc.ZeroLinOp)
    assert unit is A
    assert isinstance(nested, sc.ScaledLinOp)
    assert nested.scalar == 6
    assert nested.op is A
    assert np.allclose(zero.apply(x), np.zeros(2))
    assert np.allclose(zero.rapply(y), np.zeros(2))
    assert np.allclose(nested.apply(x), 6 * dense @ np.asarray(x))
    assert np.allclose(nested.rapply(y), 6 * dense.T @ np.asarray(y))


def test_composition_factory_simplifies_identity_and_zero_factors():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    A = _op([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], (2,), (3,), ctx)
    id_domain = sc.IdentityLinOp(A.domain, ctx)
    id_codomain = sc.IdentityLinOp(A.codomain, ctx)
    left_zero = sc.ZeroLinOp(A.codomain, sc.VectorSpace((4,), ctx), ctx)
    right_zero = sc.ZeroLinOp(sc.VectorSpace((5,), ctx), A.domain, ctx)
    x = ctx.asarray([7.0, 8.0])
    y = ctx.asarray([1.0, -1.0, 2.0])
    dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert A @ id_domain is A
    assert id_codomain @ A is A

    left_simplified = left_zero @ A
    right_simplified = A @ right_zero

    assert isinstance(left_simplified, sc.ZeroLinOp)
    assert left_simplified.domain == A.domain
    assert left_simplified.codomain == left_zero.codomain
    assert isinstance(right_simplified, sc.ZeroLinOp)
    assert right_simplified.domain == right_zero.domain
    assert right_simplified.codomain == A.codomain

    unsimplified_left = sc.ComposedLinOp(left_zero, A)
    assert np.allclose((A @ id_domain).apply(x), dense @ np.asarray(x))
    assert np.allclose((id_codomain @ A).rapply(y), dense.T @ np.asarray(y))
    assert np.allclose(left_simplified.apply(x), unsimplified_left.apply(x))
    assert np.allclose(left_simplified.rapply(ctx.asarray([1.0, 2.0, 3.0, 4.0])), np.zeros(2))
