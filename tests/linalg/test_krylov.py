import importlib
import inspect

import numpy as np
import pytest

from tests._helpers import has_cupy, has_jax, has_torch, jax_real_dtype, to_numpy
from tests._helpers import torch_real_dtype


def _backend_params():
    return [
        pytest.param("numpy", np.float64, id="numpy"),
        pytest.param(
            "jax",
            jax_real_dtype(),
            marks=pytest.mark.skipif(not has_jax(), reason="jax is not installed"),
            id="jax",
        ),
        pytest.param(
            "torch",
            torch_real_dtype(),
            marks=pytest.mark.skipif(not has_torch(), reason="torch is not installed"),
            id="torch",
        ),
        pytest.param(
            "cupy",
            np.float64,
            marks=pytest.mark.skipif(not has_cupy(), reason="cupy is not installed"),
            id="cupy",
        ),
    ]


def _numpy_jax_params():
    return [
        pytest.param("numpy", np.float64, id="numpy"),
        pytest.param(
            "jax",
            jax_real_dtype(),
            marks=pytest.mark.skipif(not has_jax(), reason="jax is not installed"),
            id="jax",
        ),
    ]


def _ops_for_backend(name):
    sc = importlib.import_module("spacecore")
    if name == "numpy":
        return sc.NumpyOps()
    if name == "jax":
        return sc.JaxOps()
    if name == "torch":
        return sc.TorchOps()
    if name == "cupy":
        return sc.CuPyOps()
    raise ValueError(f"Unknown backend {name!r}.")


def _ctx(backend_name="numpy", dtype=np.float64):
    sc = importlib.import_module("spacecore")
    return sc.Context(_ops_for_backend(backend_name), dtype=dtype, enable_checks=False)


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_cg_solves_spd_system(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    space = sc.VectorSpace((2,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
    b = ctx.asarray([1.0, 2.0])

    result = sc.cg(A, b, tol=1e-7, maxiter=10)

    np.testing.assert_allclose(
        to_numpy(result.x),
        np.linalg.solve(np.array([[4.0, 1.0], [1.0, 3.0]]), np.array([1.0, 2.0])),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(to_numpy(A.apply(result.x)), to_numpy(b), rtol=1e-5, atol=1e-5)
    assert bool(to_numpy(result.converged))


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_lsqr_solves_rectangular_least_squares(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    A = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    b = ctx.asarray([1.0, 2.0, 4.0])

    result = sc.lsqr(A, b, tol=1e-7, maxiter=10)

    expected, *_ = np.linalg.lstsq(matrix, np.array([1.0, 2.0, 4.0]), rcond=None)
    np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(A.H.apply(A.apply(result.x) - b)), [0.0, 0.0], atol=1e-5)
    assert bool(to_numpy(result.converged))


def test_lsqr_works_with_matrix_free_linop_and_uses_rapply():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    matrix = ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    calls = {"rapply": 0}

    def apply(x):
        return matrix @ x

    def rapply(y):
        calls["rapply"] += 1
        return matrix.T @ y

    A = sc.MatrixFreeLinOp(apply, rapply, domain, codomain, ctx)
    b = ctx.asarray([1.0, 2.0, 3.0])

    result = sc.lsqr(A, b, tol=1e-8, maxiter=10)

    np.testing.assert_allclose(result.x, [1.0, 2.0], rtol=1e-6, atol=1e-6)
    assert calls["rapply"] > 0


def test_cg_solves_complex_hermitian_positive_definite_system():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.complex128)
    space = sc.VectorSpace((2,), ctx)
    matrix = np.array([[4.0, 1.0 + 1.0j], [1.0 - 1.0j, 3.0]], dtype=np.complex128)
    A = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)
    b = ctx.asarray([1.0 + 2.0j, 3.0 - 1.0j])

    result = sc.cg(A, b, tol=1e-10, maxiter=10)

    np.testing.assert_allclose(to_numpy(result.x), np.linalg.solve(matrix, to_numpy(b)), rtol=1e-8)
    np.testing.assert_allclose(to_numpy(A.apply(result.x)), to_numpy(b), rtol=1e-8, atol=1e-8)
    assert bool(to_numpy(result.converged))


def test_cg_float64_spd_reaches_residual_below_one_e_minus_ten():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)
    space = sc.VectorSpace((4,), ctx)
    matrix = np.array(
        [
            [6.0, 1.0, 0.5, 0.0],
            [1.0, 5.0, 0.0, 0.25],
            [0.5, 0.0, 4.0, 0.75],
            [0.0, 0.25, 0.75, 3.0],
        ],
        dtype=np.float64,
    )
    A = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)
    b = ctx.asarray([1.0, -2.0, 0.5, 3.0])

    result = sc.cg(A, b, tol=1e-13, maxiter=20, check_every=1)

    residual = space.norm(A.apply(result.x) - b)
    assert bool(to_numpy(result.converged))
    assert float(to_numpy(residual)) < 1e-10


def test_cg_regression_removes_sqrt_epsilon_residual_floor():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)
    space = sc.VectorSpace((2,), ctx)
    A = sc.DiagonalLinOp(ctx.asarray([1.0, 1.0e4]), space, ctx)
    b = ctx.asarray([1.0, 1.0e-12])

    result = sc.cg(A, b, tol=1e-13, maxiter=4, check_every=1)

    residual = space.norm(A.apply(result.x) - b)
    assert bool(to_numpy(result.converged))
    assert float(to_numpy(residual)) < 1e-10


def test_cg_final_iteration_refreshes_residual_with_sparse_checks():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.float64)
    space = sc.VectorSpace((2,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
    b = ctx.asarray([1.0, 2.0])

    result = sc.cg(A, b, tol=1e-12, maxiter=2, check_every=10)

    actual_residual = space.norm(A.apply(result.x) - b)
    assert bool(to_numpy(result.converged))
    np.testing.assert_allclose(to_numpy(result.residual_norm), to_numpy(actual_residual), atol=1e-14)


def test_lsqr_solves_complex_least_squares():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.complex128)
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    matrix = np.array(
        [[1.0 + 1.0j, 0.0], [0.0, 2.0 - 1.0j], [1.0, 1.0j]],
        dtype=np.complex128,
    )
    A = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    b = ctx.asarray([1.0 - 1.0j, 2.0 + 0.5j, 3.0j])

    result = sc.lsqr(A, b, tol=1e-10, maxiter=20)

    expected, *_ = np.linalg.lstsq(matrix, to_numpy(b), rcond=None)
    np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        to_numpy(A.H.apply(A.apply(result.x) - b)),
        np.zeros(2, dtype=np.complex128),
        atol=1e-7,
    )
    assert bool(to_numpy(result.converged))


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_power_iteration_estimates_dominant_eigenpair(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    space = sc.VectorSpace((2,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    x0 = ctx.asarray([1.0, 1.0])

    result = sc.power_iteration(A, x0=x0, tol=1e-5, maxiter=60)

    np.testing.assert_allclose(to_numpy(result.eigenvalue), 5.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.abs(to_numpy(result.eigenvector)),
        [0.0, 1.0],
        rtol=1e-4,
        atol=1e-4,
    )
    assert bool(to_numpy(result.converged))


def test_power_iteration_accepts_quadratic_form_hessian_action():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    q = sc.LinOpQuadraticForm(op, ctx=ctx)
    x0 = ctx.asarray([1.0, 1.0])

    op_result = sc.power_iteration(op, x0=x0, tol=1e-5, maxiter=60)
    q_result = sc.power_iteration(q, x0=x0, tol=1e-5, maxiter=60)

    np.testing.assert_allclose(to_numpy(q_result.eigenvalue), to_numpy(op_result.eigenvalue))
    np.testing.assert_allclose(
        np.abs(to_numpy(q_result.eigenvector)),
        np.abs(to_numpy(op_result.eigenvector)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_power_iteration_dispatches_quadratic_form_before_core(monkeypatch):
    sc = importlib.import_module("spacecore")
    power_mod = importlib.import_module("spacecore.linalg._power")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    q = sc.LinOpQuadraticForm(op, ctx=ctx)
    x0 = ctx.asarray([1.0, 0.0])
    captured = {}

    def fake_core(action, x, tol, maxiter, check_every):
        captured["action"] = action
        captured["x"] = x
        return ctx.asarray(0.0), x, ctx.asarray(True), 0, ctx.asarray(0.0)

    monkeypatch.setattr(power_mod, "_power_iteration_core", fake_core)
    result = power_mod.power_iteration(q, x0=x0, maxiter=1)

    assert result.eigenvector is x0
    assert isinstance(captured["action"], power_mod._SelfAdjointAction)
    assert captured["action"].domain == q.domain
    x = ctx.asarray([1.0, 2.0])
    np.testing.assert_allclose(captured["action"].apply(x), q.hess_apply(x))


def test_power_iteration_core_has_no_dispatch_logic():
    power_mod = importlib.import_module("spacecore.linalg._power")
    source = inspect.getsource(power_mod._power_iteration_core)

    assert "isinstance" not in source
    assert "hasattr" not in source
    assert "getattr" not in source
    assert "_SelfAdjointAction(" not in source
    assert "PowerIterationResult(" not in source


@pytest.mark.parametrize("backend_name,dtype", _backend_params())
def test_lanczos_smallest_approximates_smallest_eigenpair(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    initial = ctx.asarray([1.0, 1.0])

    result = sc.lanczos_smallest(
        op,
        initial,
        max_iter=2,
        tol=1e-8,
    )

    np.testing.assert_allclose(to_numpy(result.eigenvalue), 2.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.abs(to_numpy(result.eigenvector)),
        [1.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )
    assert bool(to_numpy(result.converged))
    np.testing.assert_allclose(to_numpy(result.residual_norm), 0.0, atol=1e-5)
    assert int(to_numpy(result.krylov_dim)) == 2


@pytest.mark.parametrize("backend_name,dtype", _numpy_jax_params())
def test_lanczos_smallest_uses_true_krylov_dim_after_delayed_breakdown(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    ctx = _ctx(backend_name, dtype)
    space = sc.VectorSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.5, 2.0, 3.0]), space, ctx)

    result = sc.lanczos_smallest(
        op,
        ctx.asarray([1.0, 1.0, 1.0]),
        max_iter=20,
        tol=1e-5,
        check_every=10,
    )

    assert int(to_numpy(result.krylov_dim)) <= 3
    np.testing.assert_allclose(to_numpy(result.eigenvalue), 1.5, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend_name,dtype", _numpy_jax_params())
def test_lanczos_basis_sentinel_masks_ghost_iterations_after_breakdown(backend_name, dtype):
    sc = importlib.import_module("spacecore")
    lanczos_mod = importlib.import_module("spacecore.linalg._lanczos")
    ctx = _ctx(backend_name, dtype)
    space = sc.VectorSpace((3,), ctx)
    op = sc.DiagonalLinOp(ctx.asarray([1.5, 2.0, 3.0]), space, ctx)

    basis = lanczos_mod._lanczos_basis_and_tridiag(
        op,
        ctx.asarray([1.0, 1.0, 1.0]),
        max_iter=20,
        tol=1e-5,
        real_dtype=ctx.ops.real_dtype(ctx.dtype),
        check_every=10,
    )

    krylov_dim = int(to_numpy(basis.krylov_dim))
    T_diag = np.diag(to_numpy(basis.T))
    assert krylov_dim == 3
    assert np.all(T_diag[krylov_dim:] > 3.0)
    np.testing.assert_allclose(np.linalg.eigvalsh(to_numpy(basis.T))[0], 1.5, rtol=1e-5, atol=1e-5)


def test_lanczos_smallest_returns_result_object():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    result = sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=2, tol=1e-8)

    assert isinstance(result, sc.LanczosResult)
    np.testing.assert_allclose(result.eigenvalue, 2.0)


def test_lanczos_smallest_uses_e0_for_zero_initial_vector():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    initial = ctx.asarray([0.0, 0.0])

    result = sc.lanczos_smallest(
        op,
        initial,
        max_iter=2,
        tol=1e-8,
    )

    np.testing.assert_allclose(result.eigenvalue, 2.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.eigenvector, [1.0, 0.0], rtol=1e-6, atol=1e-6)


def test_lanczos_smallest_rejects_invalid_max_iter():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((1,), ctx)
    op = sc.IdentityLinOp(space, ctx)

    with pytest.raises(ValueError, match="max_iter"):
        sc.lanczos_smallest(op, ctx.asarray([1.0]), max_iter=0)


def test_lanczos_smallest_rejects_structurally_non_hermitian_operator():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, 3.0]]), space, space, ctx)

    with pytest.raises(ValueError, match="Hermitian"):
        sc.lanczos_smallest(op, ctx.asarray([1.0, 1.0]), max_iter=2)


def test_lanczos_smallest_handles_eigenvalues_larger_than_1e10():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    matrix = np.diag([2.0e12, 3.0e12])
    op = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)

    result = sc.lanczos_smallest(
        op,
        ctx.asarray([1.0, 1.0]),
        max_iter=4,
        tol=1e-8,
    )

    np.testing.assert_allclose(to_numpy(result.eigenvalue), 2.0e12, rtol=1e-6)
    np.testing.assert_allclose(np.abs(to_numpy(result.eigenvector)), [1.0, 0.0], atol=1e-5)


def test_lanczos_smallest_handles_complex_hermitian_operator():
    sc = importlib.import_module("spacecore")
    ctx = _ctx(dtype=np.complex128)
    space = sc.VectorSpace((2,), ctx)
    matrix = np.array([[2.0, 1.0 + 2.0j], [1.0 - 2.0j, 5.0]], dtype=np.complex128)
    op = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)

    result = sc.lanczos_smallest(
        op,
        ctx.asarray([1.0 + 0.0j, 1.0j]),
        max_iter=2,
        tol=1e-10,
    )

    expected = np.linalg.eigvalsh(matrix)[0]
    np.testing.assert_allclose(to_numpy(result.eigenvalue), expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        to_numpy(op.apply(result.eigenvector)),
        to_numpy(result.eigenvalue) * to_numpy(result.eigenvector),
        rtol=1e-6,
        atol=1e-6,
    )


def test_lanczos_smallest_uses_domain_geometry_for_weighted_inner_product():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()

    class WeightedVectorSpace(sc.VectorSpace):
        def __init__(self, weights, ctx):
            weights = ctx.asarray(weights)
            super().__init__(tuple(weights.shape), ctx)
            self.weights = weights

        def inner(self, x, y):
            if self._enable_checks:
                self._check_member(x)
                self._check_member(y)
            return self.ops.vdot(x, self.weights * y)

        def _convert(self, new_ctx):
            return WeightedVectorSpace(new_ctx.asarray(self.weights), new_ctx)

    space = WeightedVectorSpace([1.0, 4.0], ctx)
    assert type(space) is not sc.VectorSpace

    matrix = np.array([[2.0, 1.0], [0.25, 0.75]])
    op = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)

    result = sc.lanczos_smallest(
        op,
        ctx.asarray([1.0, 1.0]),
        max_iter=2,
        tol=1e-12,
    )

    expected = np.min(np.linalg.eigvals(matrix).real)
    np.testing.assert_allclose(to_numpy(result.eigenvalue), expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        to_numpy(op.apply(result.eigenvector)),
        to_numpy(result.eigenvalue) * to_numpy(result.eigenvector),
        rtol=1e-6,
        atol=1e-6,
    )


def test_safe_inverse_nonneg_returns_reciprocal_for_positive_values_only():
    sc = importlib.import_module("spacecore")
    utils = importlib.import_module("spacecore.linalg._utils")
    ctx = _ctx()

    values = ctx.asarray([-2.0, 0.0, 4.0])

    np.testing.assert_allclose(to_numpy(utils.safe_inverse_nonneg(sc.NumpyOps(), values)), [0.0, 0.0, 0.25])


def test_iterative_solvers_poll_convergence_on_check_interval():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    space = sc.VectorSpace((2,), ctx)
    spd = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)
    rectangular = sc.DenseLinOp(
        ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        space,
        sc.VectorSpace((3,), ctx),
        ctx,
    )
    diagonal = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    cg_result = sc.cg(spd, ctx.asarray([1.0, 2.0]), maxiter=65)
    lsqr_result = sc.lsqr(rectangular, ctx.asarray([1.0, 2.0, 3.0]), maxiter=65)
    power_result = sc.power_iteration(diagonal, x0=ctx.asarray([1.0, 1.0]), maxiter=65)

    assert cg_result.num_iters < 64
    assert lsqr_result.num_iters == 64
    assert power_result.num_iters == 64
    np.testing.assert_allclose(cg_result.residual_norm, 0.0, atol=1e-12)
    np.testing.assert_allclose(lsqr_result.normal_residual_norm, 0.0, atol=1e-12)
    np.testing.assert_allclose(power_result.residual_norm, 0.0, atol=1e-12)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_cg_jit_compiles_with_operator_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[4.0, 1.0], [1.0, 3.0]]), space, space, ctx)

    solve = jax.jit(lambda A, b: sc.cg(A, b, maxiter=10).x)
    x = solve(op, ctx.asarray([1.0, 2.0]))

    np.testing.assert_allclose(to_numpy(x), [0.09090909, 0.63636364], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_lsqr_jit_compiles_with_operator_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)

    solve = jax.jit(lambda A, b: sc.lsqr(A, b, maxiter=10).x)
    x = solve(op, ctx.asarray([1.0, 2.0, 4.0]))

    np.testing.assert_allclose(to_numpy(x), [1.33333333, 2.33333333], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_power_iteration_jit_compiles_with_operator_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    run = jax.jit(lambda A, x: sc.power_iteration(A, x0=x, maxiter=60).eigenvalue)
    eigenvalue = run(op, ctx.asarray([1.0, 1.0]))

    np.testing.assert_allclose(to_numpy(eigenvalue), 5.0, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_power_iteration_jit_compiles_with_quadratic_form_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)
    q = sc.LinOpQuadraticForm(op, ctx=ctx)

    run = jax.jit(lambda quad, x: sc.power_iteration(quad, x0=x, maxiter=60).eigenvalue)
    eigenvalue = run(q, ctx.asarray([1.0, 1.0]))

    np.testing.assert_allclose(to_numpy(eigenvalue), 5.0, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not has_jax(), reason="jax is not installed")
def test_lanczos_smallest_jit_compiles_with_operator_argument():
    jax = pytest.importorskip("jax")
    sc = importlib.import_module("spacecore")
    ctx = _ctx("jax", jax_real_dtype())
    space = sc.VectorSpace((2,), ctx)
    op = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 5.0]]), space, space, ctx)

    def run(A, initial):
        result = sc.lanczos_smallest(
            A,
            initial,
            max_iter=2,
            tol=1e-8,
        )
        return result.eigenvalue, result.eigenvector

    eigenvalue, eigenvector = jax.jit(run)(op, ctx.asarray([1.0, 1.0]))

    np.testing.assert_allclose(to_numpy(eigenvalue), 2.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.abs(to_numpy(eigenvector)),
        [1.0, 0.0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_cg_and_power_iteration_reject_rectangular_operator():
    sc = importlib.import_module("spacecore")
    ctx = _ctx()
    domain = sc.VectorSpace((2,), ctx)
    codomain = sc.VectorSpace((3,), ctx)
    A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), domain, codomain, ctx)

    with pytest.raises(ValueError, match="square LinOp"):
        sc.cg(A, ctx.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="square LinOp"):
        sc.power_iteration(A)
    with pytest.raises(ValueError, match="square LinOp"):
        sc.lanczos_smallest(A, ctx.asarray([1.0, 2.0]))
