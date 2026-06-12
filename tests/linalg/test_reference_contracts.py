from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc
from tests._helpers import to_numpy


def _ctx(dtype=np.float64, check_level="standard"):
    return sc.Context(sc.NumpyOps(), dtype=dtype, check_level=check_level)


@pytest.mark.parametrize("operator_kind", ["dense", "diagonal", "matrix-free"])
def test_cg_reference_cases_match_direct_solve(operator_kind):
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((3,), ctx)
    matrix = np.asarray(
        [[6.0, 1.0, 0.5], [1.0, 5.0, -0.25], [0.5, -0.25, 4.0]],
        dtype=np.float64,
    )
    if operator_kind == "diagonal":
        matrix = np.diag([2.0, 5.0, 11.0])
        operator = sc.DiagonalLinOp(ctx.asarray(np.diag(matrix)), space, ctx)
    elif operator_kind == "matrix-free":
        matrix_backend = ctx.asarray(matrix)
        operator = sc.MatrixFreeLinOp(
            lambda x: matrix_backend @ x,
            lambda y: matrix_backend.T @ y,
            space,
            space,
            ctx,
        )
    else:
        operator = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)
    b = ctx.asarray([1.0, -2.0, 3.0])

    result = sc.cg(operator, b, tol=1e-12, maxiter=8, check_every=1)
    expected = np.linalg.solve(matrix, to_numpy(b))
    residual = to_numpy(operator.apply(result.x) - b)

    np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)
    assert bool(to_numpy(result.converged))
    assert 0 < int(to_numpy(result.num_iters)) <= 3
    assert float(to_numpy(result.residual_norm)) <= 1e-10


def test_cg_tolerance_and_iteration_status_are_observable():
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((2,), ctx)
    operator = sc.DiagonalLinOp(ctx.asarray([2.0, 5.0]), space, ctx)
    b = ctx.asarray([1.0, 1.0])

    stopped = sc.cg(operator, b, tol=0.0, atol=0.0, maxiter=0)
    converged = sc.cg(operator, b, tol=1e-12, maxiter=2, check_every=1)

    assert not bool(to_numpy(stopped.converged))
    assert int(to_numpy(stopped.num_iters)) == 0
    assert bool(to_numpy(converged.converged))
    assert int(to_numpy(converged.num_iters)) <= 2


@pytest.mark.parametrize(
    "matrix,b",
    [
        (
            np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]),
            np.asarray([1.0, 2.0, 2.5, -0.5]),
        ),
        (
            np.asarray([[1.0, 0.0, 1.0], [0.0, 2.0, -1.0]]),
            np.asarray([2.0, -1.0]),
        ),
        (
            np.asarray([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]),
            np.asarray([3.0, -4.0, 1.0, 0.25]),
        ),
    ],
    ids=["overdetermined", "underdetermined", "rectangular-diagonal"],
)
def test_lsqr_reference_cases_match_numpy_lstsq(matrix, b):
    ctx = _ctx()
    domain = sc.DenseCoordinateSpace((matrix.shape[1],), ctx)
    codomain = sc.DenseCoordinateSpace((matrix.shape[0],), ctx)
    operator = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    b_backend = ctx.asarray(b)

    result = sc.lsqr(operator, b_backend, tol=1e-12, maxiter=20, check_every=1)
    expected, *_ = np.linalg.lstsq(matrix, b, rcond=None)
    residual = operator.apply(result.x) - b_backend
    normal_residual = operator.H.apply(residual)

    np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        to_numpy(result.residual_norm),
        to_numpy(codomain.norm(residual)),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        to_numpy(result.normal_residual_norm),
        to_numpy(domain.norm(normal_residual)),
        rtol=1e-10,
        atol=1e-10,
    )
    assert bool(to_numpy(result.converged))


def test_lsqr_uses_metric_adjoint_on_weighted_spaces():
    ctx = _ctx()
    domain_weights = ctx.asarray([2.0, 7.0])
    codomain_weights = ctx.asarray([3.0, 5.0, 11.0])
    domain = sc.DenseCoordinateSpace(
        (2,), ctx, geometry=sc.WeightedInnerProduct(domain_weights)
    )
    codomain = sc.DenseCoordinateSpace(
        (3,), ctx, geometry=sc.WeightedInnerProduct(codomain_weights)
    )
    matrix = np.asarray([[1.0, 0.5], [-0.25, 2.0], [1.5, -1.0]])
    operator = sc.DenseLinOp(ctx.asarray(matrix), domain, codomain, ctx)
    b = ctx.asarray([1.0, -2.0, 0.75])

    result = sc.lsqr(operator, b, tol=1e-12, maxiter=10, check_every=1)
    weighted_normal = matrix.T @ np.diag(to_numpy(codomain_weights))
    expected = np.linalg.solve(weighted_normal @ matrix, weighted_normal @ to_numpy(b))
    residual = operator.apply(result.x) - b
    metric_normal_residual = operator.H.apply(residual)
    coordinate_transpose_residual = matrix.T @ to_numpy(residual)
    x_probe = ctx.asarray([0.5, -1.25])
    y_probe = ctx.asarray([1.0, -0.5, 2.0])

    np.testing.assert_allclose(to_numpy(result.x), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(to_numpy(metric_normal_residual), 0.0, atol=1e-9)
    assert not np.allclose(
        to_numpy(operator.rapply(y_probe)), matrix.T @ to_numpy(y_probe)
    )
    np.testing.assert_allclose(
        to_numpy(codomain.inner(operator.apply(x_probe), y_probe)),
        to_numpy(domain.inner(x_probe, operator.rapply(y_probe))),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.linalg.norm(coordinate_transpose_residual) > 1e-3
    assert bool(to_numpy(result.converged))


def test_lanczos_matrix_free_reference_has_normalized_ritz_vector():
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((3,), ctx)
    matrix = np.asarray([[4.0, 1.0, 0.5], [1.0, 3.0, -0.25], [0.5, -0.25, 2.0]])
    matrix_backend = ctx.asarray(matrix)
    operator = sc.MatrixFreeLinOp(
        lambda x: matrix_backend @ x,
        lambda y: matrix_backend.T @ y,
        space,
        space,
        ctx,
    )

    result = sc.lanczos_smallest(
        operator, ctx.asarray([1.0, -0.5, 0.75]), max_iter=3, tol=1e-12, check_every=1
    )
    expected = np.linalg.eigvalsh(matrix)[0]
    residual = operator.apply(result.eigenvector) - result.eigenvalue * result.eigenvector

    np.testing.assert_allclose(to_numpy(result.eigenvalue), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(to_numpy(space.norm(result.eigenvector)), 1.0, atol=1e-12)
    np.testing.assert_allclose(to_numpy(space.norm(residual)), 0.0, atol=1e-10)
    assert tuple(result.eigenvector.shape) == space.shape
    assert int(to_numpy(result.krylov_dim)) == 3
    assert bool(to_numpy(result.converged))


def test_lanczos_krylov_basis_is_orthonormal_before_breakdown():
    from spacecore.linalg._lanczos import _lanczos_basis_and_tridiag

    ctx = _ctx()
    space = sc.DenseCoordinateSpace((4,), ctx)
    matrix = ctx.asarray(
        [[5.0, 1.0, 0.0, 0.25], [1.0, 4.0, -0.5, 0.0],
         [0.0, -0.5, 3.0, 0.75], [0.25, 0.0, 0.75, 2.0]]
    )
    operator = sc.DenseLinOp(matrix, space, space, ctx)

    basis = _lanczos_basis_and_tridiag(
        operator,
        ctx.asarray([1.0, -0.5, 0.75, 1.5]),
        max_iter=4,
        tol=1e-12,
        real_dtype=ctx.ops.real_dtype(ctx.dtype),
        check_every=1,
    )
    krylov_dim = int(to_numpy(basis.krylov_dim))
    vectors = to_numpy(basis.V[:krylov_dim])

    np.testing.assert_allclose(vectors @ vectors.conj().T, np.eye(krylov_dim), atol=1e-11)


@pytest.mark.parametrize("operator_kind", ["dense", "matrix-free"])
def test_power_iteration_symmetric_reference_and_deterministic_start(operator_kind):
    ctx = _ctx()
    space = sc.DenseCoordinateSpace((3,), ctx)
    matrix = np.asarray([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
    if operator_kind == "matrix-free":
        matrix_backend = ctx.asarray(matrix)
        operator = sc.MatrixFreeLinOp(
            lambda x: matrix_backend @ x,
            lambda y: matrix_backend.T @ y,
            space,
            space,
            ctx,
        )
    else:
        operator = sc.DenseLinOp(ctx.asarray(matrix), space, space, ctx)
    x0 = ctx.asarray([1.0, -0.25, 0.5])

    first = sc.power_iteration(operator, x0=x0, tol=1e-10, maxiter=100)
    second = sc.power_iteration(operator, x0=x0, tol=1e-10, maxiter=100)
    expected = np.linalg.eigvalsh(matrix)[-1]
    residual = operator.apply(first.eigenvector) - first.eigenvalue * first.eigenvector

    np.testing.assert_allclose(to_numpy(first.eigenvalue), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(to_numpy(space.norm(residual)), 0.0, atol=1e-9)
    np.testing.assert_allclose(to_numpy(first.eigenvalue), to_numpy(second.eigenvalue))
    np.testing.assert_allclose(to_numpy(first.eigenvector), to_numpy(second.eigenvector))
    assert int(to_numpy(first.num_iters)) == int(to_numpy(second.num_iters))
    assert bool(to_numpy(first.converged))


@pytest.mark.parametrize("solver", ["cg", "lsqr", "lanczos", "power"])
def test_iterative_solvers_explicitly_reject_batched_inputs(solver):
    ctx = _ctx(check_level="standard")
    space = sc.DenseCoordinateSpace((2,), ctx)
    operator = sc.DiagonalLinOp(ctx.asarray([2.0, 5.0]), space, ctx)
    batch = ctx.asarray([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(TypeError, match="Expected shape"):
        if solver == "cg":
            sc.cg(operator, batch)
        elif solver == "lsqr":
            sc.lsqr(operator, batch)
        elif solver == "lanczos":
            sc.lanczos_smallest(operator, batch, max_iter=2)
        else:
            sc.power_iteration(operator, x0=batch)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_solver_vector_and_scalar_workspace_dtypes_follow_stage_one_policy(dtype):
    ctx = _ctx(dtype, check_level="standard")
    real_dtype = ctx.ops.real_dtype(ctx.dtype)
    space = sc.DenseCoordinateSpace((2,), ctx)
    diagonal = ctx.asarray(np.asarray([2.0, 5.0], dtype=dtype))
    operator = sc.DiagonalLinOp(diagonal, space, ctx)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        b = ctx.asarray(np.asarray([1.0 + 2.0j, -0.5 + 0.75j], dtype=dtype))
        x0 = ctx.asarray(np.asarray([1.0 + 0.5j, 1.0 - 0.25j], dtype=dtype))
    else:
        b = ctx.asarray(np.asarray([1.0, -0.5], dtype=dtype))
        x0 = ctx.asarray(np.asarray([1.0, 1.0], dtype=dtype))

    cg_result = sc.cg(operator, b, tol=1e-6, maxiter=4, check_every=1)
    lsqr_result = sc.lsqr(operator, b, tol=1e-6, maxiter=4, check_every=1)
    power_result = sc.power_iteration(operator, x0=x0, tol=1e-5, maxiter=40)
    lanczos_result = sc.lanczos_smallest(
        operator, x0, max_iter=2, tol=1e-6, check_every=1
    )

    vector_results = (
        cg_result.x,
        lsqr_result.x,
        power_result.eigenvector,
        lanczos_result.eigenvector,
    )
    scalar_results = (
        cg_result.residual_norm,
        lsqr_result.residual_norm,
        lsqr_result.normal_residual_norm,
        power_result.eigenvalue,
        power_result.residual_norm,
        lanczos_result.eigenvalue,
        lanczos_result.residual_norm,
    )
    assert all(ctx.ops.get_dtype(value) == ctx.dtype for value in vector_results)
    assert all(np.asarray(to_numpy(value)).dtype == np.dtype(real_dtype) for value in scalar_results)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        assert np.any(np.abs(np.imag(to_numpy(cg_result.x))) > 0)
        assert np.any(np.abs(np.imag(to_numpy(lsqr_result.x))) > 0)
