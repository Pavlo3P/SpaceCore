import subprocess
import sys

import numpy as np

from examples.weighted_tikhonov import (
    adjoint_diagnostics,
    dense_reference_solve,
    make_weighted_tikhonov_problem,
    objective_value,
    run_example,
    solve_with_spacecore,
)


def test_reference_solution_satisfies_dense_optimality_system():
    problem = make_weighted_tikhonov_problem(n=16, m=24, seed=3)
    x_ref, ref_diag = dense_reference_solve(problem)

    assert ref_diag.first_order_residual_norm <= 1e-10
    np.testing.assert_allclose(objective_value(problem, x_ref), ref_diag.objective, rtol=0.0, atol=0.0)


def test_spacecore_solution_matches_dense_reference():
    problem = make_weighted_tikhonov_problem(n=24, m=36, seed=4)
    x_ref, ref_diag = dense_reference_solve(problem)
    sc_solve = solve_with_spacecore(problem)

    relerr = np.linalg.norm(sc_solve.x - x_ref) / max(1.0, np.linalg.norm(x_ref))
    assert sc_solve.cg_converged
    assert relerr <= 1e-8
    np.testing.assert_allclose(sc_solve.diagnostics.objective, ref_diag.objective, rtol=1e-10, atol=1e-12)
    assert sc_solve.diagnostics.first_order_residual_norm <= 1e-8


def test_metric_adjoint_holds_and_coordinate_transpose_fails():
    problem = make_weighted_tikhonov_problem(n=12, m=18, seed=5)
    adjoint = adjoint_diagnostics(problem)

    assert adjoint.metric_identity_error <= 1e-12
    assert adjoint.wrong_transpose_identity_error >= 1e-2


def test_run_example_returns_expected_accuracy():
    result = run_example()

    assert result["spacecore"].cg_converged
    assert result["relative_solution_error"] <= 1e-8
    assert result["objective_difference"] <= 1e-10
    assert result["adjoint"].metric_identity_error <= 1e-12
    assert result["adjoint"].wrong_transpose_identity_error >= 1e-2


def test_example_script_runs_from_command_line():
    completed = subprocess.run(
        [sys.executable, "examples/weighted_tikhonov.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CG converged: True" in completed.stdout
    assert "wrong-transpose identity error" in completed.stdout
