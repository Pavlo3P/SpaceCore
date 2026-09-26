Phase I implementation plan: Functional and linalg coverage.

Goal:
Add generated and reference-based tests for scalar-valued functionals and iterative linalg routines. This phase should not redesign linalg algorithms; it should make their expected behavior precise and tested.

Use only repository `.venv` commands:

* `.venv/bin/python -m pytest ...`
* `.venv/bin/python -m ruff check .`

Use `GRAPH_REPORT.md` first to navigate the repo and save context. Verify all graph-derived conclusions against source files. If the graph is stale, update or ignore stale ProductSpace references.

1. Preparation

Inspect:

* `spacecore/functional/`
* `spacecore/linalg/`
* `spacecore/linop/`
* `tests/generators/`
* `tests/functional/`
* `tests/linalg/`
* check-policy helpers
* dtype/field docs
* ADR-015

Confirm existing generator protocol from Phase F and existing space/LinOp generators from Phases G–H.

2. Functional generator

Implement a generator that produces scalar-valued maps (f : X \to \mathbb{R}) or (f : X \to \mathbb{C}) where appropriate.

Cover at least:

* linear functionals;
* quadratic forms;
* composed functionals (f \circ A);
* zero/simple constant cases if supported;
* Euclidean and non-Euclidean domains.

Each generated case should include:

* functional object;
* domain;
* deterministic input element;
* reference value;
* expected gradient if supported;
* expected pull-back if supported;
* check-level parameters.

3. Functional value tests

For each generated functional:

* evaluate on deterministic inputs;
* compare against direct reference formulas;
* test conversion where supported;
* include real and complex dtype cases if supported;
* include TreeSpace domains only if current functional implementation supports them.

4. Gradient reference tests

Add analytic or finite-difference gradient tests.

Preferred order:

* analytic references for linear functionals;
* analytic references for quadratic forms;
* finite-difference tests only where analytic formula is unavailable.

For (f : X \to \mathbb{R}), compare gradients in the correct space (X), respecting the inner product and Riesz map.

Non-Euclidean check:

* verify the gradient is the metric gradient, not accidentally the coordinate gradient.

5. Pull-back tests

For supported pull-backs (f \circ A):

* test value identity: `(f.pullback(A))(x) == f(A x)`;
* test gradient identity when implemented;
* test domain/codomain validation;
* test conversion behavior.

Use simple dense operators first, then diagonal/matrix-free/block operators if already available.

6. Functional validation-level tests

For check levels:

* `none`: skip optional domain checks where currently intended;
* `cheap`: catch shape, dtype, field, context, domain/codomain mismatch;
* `standard`: catch normal membership failures;
* `strict`: catch expensive consistency failures if applicable.

Do not invent new check semantics in this phase. Use Phase B policy.

7. CG reference tests

Add SPD reference tests for `cg`.

Cases:

* Euclidean dense SPD matrix;
* diagonal SPD matrix;
* matrix-free SPD operator;
* non-Euclidean inner-product space if supported.

Compare:

* solution residual;
* iteration status;
* convergence flag;
* tolerance behavior;
* dtype of solution/workspace where visible.

Reference:

* direct dense solve for small systems;
* known diagonal solution for diagonal systems.

8. LSQR reference tests

Add least-squares tests for `lsqr`.

Cases:

* full-rank rectangular system;
* overdetermined system;
* underdetermined system if behavior is documented;
* diagonal/sparse-compatible case if supported.

Compare:

* residual norm;
* normal-equation residual;
* direct NumPy least-squares reference;
* convergence status.

9. Lanczos reference tests

Add Lanczos tests on small self-adjoint operators.

Cases:

* diagonal Hermitian operator with known eigenvalues;
* small dense symmetric/Hermitian operator;
* matrix-free self-adjoint operator.

Check:

* Ritz values are within documented tolerance;
* returned vectors have correct shape;
* orthogonality is reasonable within tolerance;
* no ghost/duplicate iteration regression;
* complex Hermitian cases where supported.

10. Power-method reference tests

Add power-method tests.

Cases:

* diagonal operator with unique dominant eigenvalue;
* symmetric dense operator;
* negative dominant eigenvalue case if documented;
* matrix-free operator.

Check:

* dominant eigenvalue approximation;
* eigenvector residual;
* deterministic seed behavior;
* stopping criteria.

11. Non-Euclidean linalg usage

Add explicit tests that linalg routines respect space geometry.

At minimum:

* construct a non-Euclidean inner-product space (X);
* construct an operator whose metric adjoint differs from coordinate transpose;
* verify linalg uses the correct `rapply` / `.H` behavior;
* test the adjoint identity in linalg-relevant paths.

Do not silently Euclideanize the test reference.

12. Linalg batching

Determine per solver whether batching is supported.

For each solver:

* if supported, add batched tests comparing to a loop over unbatched calls;
* if unsupported, add explicit tests/documentation that batched input raises a clear error.

Do not redesign batching in this phase.

13. Solver workspace dtype tests

Follow ADR-015 Stage decision.

If Stage 2 is deferred:

* assert current documented dtype behavior;
* add tests proving no silent complex-to-real narrowing;
* record unresolved mixed-precision questions if missing.

If Stage 2 has landed:

* test that solver workspaces follow operand dtype where possible;
* test exact-dtype opt-in behavior if available.

14. Documentation

Add or update a short developer note:

* how functional generators are structured;
* which linalg routines have reference tests;
* what tolerances are used and why;
* which batching cases are unsupported;
* how non-Euclidean geometry is tested.

15. Acceptance criteria

* Functional generator exists.
* Functional value tests pass.
* Gradient reference tests pass.
* Pull-back tests pass where supported.
* Functional validation-level tests pass.
* CG reference tests pass.
* LSQR reference tests pass.
* Lanczos reference tests pass.
* Power-method reference tests pass.
* Non-Euclidean linalg behavior is tested.
* Linalg batching is tested or explicitly marked unsupported.
* Solver workspace dtype behavior follows ADR-015 decision.
* `.venv/bin/python -m pytest tests/ -x -q` passes.
* `.venv/bin/python -m ruff check .` passes.
* Any missing solver feature is recorded in `docs/dev/current.md` with a narrow follow-up.
