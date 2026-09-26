Implement SpaceCore 0.4.0 Phase 2: check-policy implementation.

Goal:
Replace the Boolean `enable_checks` model with an explicit check policy that supports four levels:

* `none`
* `cheap`
* `standard`
* `strict`

This task implements Phase B only. Do not implement dtype/field, TreeSpace, block LinOps, or generator infrastructure except where a small local test helper is needed for this phase.

Inputs:

* `task_list.md`
* `GRAPH_REPORT.md`
* `graph.json`

Use `GRAPH_REPORT.md` first to navigate the repository and save token/context usage. Prefer the graph report’s community summaries and god-node list before broad repository searches. Use `graph.json` only when exact file/function locations are needed. Verify every inferred graph relationship against actual source code before changing behavior.

Important graph-navigation hints:

* Start from the checks communities: `space/checks/*`, `_checks.py`, `_batching.py`.
* Inspect `Context`, `ContextBound`, `Space`, `LinOp`, `Functional`, and linalg validation paths.
* Pay attention to the reported connection `checked_method()` → `_check_batched()` and `_check_batched()` → `_run_checks()`.
* Treat inferred graph edges as hints, not truth.

Tasks:

1. Design the public check-policy API.

   * Prefer a simple `check_level` API unless the existing architecture clearly favors a `CheckPolicy` object.
   * Document the chosen public spelling and why it is compatible with current constructors.

2. Add a check-level type.

   * Represent `none`, `cheap`, `standard`, and `strict` in code.
   * Use a stable public type, for example a `Literal` alias, enum, or small frozen object.
   * Keep the API easy to type-check.

3. Implement level semantics.

   * `none`: skip optional runtime checks where safe; do not skip checks required to prevent internal corruption.
   * `cheap`: include shape, dtype, backend/context, tree/product structure, arity, domain, and codomain checks.
   * `standard`: include cheap checks plus linear or near-linear membership checks.
   * `strict`: include standard checks plus expensive PSD, metric, adjoint, spectral, and consistency checks.

4. Migrate spaces.

   * Replace public constructor use of `enable_checks` with the new policy.
   * Preserve current behavior through a compatibility path where reasonable.
   * Ensure membership validation dispatches by check level.

5. Migrate LinOps.

   * Update LinOp constructors and validation paths to use check policy.
   * Ensure `apply`, `rapply`, batched application, `.H`, composition, sum, scaling, identity, zero, dense, diagonal, sparse, product/block, and matrix-free paths remain coherent.
   * Matrix-free non-Euclidean adjoint behavior must not regress.

6. Migrate functionals.

   * Update domain validation and pull-back/composition validation to use check policy.
   * Keep scalar-output shape checks intact.

7. Migrate linalg validation.

   * Apply check policy to solver precondition checks where relevant.
   * Do not weaken essential solver invariants such as square-operator checks for CG-like algorithms.

8. Add migration path.

   * `enable_checks=True` should map to a documented default level, likely `standard`.
   * `enable_checks=False` should map to `none`.
   * If both `enable_checks` and `check_level` are passed, raise a clear error.
   * Add deprecation notes if appropriate.

9. Add tests.

   * Tests must show that each level catches only intended categories of errors.
   * Include spaces, LinOps, functionals, linalg, batching, and ProductSpace cases.
   * Add regression tests for `checked_method()` and batched validation.
   * Include at least one strict-only test, for example Hermitian/PSD/spectral/adjoint validation.

10. Add docs.

* Document each check level by cost and mathematical strength.
* Explain migration from `enable_checks`.
* Add examples showing how users should choose levels:

  * development: `strict` or `standard`
  * performance-sensitive trusted code: `cheap` or `none`
  * user-facing libraries: usually `standard`

Acceptance criteria:

* Public constructors use `check_level` or the chosen equivalent.
* `none`, `cheap`, `standard`, and `strict` are implemented and tested.
* Legacy `enable_checks=True/False` has a clear migration path.
* Spaces, LinOps, functionals, and linalg validation use the new policy where relevant.
* Documentation explains cost and mathematical strength of each level.
* Existing public behavior is preserved unless intentionally migrated.
* `pytest tests/ -x -q` passes.
* `ruff check .` passes.
* Any incomplete migration is recorded in `docs/dev/current.md` with a narrow follow-up.
