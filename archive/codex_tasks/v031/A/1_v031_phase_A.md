Task: complete SpaceCore 0.3.1 Phase A — immediate correctness and migration cleanup.

Scope:
Fix only repository-local correctness, README, migration, and stale-documentation issues. Do not perform downstream project audits or integrations.

Required changes:

1. Fix the README conjugate-gradient example.

   * The example matrix must be explicitly Hermitian positive-definite.
   * The example must execute successfully.

2. Fix or remove the `self.ops.fft` example.

   * Use only existing public APIs.
   * Remove the example if no valid replacement exists.

3. Correct the weighted-inner-product example.

   * Riesz maps must be mathematically correct.
   * `is_euclidean` claims must match the implementation.

4. Remove stale batching references.

   * Remove unintended references to `BatchSpace`.
   * Remove stale `Space.batch` and `batch_space=` references.
   * Current docs should describe `vapply`/vectorization instead.

5. Remove stale policy documentation.

   * Removed context-policy APIs must not appear as current APIs.
   * Removed dtype-policy APIs must not appear as current APIs.

6. Repair the broken `CONTRIBUTING.md` link.

   * Either add a minimal placeholder file or remove the link.

7. Complete `0.3.0` migration notes.

   * Include batching removal.
   * Include context/dtype policy removals.
   * Include release date and release link if available.

8. Decide whether `ProductLinOp` should be top-level exported.

   * Record the decision in a short comment, changelog note, or docs note.
   * If exported, add the import and a minimal test.

Validation:
Run formatting, linting, tests relevant to touched files, README examples if possible, and docs build if docs are changed.
