# Follow-up task: remove the deprecated `eigh` after the `spectrum` migration

This runs **after** the Jordan-algebra `spectrum` task has shipped and been through one release with `eigh` as a deprecated alias. Do not run it in the same release as the migration — the deprecation needs at least one published version of warning before removal, so users have a window to migrate.

**Precondition.** The `spectrum` / `spectral_decompose` / `from_spectrum` contract is in place on `VectorSpace`, `HermitianSpace`, `ProductSpace` (and `StackedSpace` if present); `eigh` currently exists only as a thin alias emitting `DeprecationWarning`; one release has gone out with that alias. If any of these is false, STOP — it is too early to remove `eigh`.

## Phase 0 — Confirm SpaceCore no longer depends on `eigh`

**Do.**
1. `grep -rn "\.eigh(\|def eigh\|eigh =" spacecore/ tests/` — every remaining hit must be either the alias definition itself or a test asserting the deprecation. No internal *functional* caller should remain. The migration task should already have routed `psd_proj`, solvers, and spectral functions to `spectrum` / `spectral_decompose`.
2. Distinguish `space.eigh` from `ops.eigh`:
   - `space.eigh` is the deprecated public-space alias being removed.
   - `ops.eigh` is the backend primitive used by `HermitianSpace.spectral_decompose`; **keep it**.
   Do not touch `ops.eigh` anywhere.

**Acceptance.** The only `space.eigh` references are the alias and its deprecation test; `ops.eigh` references are untouched.

## Phase 1 — Remove the alias

**Do.**
1. Delete the `eigh` method/alias from `VectorSpace`, `HermitianSpace`, `ProductSpace` (and `StackedSpace` if it has one) and from the `Space` base if a base alias was added.
2. Remove the warn-once machinery that existed solely for the `eigh` deprecation: module-level warned-set entries, warning helpers used only for `eigh`, and `warnings` / `DeprecationWarning` imports if now unused. Leave any shared warning helper if other deprecations use it.
3. Remove `eigh` from any `__all__` / public-API export lists and from the public-API test's expected surface.

**Acceptance.** `space.eigh` no longer exists; calling it raises `AttributeError`; `ops.eigh` still works.

## Phase 2 — Update tests and docs

**Do.**
1. Delete the test that asserted `eigh` emits `DeprecationWarning` and returns the same as `spectral_decompose`.
2. Keep all `spectrum` / `spectral_decompose` / `from_spectrum` tests — they are the replacement coverage and must stay green.
3. Update docs, docstrings, tutorials, and notebooks that referenced `space.eigh` to use:
   - `spectral_decompose` for `(evals, frame)`;
   - `spectrum` for eigenvalues only.
4. Search docs and notebooks:
   ```bash
   grep -rn "eigh" docs/ tutorials/ notebooks/
   ```
   Any remaining `eigh` hit must refer to `ops.eigh` or a backend primitive, not `space.eigh`.
5. CHANGELOG: under **Removed**, note that the deprecated `Space.eigh` alias is gone and that callers should use `spectral_decompose` for eigenvalues plus frame, or `spectrum` for eigenvalues only. Point to the migration release where the deprecation was introduced.

**Acceptance.** No `space.eigh` remains in code, tests, docs, tutorials, or notebooks; the spectrum test suite passes unchanged; CHANGELOG records the removal with the migration path.

## Phase 3 — Sweep for residual special-casing

**Do.** The original `eigh` was special-cased: defined on Hermitian spaces and raised on vector spaces. Confirm no vestige of that branching survives now that the alias is gone:
1. Grep for leftover `TypeError` / `NotImplementedError` messages mentioning `eigh`, for example the old `VectorSpace.eigh is not defined`, and delete them.
2. Confirm `psd_proj` and any matrix-function code path call `spectral_decompose` / `from_spectrum` directly, not the removed alias.
3. Confirm batch-safe reconstruction remains in place:
   - `symmetrize` uses `swapaxes(-1, -2)`, not `.T`;
   - `eig_to_dense` uses a batched reconstruction such as `einsum`, not a 2D-only `.T` reconstruction or materialized diagonal.

**Acceptance.** No `space.eigh`-related error strings or special-case branches remain; reconstruction is uniformly batch-safe.

## Constraints

- Removal only — no behavior change to `spectrum` / `spectral_decompose` / `from_spectrum`.
- Never touch `ops.eigh` or backend eigensolver implementations.
- Only `space.eigh` is in scope for removal.
- Full suite green after each phase.

## Report

- Grep results proving no `space.eigh` callers remain in `spacecore/`, `tests/`, docs, tutorials, or notebooks.
- Files where the alias and its deprecation machinery were removed.
- Confirmation that `ops.eigh` is untouched.
- Confirmation that the spectrum suite still passes.
- CHANGELOG **Removed** entry text.
