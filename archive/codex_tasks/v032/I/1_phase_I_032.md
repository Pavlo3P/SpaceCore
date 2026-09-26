# Codex Task: Complete the SpaceCore 0.3.2 Release-Candidate Validation

## Goal

Complete the final release-candidate validation for SpaceCore `0.3.2`.

This task is not about adding new features. It is a release-readiness pass: run the canonical validation script, run the extra `0.3.2` checks, fix small release-blocking issues found during validation, and record any explicitly deferred non-blocking items.

## Required starting assumptions

Work from the default branch after all accepted `0.3.2` contributor-infrastructure work has been merged.

Do not validate a side branch as release-ready unless the branch is intended to be merged before tagging.

The repository should already contain:

* `scripts/verify_release_candidate.sh`;
* contributor docs under `docs/dev/contributing/`;
* ADRs under `docs/dev/adr/`;
* release notes or changelog entries for `0.3.2`;
* matrix-free adjoint regression tests;
* GitHub issue and PR templates;
* 0.4.0 GitHub issue decomposition.

## Main validation

Run the canonical release-candidate script:

```bash
bash scripts/verify_release_candidate.sh
```

Treat this script as the source of truth for the general release gate. It should cover, at minimum:

* install;
* full tests;
* Ruff;
* docstring/API audits;
* strict documentation build;
* package build;
* twine check;
* clean install or smoke verification;
* active notebook execution if configured.

If the script fails, fix the failure rather than bypassing it. Only mark a check as informational if the script and release docs explicitly say it is informational.

## Minimal-install validation

Verify that a clean checkout works under the documented minimal development install.

Run something equivalent to:

```bash
python -m pip install -e ".[dev]"
pytest --co -q
pytest tests/ -q
```

Expected behavior:

* no collection errors;
* NumPy-backed tests pass;
* optional backend tests for JAX, Torch, or CuPy skip cleanly when those packages are absent;
* no module-level fixture or parameter construction imports optional backends unconditionally.

If optional-backend tests fail during collection, add the appropriate skip guards or lazy context construction.

## Matrix-free adjoint validation

Run the targeted tests for the `MatrixFreeLinOp` correctness fix.

The tests must cover:

* non-Euclidean domain and codomain spaces;
* `apply(x)` calls the user-supplied forward callable directly;
* `rapply(y)` calls the user-supplied adjoint callable directly;
* `.H.apply(y)` delegates to the supplied `rapply(y)`;
* `.H.rapply(x)` delegates to the supplied forward callable;
* `.H.H` behaves like the original operator;
* batched `vapply` rows agree with rowwise `apply`;
* batched `rvapply` rows agree with rowwise `rapply`;
* the adjoint identity holds:

```text
<Ax, y>_Y = <x, A♯y>_X
```

Use the existing targeted test names if they already exist. Otherwise, locate them with:

```bash
pytest tests/ -k "matrix_free or MatrixFree or adjoint" -q
```

Do not weaken matrix-backed adjoint tests. Matrix-backed operators should still use the metric-adjoint formula:

```text
A♯ = R_X^{-1} A† R_Y
```

## Batching documentation check

Review the documentation and ADRs for batching.

Ensure they state that batching is current supported behavior, not removed behavior.

Check at least:

* `docs/dev/adr/006_current_batching_model.md`;
* contributor architecture docs;
* release notes;
* any public docs mentioning `vapply`, `rvapply`, batched application, or `_batching`.

The documentation should describe:

* what batching paths currently exist;
* which abstractions own batching behavior;
* how batched `apply` / `rapply` relate to scalar `apply` / `rapply`;
* known limitations;
* what 0.4.0 test generators must cover.

Remove stale language about “batch removal” unless it is clearly historical and not presented as the current design.

## Contributor-link validation

Check that contributor-facing links resolve.

At minimum, inspect:

* root `CONTRIBUTING.md`;
* `docs/dev/contributing/setup.md`;
* `docs/dev/contributing/architecture.md`;
* `docs/dev/contributing/prerequisites.md`;
* `docs/dev/contributing/process.md`;
* `docs/dev/current.md`;
* `docs/dev/vision.md`;
* `docs/dev/adr/README.md`.

Run any existing docs link checker if available. If no link checker exists, manually check relative links touched by this release.

Fix broken links, stale install commands, stale ADR numbers, and links to files that do not exist.

## ADR index validation

Check `docs/dev/adr/README.md`.

Required release-gating ADRs must exist and be indexed:

* `004_inner_product_and_geometry`;
* `006_current_batching_model`;
* `009_metric_adjoint`;
* `014_check_policy`;
* `015_dtype_default_vs_scalar_field`.

All written ADR files must appear in the index.

If a non-gating ADR is deferred, record the deferral explicitly in `docs/dev/current.md` with:

* ADR number;
* intended topic;
* reason for deferral;
* expected follow-up release, usually `0.3.3`.

Do not leave unindexed ADR files or dangling ADR references.

## GitHub template validation

Verify that these files exist and render as usable Markdown:

```text
.github/PULL_REQUEST_TEMPLATE.md
.github/ISSUE_TEMPLATE/bug_report.md
.github/ISSUE_TEMPLATE/feature_request.md
.github/ISSUE_TEMPLATE/contribution_task.md
```

The PR template must include checklist items for:

* tests passing;
* docstring added or updated;
* changelog entry under `[Unreleased]`;
* mathematical invariant statement for changes touching geometry, adjoints, spectral methods, fields, or batching.

The contribution-task issue template must include:

* file to touch;
* example to follow;
* done condition;
* mathematical note;
* prerequisite level.

## GitHub label and issue validation

Verify that the label taxonomy is reproducible from the repository.

Expected file:

```text
scripts/setup_labels.sh
```

The script should create or update labels including:

* `good-first-issue`;
* `help-wanted`;
* `needs-design`;
* `documentation`;
* `test`;
* `fix`;
* `feature`;
* `space`;
* `linop`;
* `functional`;
* `linalg`;
* `backend`.

If GitHub CLI is available and authenticated, run the script or verify equivalent label state with `gh`.

Then verify that the `0.4.0` issue decomposition exists:

* at least 15 issues;
* issues use the contribution-task structure;
* beginner-safe issues are labeled `good-first-issue`;
* design-sensitive issues are not mislabeled as beginner-safe;
* dtype, adjoint, block-operator, tree-space, and batching-redesign issues are labeled conservatively.

If live GitHub access is unavailable in the environment, document exactly what was not verified and what command should be run by the maintainer.

## Default-branch validation

Confirm that accepted contributor infrastructure is present on the default branch, not only on a temporary working branch.

Check:

```bash
git branch --show-current
git status --short
git log --oneline -n 5
```

Expected:

* current branch is the intended release branch, usually `main`;
* no uncommitted release-critical files;
* contributor docs, ADRs, templates, label script, and current-state docs are committed.

If files are uncommitted, commit them before release validation is considered complete.

## Changelog and release-note finalization

Update the changelog/release notes for `0.3.2`.

The entry must mention:

* contributor-infrastructure work;
* ADR architecture memory;
* matrix-free adjoint correctness fix;
* minimal-install stabilization;
* current batching documentation;
* 0.4.0 issue decomposition;
* known limitations or deferred ADRs.

At tag time, add:

* release date;
* release link;
* version comparison link if the changelog uses link blocks.

Do not claim checks that are not actually performed by `scripts/verify_release_candidate.sh`.

## Final acceptance criteria

The task is complete when:

* `bash scripts/verify_release_candidate.sh` passes;
* minimal `.[dev]` install collects and runs tests without optional-backend collection errors;
* matrix-free non-Euclidean adjoint tests pass, including `.H` and batched/scalar consistency;
* batching docs describe current behavior accurately;
* contributor links are valid;
* release-gating ADRs exist and are indexed;
* deferred ADRs, if any, are recorded in `docs/dev/current.md`;
* GitHub templates exist and render correctly;
* label taxonomy is reproducible from `scripts/setup_labels.sh`;
* at least 15 labeled `0.4.0` issues exist or the exact maintainer-side verification command is recorded;
* contributor infrastructure is committed on the default branch;
* changelog/release notes are accurate;
* the working tree is clean except for intentional release-version changes.
