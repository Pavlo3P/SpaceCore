# Task: Decompose SpaceCore 0.4.0 into contributor-ready GitHub issues

## Goal

Prepare the SpaceCore `0.4.0` work as a set of 15–20 well-scoped GitHub issues.

The `0.4.0` release is about test infrastructure, backend conformance, batching conformance, validation/check policy, dtype/field behavior, block LinOps, and TreeSpace/ProductSpace migration. This task does **not** implement those features. It only creates the issue set that contributors will use.

## Required outcome

Create at least **15 GitHub issues** for the `0.4.0` milestone. Each issue must be concrete enough that a contributor knows:

* which file or directory to start from;
* which existing pattern to follow;
* what command proves completion;
* whether mathematical care is needed;
* whether the issue is beginner-safe.

If GitHub CLI access is unavailable, create reproducible issue draft files in the repository instead.

## Required labels

Use the existing label taxonomy, or create missing labels if the repo has a script for it.

Use type labels:

* `test`
* `documentation`
* `feature`
* `fix`

Use component labels:

* `space`
* `linop`
* `functional`
* `linalg`
* `backend`

Use contribution labels:

* `good-first-issue`
* `help-wanted`
* `needs-design`

Do **not** apply `good-first-issue` to design-sensitive work such as dtype/field semantics, TreeSpace migration, block LinOp design, batching redesign, or metric-adjoint changes unless the issue body pins exact files and expected behavior.

## Issue template

Every issue must follow this structure:

````markdown
## Goal

[One short paragraph.]

## Files to inspect

- `path/to/file.py`
- `tests/path/to/test_file.py`

## Existing pattern to follow

[Point to an existing test, class, fixture, or documentation pattern.]

## Required work

- [ ] ...
- [ ] ...
- [ ] ...

## Done condition

```bash
pytest ...
````

## Mathematical note

[State the relevant invariant, or write "None beyond existing behavior."]

## Prerequisite level

[None / Basic NumPy / Linear algebra / Functional analysis / Backend API knowledge]

## Suggested labels

`test`, `space`, `good-first-issue`

````

## Required issue set

Create issues covering the following areas.

### 1. Space test generators

Create at least two issues:

1. Generator for dense coordinate/vector spaces.
2. Generator for product or structured spaces, including current batching behavior where supported.

These should target existing behavior only. The dense-space generator can be `good-first-issue` if the file and pattern are obvious.

### 2. LinOp test generators

Create at least three issues:

1. Matrix-backed LinOp generator.
2. Matrix-free LinOp generator, including the user-supplied `rapply` contract.
3. Diagonal/identity/zero/composed/summed/scaled LinOp generator coverage.

Do not require new LinOp features in these issues. They should test the existing public contracts.

### 3. Functional test generators

Create at least one issue for reusable functional fixtures.

It should cover:

- domain ownership;
- value evaluation;
- gradients where implemented;
- comparison with direct reference implementations.

### 4. Linalg test generators

Create at least one issue for iterative solver test fixtures.

It should cover:

- geometry-aware solver behavior;
- comparison with direct/backend references;
- residual or optimality checks;
- backend-independent fixture structure.

### 5. Backend conformance

Create at least three issues:

1. `NumpyOps` versus NumPy behavior.
2. JAX/Torch/CuPy versus `NumpyOps` behavior, with optional-backend skipping.
3. dtype, scalar-field, batching, conversion, and JIT-specific conformance cases.

These issues should be careful about optional dependencies. Missing JAX/Torch/CuPy must skip tests, not fail collection.

### 6. Batching conformance

Create at least one issue dedicated to current batching behavior.

It should test:

- batched and unbatched public contracts;
- `vapply` / `rvapply` behavior where present;
- consistency between batched application and row-wise scalar application;
- no redesign unless a defect is found.

Suggested label: `test`, not `needs-design`, unless the issue asks for redesign.

### 7. Check-policy migration

Create at least one issue for replacing Boolean check flags with validation levels.

It should cover the intended levels:

- `none`
- `cheap`
- `standard`
- `strict`

This issue is design-sensitive and should use `needs-design` unless the implementation details are already pinned by an ADR.

### 8. Dtype/field behavior

Create at least one issue for `Space.field` and dtype membership behavior.

It should cover:

- real versus complex scalar field;
- `Context.dtype` as representation default;
- complex-to-real narrowing refusal;
- backend-specific dtype behavior.

This issue must **not** be marked `good-first-issue`.

### 9. Block LinOps

Create at least two issues:

1. `BlockDiagonalLinOp`.
2. `BlockMatrixLinOp`.

Each issue must state explicitly that these are direct-product operators, not tensor-product operators.

Include the invariant:

```text
For product spaces with component inner products, the adjoint of a block matrix is the transposed block grid of component metric adjoints.
````

Use `needs-design` unless the implementation is fully specified.

### 10. TreeSpace and ProductSpace migration

Create at least two issues:

1. Introduce `TreeSpace` / `TreeElement` skeleton and tests.
2. Migrate or adapt `ProductSpace` behavior onto tree structure.

These are design-sensitive. Do not mark them as `good-first-issue`.

## Draft storage fallback

If GitHub issue creation is not possible, create:

```text
docs/dev/issues/0.4.0/
```

with one Markdown file per issue, for example:

```text
001-space-generator-dense.md
002-space-generator-product.md
003-linop-generator-matrix-backed.md
...
```

Also create:

```text
docs/dev/issues/0.4.0/README.md
```

The README must list all issue drafts in recommended implementation order and show suggested labels.

## If GitHub CLI is available

After creating drafts, create the issues with `gh issue create`.

Use a consistent title prefix:

```text
0.4.0:
```

Example:

```bash
gh issue create \
  --title "0.4.0: add dense space test generator" \
  --body-file docs/dev/issues/0.4.0/001-space-generator-dense.md \
  --label test \
  --label space \
  --label good-first-issue
```

Assign all issues to the `0.4.0` milestone if it exists. If it does not exist, create a note in `docs/dev/current.md` saying that milestone assignment remains manual.

## Update current development docs

Update `docs/dev/current.md` so that the `Next` section links to the `0.4.0` issue set.

If issues were created on GitHub, link to the milestone or issue query.

If only draft files were created, link to:

```text
docs/dev/issues/0.4.0/README.md
```

## Verification

Run:

```bash
pytest --co -q
ruff check docs/dev/issues/0.4.0 docs/dev/current.md
```

If Markdown linting is available, run it on the issue draft directory.

## Acceptance criteria

This task is complete when:

* at least 15 `0.4.0` issues or issue drafts exist;
* every issue follows the contribution-task structure;
* space, LinOp, functional, linalg, backend, batching, check policy, dtype/field, block LinOp, and TreeSpace work are represented;
* beginner-safe issues are labeled conservatively;
* design-sensitive issues are not mislabeled as `good-first-issue`;
* the `0.4.0` issue set is discoverable from `docs/dev/current.md`;
* optional-backend issues mention skip behavior rather than hard dependency assumptions.
