# Contributor process

## Branch naming

Use one of these branch prefixes:

```text
feature/<short-name>
fix/<short-name>
docs/<short-name>
test/<short-name>
```

Choose the prefix by the dominant intent of the change. Split unrelated work instead of hiding multiple intents in one branch.

### Branch and merge policy

Contributor infrastructure and documentation may be drafted on a feature branch.
They count toward a release only after they are merged into the default branch.
Contributor docs stranded on side branches do not satisfy the release gate.

## Before opening a PR

A PR should normally include:

- tests for changed behavior;
- updated docstrings if public behavior changed;
- a changelog entry under `[Unreleased]`;
- documentation updates if contributor-facing behavior changed;
- mathematical invariant statement for changes touching geometry, adjoints, spectral methods, scalar fields, or batching.

Run the relevant checks before requesting review:

```bash
pytest --co -q
pytest tests/ -x -q
ruff check .
```

Use focused tests first while developing, then run the normal gate before the PR is ready.

## Review order

Review checks correctness first, then style. Formatting, naming, and cleanup matter, but they do not compensate for an invalid mathematical contract.

For mathematical code, review should check:

- domain and codomain spaces;
- scalar field;
- inner-product assumptions;
- adjoint identity;
- validation level;
- backend behavior;
- batching behavior where applicable.

A PR touching optional backends should state whether NumPy behavior is unchanged and whether optional backend tests run or skip in a clean minimal install.

## Blocked PRs

Blocked PRs should be narrowed:

- document the blocker;
- split unrelated work;
- merge completed safe parts;
- open follow-up issues for design questions.

Do not keep a large PR open only because one design question is unresolved. If the safe subset has a clear contract and tests, split it out.

## Patch release policy

A patch release is appropriate for:

- regression fix;
- packaging fix;
- documentation fix that affects install/use;
- optional-backend import or skip fix.

Feature additions, broad refactors, and changed mathematical contracts should target a normal minor release unless maintainers explicitly decide otherwise.
