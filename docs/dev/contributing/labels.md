# GitHub labels

This page defines the intended SpaceCore issue and pull request label taxonomy. Maintainers can recreate these labels with:

```bash
scripts/setup_labels.sh
```

## Contributor-readiness labels

`good-first-issue` is only for tasks where the issue body gives the exact file, an example to follow, and a concrete done condition. A contributor should be able to complete it without asking a design question.

`help-wanted` marks well-specified work that may require one clarification.

`needs-design` means the design direction is not settled. Do not assign these issues to new contributors.

## Type labels

`documentation` marks documentation-only work.

`test` marks tests and test infrastructure work.

`fix` marks bug fixes or correctness fixes.

`bug` marks reports of incorrect behavior or regressions.

`feature` marks new features or API extensions.

## Component labels

`backend` covers `BackendOps`, `Context`, Array API integration, and optional backend behavior.

`space` covers spaces, elements, inner products, capabilities, and checks.

`linop` covers linear operators, adjoints, matrix-backed operators, and matrix-free operators.

`functional` covers functionals, gradients, and pull-backs.

`linalg` covers iterative algorithms and solver infrastructure.

`docs` covers the documentation site, tutorials, and developer documentation.

`ci` covers continuous integration, release checks, and automation.

`release` covers release preparation, changelog updates, tagging, and packaging.

## Maintainer rules

`good-first-issue` is restrictive. Do not use it for tasks involving unsettled design, new mathematical abstractions, dtype/field policy changes, batching redesign, metric-adjoint semantics, or block-operator design unless the issue body pins the exact file, expected behavior, and done condition.

`needs-design` means the task is intentionally not ready for implementation. New contributors should not be assigned these issues.

Every geometry-, adjoint-, spectral-, field-, or batching-related issue should contain a mathematical note.
