# Codex Task: Add GitHub contribution infrastructure for SpaceCore 0.3.2

## Goal

Add repository-level GitHub infrastructure so contributors can open useful PRs and issues, and maintainers can reproduce the intended label taxonomy from the repo.

This task should create:

* a pull request template;
* issue templates;
* a documented label taxonomy;
* a reproducible label setup script;
* maintainer rules for `good-first-issue` and `needs-design`.

Do not change library code.

## Files to create or update

Create or update:

```text
.github/PULL_REQUEST_TEMPLATE.md
.github/ISSUE_TEMPLATE/bug_report.md
.github/ISSUE_TEMPLATE/feature_request.md
.github/ISSUE_TEMPLATE/contribution_task.md
docs/dev/contributing/labels.md
scripts/setup_labels.sh
```

If `docs/dev/contributing/index.md` or a similar contributor-doc index exists, add `labels.md` to that index.

## Pull request template

Create `.github/PULL_REQUEST_TEMPLATE.md` with this structure:

```markdown
## What this PR does

[One sentence.]

## Checklist

- [ ] Tests pass (`pytest tests/ -x -q`)
- [ ] Docstring added or updated, if public behavior changed
- [ ] CHANGELOG entry added under `[Unreleased]`
- [ ] If this PR touches geometry, adjoints, spectral methods, scalar fields, or batching: describe the mathematical invariant being preserved or added below.

## Mathematical invariant

[Only required for geometry-, adjoint-, spectral-, field-, or batching-related PRs.]
```

The invariant item is important. It should force PR authors to explain mathematical correctness when touching spaces, inner products, adjoints, spectral operations, scalar fields, or batching.

## Issue templates

Use Markdown issue templates unless the repository already uses YAML issue forms. Keep them simple and readable.

### Bug report

Create `.github/ISSUE_TEMPLATE/bug_report.md`:

````markdown
---
name: Bug report
about: Report incorrect behavior or a regression
title: "[Bug]: "
labels: ["bug"]
---

## Expected behavior

What should happen?

## Actual behavior

What happened instead?

## Minimal reproduction

```python
# Paste the smallest example that reproduces the problem.
````

## Environment

* SpaceCore version or commit:
* Python version:
* Backend: NumPy / JAX / Torch / CuPy
* Operating system:

## Additional context

Any extra information, traceback, or notes.

````

### Feature request

Create `.github/ISSUE_TEMPLATE/feature_request.md`:

```markdown
---
name: Feature request
about: Propose a new feature or API improvement
title: "[Feature]: "
labels: ["feature"]
---

## Problem

What problem are you trying to solve?

## What you tried

What workaround or existing API did you try?

## Proposed API

```python
# Sketch the API you would like to use.
````

## Mathematical note

What mathematical structure, invariant, or assumption is involved?

## Alternatives

What alternatives should be considered?

````

### Contribution task

Create `.github/ISSUE_TEMPLATE/contribution_task.md`:

```markdown
---
name: Contribution task
about: Maintainer-created task with bounded scope
title: "[Task]: "
labels: ["help-wanted"]
---

## File to touch

```text
path/to/file.py
````

## Example to follow

```text
path/to/similar/file.py
```

## Task

Describe the exact change.

## Done condition

State the concrete command or test that must pass.

```bash
pytest tests/path -k name -q
```

## Mathematical note

State the relevant invariant, if any.

## Prerequisite level

Choose one:

* No mathematics
* Basic NumPy
* Linear algebra
* Inner product spaces / adjoints
* Functional analysis
* Algorithm theory
* Jordan algebras

````

## Label taxonomy document

Create `docs/dev/contributing/labels.md`.

It must document the intended labels and their meanings.

Required labels:

### Contributor-readiness labels

- `good-first-issue` — only for tasks where the issue body gives the exact file, an example to follow, and a concrete done condition. A contributor should be able to complete it without asking a design question.
- `help-wanted` — well-specified work that may require one clarification.
- `needs-design` — design direction is not settled. Do not assign these to new contributors.

### Type labels

- `documentation`
- `test`
- `fix`
- `feature`

### Component labels

- `backend`
- `space`
- `linop`
- `functional`
- `linalg`
- `docs`
- `ci`
- `release`

### Maintainer rules

Add this text, or equivalent:

```markdown
## Maintainer rules

`good-first-issue` is restrictive. Do not use it for tasks involving unsettled design, new mathematical abstractions, dtype/field policy changes, batching redesign, metric-adjoint semantics, or block-operator design unless the issue body pins the exact file, expected behavior, and done condition.

`needs-design` means the task is intentionally not ready for implementation. New contributors should not be assigned these issues.

Every geometry-, adjoint-, spectral-, field-, or batching-related issue should contain a mathematical note.
````

## Label setup script

Create `scripts/setup_labels.sh`.

Requirements:

* POSIX shell or Bash.
* Safe to rerun.
* Uses `gh label create --force`.
* Fails if `gh` is unavailable.
* Prints a useful message before mutating labels.

Suggested implementation:

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "error: GitHub CLI 'gh' is required to set up labels" >&2
  exit 1
fi

echo "Creating/updating SpaceCore GitHub label taxonomy..."

gh label create "good-first-issue" --force \
  --description "Beginner-safe: exact file, example, and done condition provided" \
  --color "7057ff"

gh label create "help-wanted" --force \
  --description "Well-specified work where maintainer help is welcome" \
  --color "008672"

gh label create "needs-design" --force \
  --description "Design direction is not settled; not ready for new contributors" \
  --color "d93f0b"

gh label create "documentation" --force \
  --description "Documentation work" \
  --color "0075ca"

gh label create "test" --force \
  --description "Tests and test infrastructure" \
  --color "1d76db"

gh label create "fix" --force \
  --description "Bug fix or correctness fix" \
  --color "d73a4a"

gh label create "feature" --force \
  --description "New feature or API extension" \
  --color "a2eeef"

gh label create "backend" --force \
  --description "BackendOps, Context, Array API, optional backend behavior" \
  --color "5319e7"

gh label create "space" --force \
  --description "Spaces, elements, inner products, capabilities, checks" \
  --color "5319e7"

gh label create "linop" --force \
  --description "Linear operators, adjoints, matrix-backed and matrix-free operators" \
  --color "5319e7"

gh label create "functional" --force \
  --description "Functionals, gradients, pull-backs" \
  --color "5319e7"

gh label create "linalg" --force \
  --description "Iterative algorithms and solver infrastructure" \
  --color "5319e7"

gh label create "docs" --force \
  --description "Documentation site, tutorials, developer docs" \
  --color "0075ca"

gh label create "ci" --force \
  --description "Continuous integration, release checks, automation" \
  --color "c5def5"

gh label create "release" --force \
  --description "Release preparation, changelog, tagging, packaging" \
  --color "bfdadc"

echo "Done."
```

Make the script executable:

```bash
chmod +x scripts/setup_labels.sh
```

## Documentation wiring

Add `docs/dev/contributing/labels.md` to the contributor documentation index if such an index exists.

If the project uses Sphinx toctrees, add the labels page to the relevant toctree.

## Validation

Run:

```bash
python - <<'PY'
from pathlib import Path

required = [
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    ".github/ISSUE_TEMPLATE/contribution_task.md",
    "docs/dev/contributing/labels.md",
    "scripts/setup_labels.sh",
]

missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit(f"Missing files: {missing}")

print("GitHub infrastructure files exist.")
PY

bash -n scripts/setup_labels.sh
```

If Sphinx docs are available, also run:

```bash
sphinx-build -W -b html docs/source docs/source/_build/html
```

Do not run `scripts/setup_labels.sh` unless the local environment is authenticated with the correct GitHub repository. The script must be committed so maintainers can run it when ready.

## Acceptance criteria

This task is complete when:

* PR template exists and includes the invariant checklist item.
* Bug report, feature request, and contribution task issue templates exist.
* Label taxonomy is documented in the repo.
* `scripts/setup_labels.sh` can recreate the label taxonomy with `gh label create --force`.
* `good-first-issue` and `needs-design` maintainer rules are documented.
* Contributor docs link to the label taxonomy.
* Shell syntax check for `scripts/setup_labels.sh` passes.
* Docs build passes if the project documentation environment is available.
