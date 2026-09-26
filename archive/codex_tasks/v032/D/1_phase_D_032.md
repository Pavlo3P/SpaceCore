# Codex Task: Complete SpaceCore Contributor Documentation

## Goal

Create the contributor documentation for SpaceCore so that a new contributor can set up the project, understand the codebase layers, know the mathematical prerequisites for different contribution types, and follow the expected pull-request process.

This task must be self-contained. Do not rely on external planning documents.

## Files to create or update

Create:

```text
docs/dev/contributing/
docs/dev/contributing/setup.md
docs/dev/contributing/architecture.md
docs/dev/contributing/prerequisites.md
docs/dev/contributing/process.md
```

Also update the developer documentation navigation so `docs/dev/contributing/` is reachable from the docs site.

Use the repository’s existing documentation style. If the docs use Markdown, write Markdown. If the developer docs index is in reStructuredText, add the appropriate `toctree` entry.

## 1. Create `setup.md`

Write exact commands for a contributor to clone, install, and verify the project.

Include at least:

```bash
git clone https://github.com/Pavlo3P/SpaceCore
cd SpaceCore
pip install -e ".[dev]"
pytest --co -q
pytest tests/ -x -q
pytest tests/spaces/ -v
ruff check .
```

Also document the fuller optional-backend install:

```bash
pip install -e ".[jax,torch,dev]"
```

State clearly:

* `.[dev]` is the minimal contributor install;
* the minimal install should run the NumPy-backed test suite;
* optional backend tests should skip cleanly when JAX or Torch is absent;
* `[jax,torch,dev]` is for contributors working on backend-specific behavior.

Mention that `pytest --co -q` is useful because it catches collection-time import errors before running the full suite.

## 2. Create `architecture.md`

Write a compact codebase orientation with five sections.

### Backend layer

Describe `spacecore/backend/`.

Cover:

* `BackendOps`;
* Array API assumptions;
* backend detection;
* optional dependencies;
* what SpaceCore owns versus what NumPy/JAX/Torch/CuPy own.

### Space layer

Describe `spacecore/space/`.

Cover:

* abstract `VectorSpace`;
* dense coordinate/vector spaces;
* product spaces;
* elements and membership;
* zero/add/scale;
* inner products and geometry;
* checks;
* scalar fields;
* batching hooks.

Explain that spaces represent mathematical structure, not merely array shapes.

### LinOp and Functional layers

Describe `spacecore/linop/` and `spacecore/functional/`.

Cover:

* typed maps between domain and codomain spaces;
* `apply`;
* `rapply`;
* `.H`;
* algebraic operations;
* matrix-backed versus matrix-free operators;
* functionals as scalar-valued maps;
* gradients and pull-backs where present;
* batched application paths.

State that matrix-free `rapply` is a user-supplied assertion of the correct metric adjoint.

### Linalg layer

Describe `spacecore/linalg/`.

Cover:

* iterative algorithms;
* use of space geometry;
* use of LinOp contracts;
* backend independence;
* solver assumptions.

### Cross-cutting infrastructure

Describe:

* `_contextual`;
* `_batching`;
* `_version.py`;
* any other existing cross-cutting modules that are relevant.

Keep this section factual and short.

## 3. Add public concrete-class map

In `architecture.md`, add a section called `Public class map`.

List every public concrete class that is exported by SpaceCore and give one sentence for each.

The goal is navigation, not full API documentation.

Example style:

```markdown
- `DenseCoordinateSpace` — dense coordinate representation of arrays with a fixed shape and context.
- `MatrixFreeLinOp` — linear operator defined by user-supplied forward and adjoint callables.
```

Use the actual exported names from the package. Do not invent classes.

Check exports in:

```text
spacecore/__init__.py
spacecore/space/
spacecore/linop/
spacecore/functional/
spacecore/linalg/
spacecore/backend/
```

## 4. Create `prerequisites.md`

Create a table mapping contribution types to mathematical prerequisites.

Include at least these rows:

| Contribution type                   | Mathematical prerequisite                                  | Pointer                     |
| ----------------------------------- | ---------------------------------------------------------- | --------------------------- |
| Adding or fixing a docstring        | None                                                       | `setup.md`                  |
| Adding a test for existing behavior | Basic NumPy and pytest                                     | `setup.md`                  |
| New check in `_checks.py`           | Linear algebra: shapes, dtypes, validation cost            | ADR-014                     |
| Dtype or scalar-field behavior      | Scalar fields, dtype promotion, backend array semantics    | ADR-015                     |
| New `InnerProduct` subclass         | Inner product spaces and Riesz maps                        | ADR-004 and ADR-009         |
| New matrix-backed `LinOp`           | Linear maps and adjoints                                   | LinOp ADRs                  |
| New matrix-free `LinOp`             | Linear maps, metric adjoints, and algorithm knowledge      | LinOp ADRs                  |
| New linalg method                   | Algorithm theory and convergence assumptions               | Must cite a reference in PR |
| Extending Jordan hierarchy          | Jordan algebras and spectral calculus                      | ADR-012                     |
| New Space type                      | Functional analysis and all relevant lower-level contracts | Architecture ADRs           |

Add the following principle clearly:

SpaceCore does not abstract away the mathematics. It gives mathematical structures to objects. A geometrically incorrect contribution that passes tests is still wrong; mathematical correctness is part of review.

## 5. Create `process.md`

Document the contribution process.

Include:

### Branch naming

Use:

```text
feature/<short-name>
fix/<short-name>
docs/<short-name>
test/<short-name>
```

### Before opening a PR

A PR should normally include:

* tests for changed behavior;
* updated docstrings if public behavior changed;
* a changelog entry under `[Unreleased]`;
* documentation updates if contributor-facing behavior changed;
* mathematical invariant statement for changes touching geometry, adjoints, spectral methods, scalar fields, or batching.

### Review order

State that review checks correctness first, then style.

For mathematical code, review should check:

* domain and codomain spaces;
* scalar field;
* inner-product assumptions;
* adjoint identity;
* validation level;
* backend behavior;
* batching behavior where applicable.

### Blocked PRs

Explain that blocked PRs should be narrowed:

* document the blocker;
* split unrelated work;
* merge completed safe parts;
* open follow-up issues for design questions.

### Patch release policy

Document when a patch release is appropriate:

* regression fix;
* packaging fix;
* documentation fix that affects install/use;
* optional-backend import or skip fix.

## 6. Documentation navigation

Update developer docs navigation so these pages are discoverable.

At minimum, the docs should expose:

```text
Contributor guide
  - Setup
  - Architecture
  - Prerequisites
  - Process
```

Run the docs build after adding the pages.

## 7. Validation

Run:

```bash
pytest --co -q
pytest tests/ -x -q
ruff check .
sphinx-build -W -b html docs/source docs/source/_build/html
```

If the repository has a release-candidate script, run it too:

```bash
bash scripts/verify_release_candidate.sh
```

## Done condition

This task is complete when:

* `docs/dev/contributing/setup.md` exists and contains minimal and optional-backend setup commands;
* `architecture.md` gives a five-layer codebase orientation;
* `architecture.md` includes a public concrete-class map;
* `prerequisites.md` maps contribution types to mathematical prerequisites;
* `process.md` documents branch policy, PR requirements, review rules, blocked PR handling, and patch-release policy;
* contributor docs are linked from developer docs navigation;
* tests, lint, and strict docs build pass.
