# Task: Add SpaceCore developer vision, current-state, and branch-policy documentation

## Goal

Add the developer-facing documentation needed for the `0.3.2` contributor-infrastructure release.

This task creates or updates:

* `docs/dev/vision.md`
* `docs/dev/current.md`
* contributor branch-policy documentation, most likely in `docs/dev/contributing/process.md`

Do not implement library features in this task. This is documentation and release-planning infrastructure only.

## Required work

### 1. Create `docs/dev/vision.md`

Write a concise one-to-two-page project vision document.

It must contain the following sections:

```markdown
# SpaceCore Vision

## Purpose

## Non-goals

## Ecosystem position

## Release sequencing principles

## Definition of 1.0.0
```

Content requirements:

* **Purpose**: explain that SpaceCore provides explicit mathematical structures for backend-independent numerical code: spaces, elements, inner products, linear operators, functionals, and linalg routines.
* **Non-goals**: state that SpaceCore does not hide problem-specific mathematics from the user. It should provide structure and contracts, not pretend that mathematical modeling choices are automatic.
* **Ecosystem position**: add a compressed comparison with SciPy `LinearOperator`, PyLops, CoLA, Pyxu, ODL, and Geomstats. Keep this section short: one to two pages for the entire `vision.md`, not a separate long survey. Cover overlapping abstractions, key differences, likely adapter candidates, and SpaceCore’s intended niche.
* **Release sequencing principles**: explain why contributor infrastructure comes before the larger `0.4.0` test-generator and backend-conformance work.
* **Definition of 1.0.0**: give concrete stability criteria: stable public API, documented backend contracts, tested extension APIs, reliable interoperability, benchmark/regression tracking, validated generic conic abstractions, and deprecation policy.

Tone: honest, technical, contributor-facing. Avoid marketing language.

### 2. Create `docs/dev/current.md`

Create a short document rewritten at each release.

Use this structure:

```markdown
# Current Development State

## Now

## Open questions

## Next
```

Content requirements:

* **Now**: describe `0.3.2` as a contributor-infrastructure release. Mention:

  * architecture ADRs;
  * contributor documentation;
  * GitHub templates and labels;
  * 0.4.0 issue decomposition;
  * the matrix-free `LinOp` adjoint-contract fix;
  * current batching model documentation.
* Include a placeholder or real link to the `0.3.2` GitHub milestone.
* **Open questions** must include:

  * dtype defaults versus scalar-field contract;
  * dispatch architecture;
  * tensor-product spaces;
  * mixed precision and cross-precision compatibility;
  * batching limitations and batch-conformance boundaries.
* **Next**: describe `0.4.0` as tentative test-infrastructure work:

  * reusable generators for spaces, LinOps, functionals, and linalg;
  * backend conformance;
  * batching conformance;
  * check-policy migration;
  * dtype/field Stage 1;
  * block/product/tree-structured testing.

This document should be explicit that contributors should not open unsolicited PRs for unsettled open questions unless an issue already defines the desired work.

### 3. Document branch policy

Update `docs/dev/contributing/process.md` if it exists. If it does not exist, create it with a minimal process section.

Add a branch-policy section covering:

* contributor materials may be drafted on a temporary branch;
* accepted contributor infrastructure must be merged into the default branch before tagging `0.3.2`;
* release-critical docs must not be stranded on a side branch;
* PRs should use branch prefixes such as `docs/`, `fix/`, `feature/`;
* blocked PRs should either be narrowed or recorded in `docs/dev/current.md`.

### 4. Wire docs into navigation

If developer docs have an index page or Sphinx toctree, link:

* `docs/dev/vision.md`
* `docs/dev/current.md`
* `docs/dev/contributing/process.md`

Use the project’s existing documentation style.

## Validation

Run the relevant checks:

```bash
pytest --co -q
ruff check .
sphinx-build -W -b html docs/source docs/source/_build/html
```

If Sphinx does not currently include `docs/dev/*.md`, either wire the files properly or document why they are developer-only Markdown files outside the rendered docs.

## Acceptance criteria

This task is complete when:

* `docs/dev/vision.md` exists and contains purpose, non-goals, ecosystem position, sequencing, and 1.0.0 definition;
* `docs/dev/current.md` exists and contains Now, Open questions, and Next;
* branch policy is documented in contributor process docs;
* developer docs link to the new files where appropriate;
* the ecosystem-position section is concise, not a long library-comparison document;
* docs do not claim that batching was removed;
* docs identify batching as current behavior with limitations to be tested in `0.4.0`;
* strict docs build passes or any non-rendered developer-doc choice is explicitly justified.
