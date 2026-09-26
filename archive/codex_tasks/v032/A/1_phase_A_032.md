# Codex Task: Add ADR Infrastructure for SpaceCore Developer Documentation

## Goal

Add the Architecture Decision Record infrastructure for SpaceCore developer documentation.

This task does **not** require writing the actual ADRs. It only creates the ADR directory, index, template, writing rules, future reservations, and documentation navigation.

## Required changes

### 1. Create ADR directory

Create:

```text
docs/dev/adr/
```

Add:

```text
docs/dev/adr/README.md
docs/dev/adr/template.md
```

## 2. Write `docs/dev/adr/README.md`

The README must explain what ADRs are for in SpaceCore.

Include:

* ADRs record implemented architecture and design decisions.
* ADRs are for contributors, not just maintainers.
* ADRs should explain the current design, not merely describe future plans.
* ADRs should be small.

Use this rule explicitly:

```text
Each ADR should cover one implemented concept or one design decision. If an ADR becomes too large, split it into several ADRs. Long mathematical exposition belongs in a design note, tutorial, or reference document, not inside the ADR.
```

## 3. Define ADR format

In `README.md`, define the required ADR sections:

```text
# ADR-NNN: Title

## Status

Accepted / Proposed / Superseded / Deferred

## Context

What problem, concept, or architectural area this ADR records.

## Current design

What SpaceCore currently does, if the ADR records implemented architecture.

## Decision

The design commitment or rule.

## Rationale

Why this design exists.

## Alternatives considered

Other plausible designs and why they were not chosen.

## Consequences

What this means for future extensions.

## Contributor invariants

Rules contributors must preserve when touching this area.
```

## 4. Add `template.md`

Create a reusable template matching the same section structure.

Use placeholders, not specific content:

```markdown
# ADR-NNN: Short title

## Status

Proposed

## Context

...

## Current design

...

## Decision

...

## Rationale

...

## Alternatives considered

...

## Consequences

...

## Contributor invariants

...
```

## 5. Add ADR index

In `docs/dev/adr/README.md`, add an index table for planned ADRs.

Use this exact numbering:

```text
001_backend_layer
002_context_and_conversion
003_space_hierarchy
004_inner_product_and_geometry
005_space_subclasses_and_capabilities
006_current_batching_model
007_linop_contract
008_linop_subclasses
009_metric_adjoint
010_functional_contract
011_linalg_contract
012_jordan_spectrum
013_tree_structured_spaces
014_check_policy
015_dtype_default_vs_scalar_field
016_dispatch_architecture
017_tensor_product_spaces
```

Mark ADRs 016 and 017 as future reservations:

* `016_dispatch_architecture` — reserved for the dispatch/fusion design decision before implementation.
* `017_tensor_product_spaces` — reserved for tensor-product space and tensor/Kronecker LinOp design before implementation.

Do not create the ADR files themselves in this task unless they are empty placeholders. Prefer not to create placeholders if that would clutter navigation.

## 6. Add docs navigation

Make the ADR index reachable from developer documentation.

Inspect the existing docs structure and update the appropriate Sphinx index file, likely one of:

```text
docs/source/index.rst
docs/source/dev/index.rst
docs/source/developer/index.rst
```

Use the existing documentation style.

The ADR README should appear as a developer-documentation page in the built docs.

If Markdown files are already supported by the docs build, link `docs/dev/adr/README.md` directly. If not, add a small `.rst` bridge page under `docs/source/` that links to or includes the ADR README consistently with the project’s current docs setup.

## 7. Keep this task narrow

Do not:

* write ADR-001 through ADR-015;
* modify code behavior;
* change public APIs;
* add new design decisions beyond the ADR infrastructure;
* rename existing architecture concepts.

## Validation

Run:

```bash
pytest --co -q
sphinx-build -W -b html docs/source docs/source/_build/html
ruff check .
```

If the docs build has a project-specific command or script, use that as well.

## Done condition

This task is complete when:

* `docs/dev/adr/README.md` exists;
* `docs/dev/adr/template.md` exists;
* README defines ADR purpose, format, index, and small-ADR rule;
* ADRs 016 and 017 are reserved for future dispatch and tensor-product decisions;
* ADR index is reachable from developer docs navigation;
* strict docs build passes;
* no actual ADR content is written beyond infrastructure and reservations.
