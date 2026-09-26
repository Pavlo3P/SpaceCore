# Codex Task: Write Architecture ADRs for SpaceCore 0.3.2

## Goal

Create concise Architecture Decision Records under `docs/dev/adr/` that document SpaceCore’s current implemented architecture and near-term design decisions before the 0.4.0 test-infrastructure work begins.

These ADRs are contributor-facing. They should help a new contributor understand existing backend, context, space, batching, LinOp, functional, linalg, spectral, validation, and dtype/field concepts before modifying them.

## General ADR requirements

Each ADR must be small. One ADR should cover one implemented concept or one decision. If an ADR becomes long, split it instead of expanding it.

Each ADR must include:

* context;
* current design;
* rationale;
* alternatives considered or rejected;
* consequences for future extensions;
* contributor-facing invariants.

Avoid long tutorials or mathematical exposition. Put only the architectural decision and the invariants needed for contributors. Longer explanations belong in design notes or future tutorials.

Also create or update `docs/dev/adr/README.md` so all ADRs are indexed and navigable.

## Files to create

Create the following files, using zero-padded numbering:

```text
docs/dev/adr/001_backend_layer.md
docs/dev/adr/002_context_and_conversion.md
docs/dev/adr/003_space_hierarchy.md
docs/dev/adr/004_inner_product_and_geometry.md
docs/dev/adr/005_space_subclasses_and_capabilities.md
docs/dev/adr/006_current_batching_model.md
docs/dev/adr/007_linop_contract.md
docs/dev/adr/008_linop_subclasses.md
docs/dev/adr/009_metric_adjoint.md
docs/dev/adr/010_functional_contract.md
docs/dev/adr/011_linalg_contract.md
docs/dev/adr/012_jordan_spectrum.md
docs/dev/adr/013_tree_structured_spaces.md
docs/dev/adr/014_check_policy.md
docs/dev/adr/015_dtype_default_vs_scalar_field.md
```

Also reserve future ADRs in the index:

```text
016_dispatch_architecture.md
017_tensor_product_spaces.md
```

Do not create full future ADRs unless there is already a real decision to record. The README should mark them as reserved/future.

## Required ADR content

### ADR-001: Backend layer

Document:

* `BackendOps`;
* supported array namespaces;
* backend detection;
* optional dependencies;
* Array API assumptions;
* what behavior belongs to SpaceCore versus the underlying array library;
* contributor invariant: backend methods should expose stable SpaceCore behavior without silently depending on unavailable optional packages.

### ADR-002: Context and conversion

Document:

* `Context`;
* context ownership;
* normalization;
* conversion;
* dtype defaults;
* optional dependency boundaries;
* removed policy objects;
* contributor invariant: context conversion must be explicit and predictable.

### ADR-003: Space hierarchy

Document:

* `VectorSpace` as abstract space layer;
* dense coordinate spaces;
* dense vector spaces;
* product spaces;
* element construction;
* membership;
* zero/add/scale behavior;
* distinction between mathematical spaces and coordinate representations.

### ADR-004: Inner product and geometry

Document:

* why `InnerProduct` is a separate class instead of only methods on `Space`;
* reuse of geometry across spaces;
* Riesz maps;
* norms;
* metric adjoints;
* Euclidean versus non-Euclidean examples;
* contributor invariant: geometry is part of mathematical correctness, not just implementation detail.

### ADR-005: Space subclasses and capabilities

Document:

* concrete public space subclasses;
* capability mixins;
* geometry ownership;
* Jordan capabilities;
* spectral capabilities;
* conversion behavior;
* capability recomputation on `convert()`.

### ADR-006: Current batching model

Document actual current batching behavior. Do not claim batching was removed.

Include:

* current `_batching` role;
* `vapply` / `rvapply` behavior;
* which abstractions own batched behavior;
* interaction with `apply` / `rapply`;
* interaction with validation and backend contexts;
* known limitations;
* what must be tested in 0.4.0.

### ADR-007: LinOp contract

Document:

* domain and codomain spaces;
* `apply`;
* `rapply`;
* `.H`;
* algebraic operations;
* validation;
* conversion;
* geometry-aware adjoints.

### ADR-008: LinOp subclasses

Document implemented LinOp families, including where present:

* matrix-backed;
* matrix-free;
* identity;
* zero;
* diagonal;
* composed;
* summed;
* scaled;
* structured/product operators.

Explain which subclasses own coordinate matrices and which trust user callables.

### ADR-009: Metric adjoint

Document:

* Riesz-map design;
* `validate_for`;
* `is_euclidean`;
* why type guards were not the right fix;
* matrix-backed adjoint formula (A^\sharp = R_X^{-1} A^\dagger R_Y);
* matrix-free contract: user-supplied `rapply` is already the metric adjoint and must not be Riesz-wrapped.

This ADR is release-gating.

### ADR-010: Functional contract

Document:

* `Functional`;
* scalar-valued maps;
* gradients;
* pull-backs where present;
* domain ownership;
* validation;
* how functionals differ from LinOps.

### ADR-011: Linalg contract

Document:

* iterative linalg routines;
* how they use space geometry;
* backend independence;
* what is delegated to backend ops;
* solver correctness assumptions.

### ADR-012: Jordan spectrum

Document:

* why `eigh` was replaced;
* `spectrum`;
* `spectral_decompose`;
* product spectral decompositions as structured objects;
* which spaces may expose spectral operations.

### ADR-013: Tree-structured spaces

Record the planned direction:

* variables as pytrees;
* `ProductSpace` as future special case of `TreeSpace`;
* what is delegated to `optree`;
* what remains SpaceCore-specific: spaces, geometry, validation, context conversion, batching, and mathematical capabilities.

This is a planned-direction ADR, not an implementation claim.

### ADR-014: Check policy

Document intended check levels:

* `none`;
* `cheap`;
* `standard`;
* `strict`.

Explain why Boolean `enable_checks` is too coarse and which invariants belong to which level.

This ADR is release-gating for 0.4.0 test generation.

### ADR-015: Dtype default versus scalar field

Document:

* `Context.dtype` as representation default;
* `Space.field` as mathematical contract;
* real versus complex membership;
* field-level dtype checks;
* dtype-preserving construction;
* operand-driven solver workspaces;
* why exact dtype equality and cross-precision compatibility remain strict until concrete workloads justify relaxing them.

This ADR is release-gating for 0.4.0 dtype/backend conformance.

## Cross-linking

Cross-link ADRs where concepts depend on each other:

* inner products ↔ metric adjoints;
* spaces ↔ capabilities;
* spaces ↔ batching;
* LinOps ↔ metric adjoints;
* check policy ↔ spaces, LinOps, functionals, batching;
* dtype/field ↔ context and spaces;
* Jordan spectrum ↔ space capabilities.

## Done conditions

The task is complete when:

* all ADR files listed above exist;
* `docs/dev/adr/README.md` indexes them;
* future ADRs 016 and 017 are reserved in the index;
* ADRs are concise and split when necessary;
* ADR-006 documents batching as current behavior;
* ADR-004 explains why `InnerProduct` is a separate class;
* ADR-009 clearly separates matrix-backed adjoints from matrix-free `rapply`;
* ADR-014 and ADR-015 are detailed enough to support 0.4.0 test and conformance work;
* documentation builds without broken links.
