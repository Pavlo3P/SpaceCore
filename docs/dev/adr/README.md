# Architecture Decision Records

Architecture Decision Records (ADRs) record implemented architecture and design decisions for SpaceCore. They explain why important design commitments exist, what contributors must preserve, and where future changes need an explicit decision before implementation.

ADRs are for contributors, not just maintainers. A contributor should be able to read the relevant ADRs before changing an area and understand the current design constraints without reconstructing them from code review history.

ADRs should explain the current design, not merely describe future plans. Proposed or deferred ADRs may reserve a decision point, but accepted ADRs should document behavior, contracts, and invariants that SpaceCore actually implements.

Each ADR should cover one implemented concept or one design decision. If an ADR becomes too large, split it into several ADRs. Long mathematical exposition belongs in a design note, tutorial, or reference document, not inside the ADR.

## ADR format

SpaceCore ADRs use this required section structure:

```markdown
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

## Index

| ADR | Slug | Status | Notes |
| --- | --- | --- | --- |
| [ADR-001](001_backend_layer.md) | `001_backend_layer` | Accepted | Backend layer architecture. |
| [ADR-002](002_context_and_conversion.md) | `002_context_and_conversion` | Accepted | Context ownership, normalization, and explicit conversion. |
| [ADR-003](003_space_hierarchy.md) | `003_space_hierarchy` | Accepted | Space hierarchy and coordinate representation. |
| [ADR-004](004_inner_product_and_geometry.md) | `004_inner_product_and_geometry` | Accepted | Separate geometry objects, Riesz maps, norms, and metric correctness. |
| [ADR-005](005_space_subclasses_and_capabilities.md) | `005_space_subclasses_and_capabilities` | Accepted | Concrete spaces, capability mixins, and conversion recomputation. |
| [ADR-006](006_current_batching_model.md) | `006_current_batching_model` | Accepted | Current leading-axis batching model. |
| [ADR-007](007_linop_contract.md) | `007_linop_contract` | Accepted | LinOp domain/codomain, adjoint, algebra, validation, and conversion contract. |
| [ADR-008](008_linop_subclasses.md) | `008_linop_subclasses` | Accepted | Matrix-backed, matrix-free, lazy, and product LinOp families. |
| [ADR-009](009_metric_adjoint.md) | `009_metric_adjoint` | Accepted | Metric adjoint and Riesz-map design. |
| [ADR-010](010_functional_contract.md) | `010_functional_contract` | Accepted | Scalar-valued functional, gradient, and pull-back contract. |
| [ADR-011](011_linalg_contract.md) | `011_linalg_contract` | Accepted | Iterative linalg solver contract. |
| [ADR-012](012_jordan_spectrum.md) | `012_jordan_spectrum` | Accepted | Jordan spectral API and product spectral decomposition. |
| [ADR-013](013_tree_structured_spaces.md) | `013_tree_structured_spaces` | Proposed | Planned tree-structured space direction. |
| [ADR-014](014_check_policy.md) | `014_check_policy` | Proposed | Intended check levels for 0.4.0 test generation. |
| [ADR-015](015_dtype_default_vs_scalar_field.md) | `015_dtype_default_vs_scalar_field` | Accepted | Dtype defaults and scalar-field contract. |
| [ADR-016](016_kernel_layers.md) | `016_kernel_layers` | Accepted | Two-layer kernel architecture (check-free cores + benchmarked specs) and broader structural-dispatch policy; dispatch implemented (`dispatch_key`/`priority`/`cost`, registry index, `dispatch()` with off/on/verify and memory gate), off by default. |
| [ADR-017](017_tensor_product_spaces.md) | `017_tensor_product_spaces` | Deferred | Direct-product vs tensor-product boundary settled; tensor-product implementation demand-gated. |
| [ADR-018](018_external_optimizer_adapters.md) | `018_external_optimizer_adapters` | Accepted | SciPy/optax adapters taking a `Functional`, with metric→coordinate gradient handoff; implemented in `spacecore.optimize` (`minimize_scipy`, `line_search_scipy`, `minimize_optax`). |
| [ADR-019](019_everyday_toolbox.md) | `019_everyday_toolbox` | Accepted | Battery functionals (`least_squares`, coordinate norm/entropy/KL/Huber, spectral `SpectralLpNormFunctional`/`NuclearNormFunctional`) and the closed-form, metric-aware `generalized_shrinkage` proximal primitive with `prox_l1`/`prox_l2sq`/`project_nonneg`, in the `spacecore.functional.tools` subpackage; `IndicatorFunctional`/`project_C` deferred to ADR-020. |
| [ADR-020](020_sets_and_projection.md) | `020_sets_and_projection` | Proposed | `Set` abstraction owning an ambient space and a metric projector; typed `C` for indicators/projection. |
