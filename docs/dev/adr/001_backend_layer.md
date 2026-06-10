# ADR-001: Backend layer

## Status

Accepted

## Context

SpaceCore supports multiple array ecosystems while exposing one contributor-facing numerical contract. The backend layer must hide routine namespace differences without pretending all libraries have identical sparse, dtype, device, mutation, or control-flow semantics.

## Current design

`BackendOps` is the public backend contract. Concrete implementations currently cover NumPy/SciPy by default and optionally JAX, Torch, and CuPy when their packages are importable. A backend has a family name, dense and optional sparse array type predicates, dtype normalization, dense Array API style operations through `xp`, sparse conversion, control-flow primitives, vectorization, indexing mutation, and linear algebra helpers.

Backend detection is conservative. Context inference first uses `.ctx` on SpaceCore objects and then tests registered backend array predicates. Optional backend classes are registered only if import succeeds, and unknown names raise a typed backend error.

## Decision

SpaceCore code should call `BackendOps` methods for portable behavior. Direct `ops.xp` or backend-specific handles are escape hatches, not the stable SpaceCore API. Behavior that depends on mathematical spaces, validation, conversion, batching, adjoints, or geometry belongs in SpaceCore. Backend libraries own raw array semantics such as broadcasting, device placement, autograd, memory layout, and low-level dtype promotion.

## Rationale

One explicit contract keeps spaces, LinOps, functionals, and solvers backend-agnostic while still allowing each backend to implement sparse formats, tracing, and mutation idioms correctly.

## Alternatives considered

Using array libraries directly throughout the code was rejected because it spreads backend conditionals across mathematical code. Requiring all optional backends at install time was rejected because NumPy/SciPy should remain the baseline dependency set. Treating `xp` as the public contract was rejected because sparse, loops, dtype sanitization, and indexing semantics are not fully covered by the Array API.

## Consequences

New backend behavior must be added through `BackendOps` or an explicit backend-specific escape hatch. Tests for new backend methods must cover unavailable optional dependencies and avoid silently importing packages that are not installed. See [ADR-002](002_context_and_conversion.md) for context ownership and [ADR-011](011_linalg_contract.md) for solver delegation.

## Contributor invariants

- `BackendOps` methods expose stable SpaceCore behavior without silently depending on unavailable optional packages.
- Optional backend imports must fail closed: unavailable packages should omit that backend, not break NumPy usage.
- Portable core code should prefer `ops.method(...)` over `ops.xp.method(...)`.
- Backend detection must remain conservative and reject ambiguous array ownership.
