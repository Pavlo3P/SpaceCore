# ADR-013: Tree-structured spaces

## Status

Proposed

## Context

Current product spaces support tuple elements and registered pytree structures. Near-term design work should generalize this toward variable trees without losing SpaceCore-specific mathematical contracts.

## Current design

`ProductSpace` is currently the structured space implementation. It can use `TupleStructure` or `PytreeStructure`. `PytreeStructure` delegates flattening and unflattening to JAX tree utilities and requires registered pytree/dataclass structures. Product components remain ordered coordinate spaces and capabilities are computed from shared component capabilities.

## Decision

The planned direction is to model variables as pytrees and treat `ProductSpace` as a future special case of a more general `TreeSpace`. Generic tree traversal and structural definitions should be delegated to a tree library such as `optree`; SpaceCore should keep ownership of spaces, geometry, validation, context conversion, batching, and mathematical capabilities.

## Rationale

Tree-shaped models are common in optimization and ML, but tree traversal is not SpaceCore's mathematical differentiator. Delegating generic tree mechanics avoids maintaining a parallel tree library.

## Alternatives considered

Keeping only tuple product spaces was rejected as too limiting for real model variables. Reimplementing full pytree semantics inside SpaceCore was rejected because mature tree libraries already solve traversal, registration, and structure comparison.

## Consequences

Do not treat current `ProductSpace` as the final tree abstraction. Future work must specify how tree leaves map to spaces, how context conversion rebuilds leaves, how batching interacts with tree leaves, and how capabilities compose across trees.

## Contributor invariants

- This ADR is a planned direction, not an implementation claim.
- Generic tree flattening should be delegated; SpaceCore-specific math must stay in SpaceCore.
- Product-space behavior must remain stable until a `TreeSpace` decision supersedes it.
- Future tree spaces must preserve validation, geometry, batching, and conversion contracts from [ADR-003](003_space_hierarchy.md) and [ADR-006](006_current_batching_model.md).
