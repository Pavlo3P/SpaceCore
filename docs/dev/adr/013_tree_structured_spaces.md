# ADR-013: Tree-structured spaces

## Status

Accepted for 0.4.0 Phase D

## Context

Optimization and ML variables commonly use nested Python structures. SpaceCore
needs one structured finite-coordinate abstraction without maintaining custom
tree traversal or conflating a Cartesian product with a tensor product.

## Decision

`TreeSpace` is the only structured finite direct-product space. `optree` is a
required core dependency and owns generic traversal, deterministic leaf order,
structure comparison, reconstruction, registration, and paths. SpaceCore owns
leaf spaces, geometry, validation, context conversion, batching, and
mathematical capabilities.

Plain matching Python trees are normal elements. `TreeElement` is an optional
explicit binding of ordered leaves to a `TreeSpace`. Tuple-style products use
`TreeSpace.from_leaf_spaces(...)`.

## Rationale

Tree traversal is not SpaceCore's mathematical differentiator. One abstraction
avoids parallel tuple and pytree implementations while retaining direct-product
semantics and deterministic dense-coordinate flattening.

## Alternatives considered

Keeping only tuple products was rejected as too limiting. Reimplementing pytree
semantics was rejected because mature tree libraries already solve traversal,
registration, and structure comparison. Keeping a second compatibility space
was rejected because it would preserve duplicate behavior and terminology.

## Consequences

Tree-space vector, batch, conversion, star, Jordan, and Riesz operations are
leafwise. Inner products sum leaf inner products. Capabilities are advertised
only when every leaf supports them. Membership errors identify the failing leaf
path and obey the configured check level.

## Contributor invariants

- Generic tree mechanics are delegated to `optree`; SpaceCore-specific math
  remains in SpaceCore.
- A `TreeSpace` is a finite direct product, not a tensor product.
- Field and representation dtype follow the resolved context; exact membership
  checks remain the responsibility of each leaf space.
- Tree spaces preserve validation, geometry, batching, and conversion contracts
  from [ADR-003](003_space_hierarchy.md) and
  [ADR-006](006_current_batching_model.md).
