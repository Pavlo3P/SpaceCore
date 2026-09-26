# Follow-up Task: Complete truthful capability dispatch and remove compatibility layers

## Goal

Finish the hierarchy refactor by ensuring:

1. baseline product/stacked classes expose only guaranteed methods;
2. combined capabilities are preserved exactly;
3. constructors validate required capabilities early;
4. all backward-compatibility aliases and migration shims are removed.

## Phase 1 — Remove unsupported methods from baseline classes

Refactor:

```python
ProductSpace
StackedSpace
```

so they implement only `CoordinateSpace` operations.

Move these methods into the corresponding specialized classes:

```python
inner
riesz
riesz_inverse
norm
is_euclidean
star
jordan
spectrum
spectral_decompose
from_spectrum
spectral_apply
```

Required ownership:

```text
ProductInnerProductSpace / StackedInnerProductSpace
    -> inner-product and Riesz operations

ProductStarSpace / StackedStarSpace
    -> star

ProductJordanAlgebraSpace / StackedJordanAlgebraSpace
    -> Jordan and spectral operations

ProductEuclideanJordanAlgebraSpace / StackedEuclideanJordanAlgebraSpace
    -> combined Jordan + inner-product contract
```

Calling unsupported methods on baseline classes must be impossible through normal attribute lookup.

## Phase 2 — Preserve arbitrary capability combinations

Replace the current mutually exclusive product/stacked dispatch with exact capability composition.

Support at least these combinations:

```text
Coordinate
Coordinate + InnerProduct
Coordinate + Star
Coordinate + Jordan
Coordinate + InnerProduct + Star
Coordinate + InnerProduct + Jordan
Coordinate + Star + Jordan
Coordinate + InnerProduct + Star + Jordan
Coordinate + EuclideanJordan
Coordinate + EuclideanJordan + Star
```

A product or stacked space must preserve every capability shared by all components or supported by the base space.

Example:

```python
base: CoordinateSpace + InnerProductSpace + StarSpace
stacked(base)
```

must satisfy:

```python
isinstance(result, CoordinateSpace)
isinstance(result, InnerProductSpace)
isinstance(result, StarSpace)
not isinstance(result, JordanAlgebraSpace)
```

Implement this using explicit combined concrete classes, a controlled class registry, or generated composition classes. Keep dispatch deterministic and inspectable.

## Phase 3 — Centralize capability detection and dispatch

Add internal helpers such as:

```python
_space_capabilities(space) -> frozenset[type]
_product_capabilities(spaces) -> frozenset[type]
_make_product_space(spaces, structure, ctx)
_make_stacked_space(base, count, ctx)
```

For products, compute the intersection of component capabilities.

For stacked spaces, copy the base-space capabilities.

Use one registry mapping capability sets to concrete implementations.

Add tests covering every supported capability combination.

## Phase 4 — Add early constructor validation

Validate constructor inputs before accessing capability-specific attributes.

Examples:

```python
ProductSpace(spaces)
```

must verify every component is a `CoordinateSpace`.

```python
ProductInnerProductSpace(spaces)
```

must verify every component is an `InnerProductSpace`.

```python
StackedStarSpace(base, count)
```

must verify `base` is a `StarSpace`.

Raise clear `TypeError` messages such as:

```text
ProductSpace requires every component to be a CoordinateSpace;
component 2 is MinimalSpace.
```

Validate:

```python
count >= 0
spaces is nonempty, if required by the existing contract
contexts are compatible
structures match component arity
```

## Phase 5 — Remove all backward compatibility

Delete all compatibility aliases, deprecated names, re-export shims, and fallback behavior introduced during the refactor.

Remove items such as:

```python
LegacyVectorSpace
old concrete VectorSpace aliases
old module-path forwarding imports
deprecated apply aliases
compatibility constructors
deprecated property names
```

Remove the old method name:

```python
apply(x, f)
```

and keep only:

```python
spectral_apply(x, f)
```

Remove warnings that preserve old behavior. Old usage should fail immediately with `ImportError`, `AttributeError`, or `TypeError`.

Update all internal code, tests, examples, documentation, notebooks, and benchmarks to use the final API directly.

## Phase 6 — Clean public exports

Audit:

```text
spacecore/__init__.py
spacecore/space/__init__.py
spacecore/space/base/__init__.py
spacecore/space/concrete/__init__.py
```

Export only the final supported public classes.

Keep internal specialized implementation classes private unless users are expected to construct them directly.

Document the final public surface explicitly.

## Phase 7 — Update consumers

Audit consumers for capability assumptions.

Replace attribute probing and late failures with explicit requirements:

```python
isinstance(space, CoordinateSpace)
isinstance(space, InnerProductSpace)
isinstance(space, StarSpace)
isinstance(space, JordanAlgebraSpace)
isinstance(space, EuclideanJordanAlgebraSpace)
```

Use the weakest sufficient capability for every algorithm.

## Tests

Add tests verifying:

1. Baseline `ProductSpace` and `StackedSpace` expose only coordinate methods.
2. Unsupported capability methods are absent, not merely failing internally.
3. Every supported capability combination is preserved exactly.
4. Product capabilities equal the intersection of component capabilities.
5. Stacked capabilities equal the base-space capabilities.
6. Invalid constructor inputs fail early with clear `TypeError`.
7. No compatibility aliases remain importable.
8. `apply` no longer exists anywhere in the public API.
9. No old module paths remain supported.
10. Repository search finds no stale API usages.
11. Public API tests match the final export list.
12. Full suite and documentation build pass.

## Acceptance criteria

* Product and stacked spaces expose exactly their supported capabilities.
* Combined capability dispatch preserves all valid combinations.
* Invalid constructions fail before attribute access.
* The public API contains only final class and method names.
* All backward-compatibility code is removed.
* Full tests and documentation build pass.
