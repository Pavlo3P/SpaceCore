# ADR-022: Caching

## Status

**Withdrawn (0.5.0a, 2026-09-26).** The stacked-block cache (`CachedStackParts`) went with the ADR-016 dispatch layer it fed, and `StackedLinOp`/`SumToSingleLinOp` no longer exist.

## Decision (as it was)

Memoize, per operator, the input-independent stacked block array used by the uniform batched folds, and cache dispatch decisions later.

## What still holds

Two invariants apply to any future cache: a matrix-free operator is never densified to fill a cache, and a derived cache is excluded from `__eq__`, `__hash__` and pytree structure.

## Full record

The complete original text (context, design, alternatives, invariants) is in git history: `git show 0ca3f2b:docs/dev/adr/022_caching.md`.
