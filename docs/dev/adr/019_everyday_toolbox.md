# ADR-019: Everyday toolbox

## Status

Accepted; **partly moved out (0.5.0a)** to `spacecore-optimize` (commit 8b60591).

## Decision (as it was)

A small set of named constructors over existing machinery plus one closed-form, metric-aware proximal primitive, with no new `Functional` hierarchy: `least_squares`, coordinate norms, entropy, KL, Huber, spectral norms, and `generalized_shrinkage` with `prox_l1`/`prox_l2sq`/`project_nonneg`.

## What still holds

`SquaredL2NormFunctional`, `LpNormFunctional` and `L1NormFunctional` stay in `spacecore.functional`, implemented through `_CoordinateFunctional`, which turns a coordinate gradient into the Riesz gradient. Proxes are registered rules behind `Functional.prox` (ADR-020).

## Where it went

`least_squares`, entropy, KL, Huber, the spectral norms and the shrinkage helpers: `spacecore-optimize`.

## Full record

The complete original text (context, design, alternatives, invariants) is in git history: `git show 0ca3f2b:docs/dev/adr/019_everyday_toolbox.md`.
