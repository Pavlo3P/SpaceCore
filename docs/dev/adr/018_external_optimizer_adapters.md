# ADR-018: External optimizer adapters

## Status

Accepted and implemented; **moved out (0.5.0a, 2026-10-06)** to the separate `spacecore-optimize` package. SpaceCore ships no optimizer adapters.

## Decision (as it was)

Thin adapters (`minimize_scipy`, `minimize_optax`, `line_search_scipy`) take a `Functional`, evaluate it through `value`/`grad`, and own no iteration logic. JAX- and manifold-specific seams are tutorials, not adapters.

## What still holds

The handoff rule for anyone wiring SpaceCore to an external optimizer: `grad` is the Riesz gradient, so convert it to a coordinate gradient with `X.riesz` before giving it to a coordinate optimizer (identity on a Euclidean space, mandatory otherwise).

## Where it went

`spacecore-optimize`.

## Full record

The complete original text (context, design, alternatives, invariants) is in git history: `git show 0ca3f2b:docs/dev/adr/018_external_optimizer_adapters.md`.
