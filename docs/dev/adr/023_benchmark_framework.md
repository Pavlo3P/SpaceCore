# ADR-023: Benchmark framework

## Status

Proposed; **withdrawn from the package (0.5.0a, 2026-10-06).** The `bench` package left the repository with the notebooks and examples.

## Decision (as it was)

A top-level `bench` package measuring SpaceCore against a hand-optimal pure-array baseline per backend, with fixed-seed factory probes and an aggregation layer for grouped speedups and regressions; the dispatch/cache regimes it timed are gone with ADR-016/022.

## What still holds

The acceptance test it served: a new abstraction must run as fast as the hand-written array version, shown by a benchmark case against raw arrays.

## Where it went

The benchmark suite lives outside the SpaceCore repository.

## Full record

The complete original text (context, design, alternatives, invariants) is in git history: `git show 0ca3f2b:docs/dev/adr/023_benchmark_framework.md`.
