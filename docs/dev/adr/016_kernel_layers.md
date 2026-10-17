# ADR-016: Kernel layers

## Status

**Superseded (0.5.0a).** The benchmarked-spec dispatch layer (`spacecore.kernels.specs`, `dispatch`, `dispatch_mode`) was removed on 2026-09-26: its batched specs ran only on eager NumPy, measured mostly slower than the generic loops, and never ran under `jax.jit`. The remaining core-kernel package was folded into the classes it served on 2026-10-06.

## Decision (as it was)

Two layers: check-free core kernels shared by the public methods, and optimized kernel specs chosen by a structural dispatcher with a cost model and a memory gate, off by default.

## What still holds

The check-free half: every public operation is a checked wrapper over an `_<op>_core` method on its own class (ADR-007, ADR-010). Do not re-add dispatch without a benchmark under `jax.jit` showing a win.

## Full record

The complete original text (context, design, alternatives, invariants) is in git history: `git show 0ca3f2b:docs/dev/adr/016_kernel_layers.md`.
