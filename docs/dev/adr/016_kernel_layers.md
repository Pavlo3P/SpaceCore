# ADR-016: Kernel layers and dispatch policy

## Status

Accepted for the kernel-layer architecture. The optimized-kernel **dispatch
policy** (see "Decision — optimized-kernel dispatch" below) is **Proposed**: a
broader structural-dispatch design is recorded for acceptance. No dispatch code
ships until this section is marked Accepted.

## Context

SpaceCore needs optimized execution paths without polluting operator and
functional class bodies with optimization logic or special-casing, and without
compromising the matrix-free, geometry-aware contracts ([ADR-007](007_linop_contract.md),
[ADR-009](009_metric_adjoint.md), [ADR-010](010_functional_contract.md),
[ADR-011](011_linalg_contract.md)). The `0.4.0`
hot-path acceleration report (`docs/dev/0.4.0-hotpath-acceleration-report.md`)
catalogued the shipped mechanism and found that the heavier optimized-kernel
catalog, while correctness-verified, is not yet routed to by anything. This ADR
records the implemented kernel architecture and reserves the open decision about
automatic dispatch, which previously held the ADR-016 slot on its own.

## Current design

`spacecore.kernels` keeps optimization logic out of the operator/functional class
bodies and splits it into two layers.

**Core-kernel layer** (`spacecore.kernels.core`). The check-free cores of LinOp
`apply`/`rapply`/`vapply`/`rvapply` and Functional `value`/`grad`/`vvalue`/`vgrad`,
written as concrete functions grouped by kind (`algebra`, `dense`, `diagonal`,
`sparse`, `functional`). An operator binds a core-kernel set at class-definition
time through the `@core_kernels` decorator; binding is static, so consuming a core
costs nothing at call time. These cores **are** on the default apply/eval path:
the public method validates its boundary per the check policy
([ADR-014](014_check_policy.md)), then calls the check-free core. Importing the
subpackage registers every core-kernel set.

**Benchmarked-spec layer** (`spacecore.kernels.specs`). Heavier, opt-in fast paths
described by `KernelSpec`. Each spec bundles a stable name, a `generic` reference
implementation, an `optimized` implementation, an `applicable` predicate, a
`correctness_ref` (the pytest node id pinning `optimized` against `generic` over
the generated case set), a `benchmark_id`, and `rtol`/`atol`. Registration
enforces the rails: a spec without a `correctness_ref` or `benchmark_id` raises
`MissingReferenceError` / `MissingBenchmarkError`. The registry only *catalogs*
specs (`all()`, `get()`, `names()`) — **there is no selection or dispatch logic;
nothing auto-dispatches.** Two specs ship: `composed-chain-apply` and
`block-diagonal-dense-apply`. The contract is documented in
`docs/source/design/kernels_policy.rst`.

## Decision

1. Optimization logic lives in `spacecore.kernels`, never in operator or
   functional class bodies.
2. Two layers with different call-time guarantees: check-free cores on the
   default path (statically bound, zero call-time cost), and benchmarked specs
   that are catalogued but selected only by explicit, clearly-scoped call sites.
3. Every `KernelSpec` ties a generic reference, an applicability predicate, a
   correctness test, and a benchmark before it may register; tolerances are tight
   by default and loosened only when an underlying backend op already disagrees.

## Decision — optimized-kernel dispatch (Proposed)

SpaceCore adopts a **broader structural-dispatch** system over a narrow
fixed-fusion system: the benchmarked-spec layer becomes routable through a single
dispatcher that, at an instrumented call site, selects an applicable optimized
spec from the registry by structural match — rather than each call site naming a
spec or hard-coding a rewrite. This keeps the selection logic in one place,
admits third-party specs without touching call sites, and preserves the existing
correctness rails. The cost — a real correctness surface and the risk of routing
to a wrong fast path — is bought down by the rails below, not by narrowing scope.

**Dispatch key.** `KernelSpec` gains a `dispatch_key: str` naming the operation
*family* a call site requests (e.g. `"linop.composed.apply"`,
`"linop.block_diagonal.apply"`) and an integer `priority` (default `0`). `name`
stays the unique identity; many specs may share a `dispatch_key`. The registry
builds a `dispatch_key → specs sorted by descending priority` index alongside the
existing name map; `all()`/`get()`/`names()` are unchanged.

**Dispatcher.** A single entry point `dispatch(key, *args, generic) -> result`
walks the specs under `key` in priority order, returns `spec.optimized(*args)`
for the first whose `applicable(*args)` is `True`, and calls `generic(*args)` when
none applies. Selection is deterministic: two dispatch-eligible specs that are
simultaneously applicable at equal priority is a registration-time error, never a
silent pick. The dispatcher holds *all* structural-selection logic; operator and
functional bodies still contain none.

**Call-site integration.** The hot internal sites approved here — the composed
apply chain and the block-diagonal dense apply identified in the `0.4.0`
hot-path acceleration report — change from their inline path to
`dispatch(key, …, generic=<the inline path>)`. The inline path *becomes* the
`generic` fallback, so dispatch-off is byte-identical to today. This is the line
between this decision and the prior opt-in policy: the call site delegates
*selection*, it does not name a spec.

**Mode and correctness rails.** A process- and context-level `dispatch_mode`:

- `off` — always run `generic`; the regression baseline and the default until a
  key's specs are proven and benchmarked.
- `on` — route to the applicable optimized spec.
- `verify` — run both, assert agreement within the spec's `rtol`/`atol`, raise on
  mismatch. ADR-014 `check_level="strict"` implies `verify`.

A spec is **dispatch-eligible** only with `rtol == atol == 0` (exact equivalence);
specs carrying loosened tolerances may register and be called explicitly but are
never auto-routed. The existing rails still hold: registration without a
`correctness_ref` or `benchmark_id` raises (`MissingReferenceError` /
`MissingBenchmarkError`), and CI runs each dispatch-eligible spec in `verify`
against its generated case set plus a before/after benchmark probe.

**Cost and memory gate.** `applicable` proves a rewrite is *valid*; it does not
prove it is *affordable*. Any spec whose optimized path allocates more than O(1)
extra memory — forming a Gram matrix `AᵀA`, multiplying `dense ∘ dense` into one
matrix, stacking blocks for a batched call — carries a
`cost(*args) -> KernelCost` returning the optimized path's predicted peak extra
`bytes` and `flops`, computed **from shapes and dtypes only**: never from operand
data, never by touching the arrays. The dispatcher owns the **memory** gate,
because free memory is a shared runtime resource — a spec is selected only when
its estimated `peak_bytes` fits the context's memory budget (a configurable
fraction of backend-reported free memory); otherwise the dispatcher falls through
to the next spec, then to `generic`. **Compute** profitability is the spec
author's responsibility, encoded in `applicable` from shapes: a fusion that would
do more flops than the generic path must report itself inapplicable. If `cost`
cannot produce an estimate (unknown or symbolic shapes), the spec is treated as
unaffordable and skipped — **no estimate, no fuse.**

Two consequences follow. First, a super-linear materialization (Gram, dense
product) is a *build-time* decision amortized over many applies, not a per-call
one: such specs form their array once — at construction or an explicit fuse step,
behind the same memory gate plus an amortization check — and the per-apply
dispatcher then routes to the already-materialized path at O(1). Exact,
non-allocating Tier-1 specs (chain apply, identity elision) need no `cost`;
bounded-transient ones (batched stacking) carry a `cost` whose peak is a small
multiple of the operand size. Second, an operand that is matrix-free
([ADR-008](008_linop_subclasses.md)) must **never** be silently materialized into
a dense fused form regardless of budget — that abandons the matrix-free contract;
its `cost` reports prohibitive and the fusion is available only when the caller
explicitly opts into materialization.

This section is **Proposed**. It is the prerequisite the post-0.4.0 plan §5.1
names; no dispatcher, key, or call-site rewiring is implemented until it is marked
Accepted.

## Rationale

Keeping optimization out of class bodies leaves the operator/functional code
math-first and readable. Static core binding delivers that separation at zero
runtime cost. The correctness-plus-benchmark rails stop unverified fast paths from
entering the catalog. Deferring dispatch lets the verified kernels ship now
without committing to a fusion/dispatch architecture before a design exists.

## Alternatives considered

Inlining optimizations into operator methods was rejected: it reintroduces the
special-casing the kernel layer exists to remove and obscures the mathematical
contract. Auto-dispatching the spec layer immediately was rejected: it presumes
the reserved dispatch design and risks silently routing to a wrong fast path. A
single flat kernel layer was rejected because the check-free cores and the
benchmarked specs have different costs, lifecycles, and call-time guarantees.

## Consequences

Adding a core kernel is a function plus a `@core_kernels` binding; it must
preserve the public method's validated contract and the matrix-free/geometry
contracts. Adding a fast path is a `KernelSpec` carrying the correctness and
benchmark rails; registration fails without a correctness reference and a
benchmark id, and a materializing spec additionally needs a shape-only `cost`
before it may auto-dispatch. Activating dispatch requires the dispatch decision
above to be accepted first.

## Contributor invariants

- Optimization logic goes in `spacecore.kernels`, not in `LinOp`/`Functional`
  class bodies.
- Core kernels are check-free: the public method owns boundary validation
  ([ADR-014](014_check_policy.md)); the core assumes validated inputs and must
  reproduce the unoptimized behavior.
- A `KernelSpec` without a `correctness_ref` or a `benchmark_id` must not
  register.
- An optimized spec is numerically equivalent to its `generic` reference within
  the documented tolerance whenever `applicable` returns `True`.
- Affordability is decided **before** the fusion runs, never after: a spec whose
  optimized path allocates more than O(1) extra memory must carry a shape-only
  `cost`, and the dispatcher checks its `peak_bytes` against the memory budget
  before selecting it. A materializing spec with no `cost`, or whose `cost`
  returns no estimate, must not auto-dispatch — no estimate, no fuse.
- Cost estimation reads shapes and dtypes only; it must never touch operand data
  or allocate the result it is estimating. Compute profitability lives in
  `applicable`: a fusion that costs more flops than `generic` reports itself
  inapplicable.
- A matrix-free operand ([ADR-008](008_linop_subclasses.md)) is never silently
  materialized into a dense fused form; such a fusion is reachable only by
  explicit caller opt-in, regardless of available memory.
- Nothing auto-dispatches to the benchmarked-spec layer until the dispatch policy
  in this ADR is accepted; fast paths are selected explicitly by scoped call
  sites.
