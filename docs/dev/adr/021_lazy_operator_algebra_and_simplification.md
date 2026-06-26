# ADR-021: Lazy operator algebra and simplification

## Status

Proposed

## Context

SpaceCore builds operator expressions lazily: `A @ B`, `A + B`, `c * A`, `A.H`
construct `ComposedLinOp` / `SumLinOp` / `ScaledLinOp` / `_AdjointViewLinOp`
nodes rather than evaluating anything. Two distinct concerns have been
conflated:

1. **Dispatch** — mapping an *operation* to its most efficient *array*
   implementation at apply time (one batched `matmul` vs a per-block loop, an
   SpMV, etc.). This is [ADR-016](016_kernel_layers.md), now scoped to dense and
   sparse array execution, and held to a strict per-call rail: a routed spec
   must be **bit-exact** to the inline path (`rtol == atol == 0`).

2. **Lazy algebra** — simplifying the operator *expression* itself and
   *multiplying operators where possible*: collapse `A @ I @ B` to `A @ B`, fold
   `c · (d · A)` to `(c·d) · A`, multiply two dense operators into one matrix,
   combine like terms. This rewrites the operator *once*, at construction or an
   explicit step — not per call.

These are different mechanisms at different times, and the dispatch exactness
rail is the wrong constraint for the second: multiplying two dense operators and
then applying is not bit-identical to applying them in sequence (floating-point
addition is not associative), so under the ADR-016 rail "multiply operators
where possible" could never fire automatically. Yet a one-time build is not a
per-call equivalence contract — the user who asks to fuse `A @ B` into one
matrix has accepted that single reassociation.

Today the `make_composed` / `make_sum` / `make_scaled` factories already perform
cheap local rewrites at construction (`A @ I → A`, `A @ 0 → 0`,
`c · (d · A) → (c·d) · A`, flatten nested sums and drop zero terms, `(A*)* → A`
via the cached adjoint view). There is no operator *multiplication*
(`dense ∘ dense → one matrix`), no like-term combination, no composition-chain
flattening into a single operator, and no user-facing control over fusion.

## Proposed design

Operator-expression simplification lives in `spacecore.linop` (the `make_*`
factories plus a `fuse` API), **not** in `spacecore.kernels`, and splits into two
tiers.

**Tier 1 — automatic at construction.** Cheap, local, non-materializing rewrites
run inside the `make_*` factories with no caller action. A rewrite is automatic
iff it is `O(1)`-local in the expression and allocates no array proportional to
operand size: identity elision, zero annihilation, scalar folding, sum
flattening + zero-term removal, the adjoint involution. These already largely
exist; this ADR names them a tier and fixes the boundary.

**Tier 2 — explicit, opt-in fusion.** Materializing or contract-changing
rewrites the caller must *request*, through a dedicated method / argument whose
arguments decide what is fused: multiply `dense ∘ dense` into one matrix, form a
Gram product `A.H @ A` as one materialized operator, flatten a composition chain
into a single fused operator, combine like terms (`2A + 3A → 5A`). Surface:
`expr.fuse(...)` and/or a `fuse=` argument on the algebra constructors (e.g.
`fuse="dense"`, `materialize=True`). The materialization itself is the subject of
[ADR-022](022_caching.md) (build it once, amortize over many applies).

**The auto/opt-in boundary is cost and locality, not strict bit-exactness.**
Tier-1 rewrites are cheap and local even when they reassociate at the ulp level
(scalar folding computes `c·d` first, which can differ from `c·(d·…)` by one
rounding); that is acceptable because the user invoking lazy algebra has accepted
construction-time reassociation. Tier-2 fusion is gated on *materialization /
contract change*, not on exactness.

**Matrix-free is a hard rail in both tiers** ([ADR-008](008_linop_subclasses.md)):
a matrix-free operand is never silently densified. Tier-1 rules never materialize
it; Tier-2 fusion of a matrix-free operand requires the caller to explicitly opt
into densification and thereby give up the matrix-free contract.

## Decision

1. Operator-expression simplification is a concern distinct from kernel dispatch
   ([ADR-016](016_kernel_layers.md)); it lives in `spacecore.linop` (the `make_*`
   factories and a `fuse` API), never in `spacecore.kernels`, and is not gated on
   the ADR-016 bit-exact rail.
2. Two tiers: cheap, local, non-materializing rewrites run automatically at
   construction; materializing or contract-changing fusion is explicit and
   argument-driven.
3. The auto/opt-in line is cost and locality. Tier-1 rewrites may reassociate at
   the ulp level; using lazy algebra accepts that.
4. A matrix-free operand is never silently materialized in either tier.

## Rationale

Keeping the runtime dispatch rail strict and exact while giving the user a way to
*buy* operator multiplication when they want it puts each kind of work where it
belongs: construction-time tree rewriting for the algebra, runtime kernel
selection for execution. It also resolves the contradiction that the dispatch
exactness rail otherwise forbids the very "multiply operators where possible"
the lazy algebra is meant to provide.

## Alternatives considered

- **Route operator multiplication through ADR-016 dispatch with `rtol=atol=0`.**
  Rejected: the exact rail forbids the floating-point reassociation operator
  multiplication inevitably introduces, so it would never auto-fire; and a
  one-time build is not a per-call equivalence contract.
- **Always-on eager fusion** (multiply operators at construction). Rejected:
  surprising memory blowups (Gram, `dense ∘ dense`) and silent densification of
  matrix-free operands. Materialization must be opt-in.
- **No fusion at all** (keep only the existing cheap rewrites). Rejected: it
  leaves "multiply operators where possible" — an explicit goal — unaddressed.

## Consequences

- New public API surface: a `fuse` method / argument with documented options;
  each fusion documents what it materializes and what it costs.
- The existing `make_*` cheap rewrites are retroactively framed as Tier 1.
- Tests pin Tier-1 rewrites as result-equivalent (within ulp where they
  reassociate) and Tier-2 fusion as opt-in and matrix-free-safe.
- This ADR is the prerequisite framing for [ADR-022](022_caching.md): the cached
  thing is the Tier-2 materialized form.

## Contributor invariants

- Cheap, local, non-materializing operator rewrites go in the `make_*` factories
  (Tier 1). Materializing or contract-changing fusion is explicit, argument-
  driven, and never automatic (Tier 2).
- A matrix-free operand ([ADR-008](008_linop_subclasses.md)) is never densified
  without explicit caller opt-in.
- Expression simplification is not kernel dispatch: do not add operator-algebra
  rewrites to `spacecore.kernels`, and do not gate them on the ADR-016 exact
  rail.
- A construction-time rewrite must preserve the operator's mathematical action
  (within ulp where it reassociates) and its declared domain/codomain and
  scalar-field/dtype identity.
