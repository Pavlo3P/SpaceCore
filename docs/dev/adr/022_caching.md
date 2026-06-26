# ADR-022: Caching of materialized and fused operator forms

## Status

Proposed

## Context

Two parts of SpaceCore materialize an array from operands:

- The [ADR-016](016_kernel_layers.md) dispatch folds (uniform block-diagonal and
  stacked / sum-to-single batched `matmul`) stack `K` cached matrices into one
  `(K, m, n)` array per call.
- The [ADR-021](021_lazy_operator_algebra_and_simplification.md) explicit fusions
  (`dense ∘ dense → one matrix`, Gram `A.H @ A`, a flattened composition chain)
  build a fused matrix.

Done per call, this materialization is repeated work. The 0.4.1 dispatch
measurements showed the per-call stacking allocation is a meaningful fraction of
the optimized path and a reason the dense folds do **not** beat the per-block
generic loop on the NumPy/CPU backend: NumPy's small-matmul loop is already
cheap, and re-stacking every apply only adds cost.

ADR-016 already states the principle — a super-linear materialization is a
*build-time* decision amortized over many applies, not a per-call one — but no
caching mechanism exists yet. Two cache candidates surfaced during the 0.4.1
work:

1. caching the **materialized array** (the stacked / fused matrix) on the
   operator, so each apply skips the re-stack; and
2. caching the **dispatch decision** (which spec applies for a fixed operator),
   so each apply skips the `applicable()` re-scan.

The maintainer scoped (2) out for now. This ADR therefore decides (1) — the
build-time materialized form — and records (2) as considered and deferred.

## Proposed design

**What is cached (precisely).** The cache target is the **input-independent
factor** of a batched fold: the stacked operand-matrix array

```python
ops.stack([matrix(p) for p in parts])   # shape (K, r, c)
```

built today on every apply in `batched_matvec` / `batched_right_matmul`
(`spacecore/kernels/specs/_batched.py`). It depends only on the operator's fixed
blocks, so it is identical across applies — yet a uniform block-diagonal /
stacked / sum-to-single operator rebuilds it each call because it has no field to
keep it in. That, and only that, is what gets cached. Explicitly:

- **Cached:** the stacked operand matrices, *per matrix accessor exercised* —
  `_A2` for the forward action (apply), `_A2H` for the Euclidean-adjoint action
  (rapply), and the right-matmul stack. An operator applied in more than one
  direction therefore holds more than one cached stack.
- **Not cached:** the stacked *input* vectors (`ops.stack(list(vecs))`) and the
  `matmul` result — both depend on the call argument and differ every apply.
- **Not new mechanism:** the [ADR-021](021_lazy_operator_algebra_and_simplification.md)
  fusions (`dense ∘ dense`, Gram `A.H @ A`, a flattened chain) already persist
  their materialized matrix in the `DenseLinOp` that `fuse()` returns. They need
  no cache field; they appear here only as the explicit opt-in amortization
  branch below. The genuinely uncached case is the dispatch fold, which
  materializes transiently on the per-call path and returns a result rather than
  an operator.

**Build-time materialized-form cache.** An operator that benefits from a
materialized fast path (uniform block-diagonal, stacked / sum-to-single) caches
the stacked operand-matrix array(s) above once — at construction or first use —
behind two gates:

- the ADR-016 **memory gate**: the predicted `peak_bytes` fits the context's
  memory budget, and a matrix-free operand is never materialized; and
- an **amortization check**: the form is cached only when the operator is
  expected to be applied enough times to repay the materialization, or when the
  caller explicitly opts in via [ADR-021](021_lazy_operator_algebra_and_simplification.md)
  fusion.

Subsequent applies route to the already-materialized array at `O(1)` extra, so
the per-apply cost is one `matmul` with no re-stacking. This is the lever that
makes a materializing fold competitive once it routes — most importantly on
accelerators, where the batched call is the win and the one-time stack is
negligible against many applies.

**Pytree hazard (mandatory).** `BlockDiagonalLinOp` / `StackedLinOp` /
`SumToSingleLinOp` and the dense operators are `@jax_pytree_class`. A cached
array must not corrupt `tree_flatten` / `tree_unflatten`: the cache is a derived,
reconstructable value, so it is either excluded from the pytree (recomputed
lazily after `tree_unflatten`) or carried consistently as aux — never silently
turned into an extra differentiable/transformable leaf. For the same reason the
cache is **excluded from the operator's mathematical identity** (`__eq__` /
`__hash__`), exactly as raw array values are.

**Decision caching (deferred).** Caching the applicable-spec selection per
operator instance — to remove the per-call `applicable()` re-scan over the `K`
blocks — was considered. It is deferred: the maintainer scoped it out, and the
0.4.1 measurement showed the dispatcher's per-call overhead (≈6–7 µs, including a
`psutil` free-memory query) is dominated by the public method's boundary
validation, so decision caching is not the first lever. If it is revisited, the
cache key is the operator's structural signature (invariant for a fixed
operator), the memo lives in the **dispatcher** (not operator/functional class
bodies, per ADR-016), and it must respect the dynamic `dispatch_mode` (cache only
the `"on"` selection; `"verify"` must always run both; the cheap mode check still
runs each call).

## Decision

1. The stacked operand-matrix array(s) of a batched fold — the input-independent
   `ops.stack([matrix(p) for p in parts])`, per accessor (`_A2`, `_A2H`,
   right-matmul) — may be cached on the operator as a derived value, gated by the
   ADR-016 memory gate **plus** an amortization check, and excluded from operator
   identity and (pytree-safely) from `tree_flatten`. The input-vector stack and
   the result are never cached. ADR-021 fusions already persist their matrix in
   the returned `DenseLinOp` and need no new field.
2. Caching never materializes a matrix-free operand ([ADR-008](008_linop_subclasses.md)).
3. Dispatch-decision caching is out of scope for now. If added later, it lives in
   the dispatcher, is keyed by structural signature, and respects `dispatch_mode`.

## Rationale

Turning a repeated per-call materialization into a one-time build amortized over
many applies is the ADR-016 intent made concrete, and the measured way to make a
materializing fold pay for itself once it routes. Keeping the cache a property of
the operator (not the per-call path) preserves the matrix-free rail and the
operator's mathematical identity, and keeps the dispatcher stateless.

## Alternatives considered

- **Per-call materialization (status quo).** Correct but re-stacks every apply —
  the measured reason the dense folds lose on CPU.
- **Eager, always-on materialization at construction.** Rejected: memory blowups
  and silent densification of matrix-free operands; must be gated, amortized, and
  opt-in.
- **Decision caching as the primary lever.** Deferred: not the dominant cost per
  the 0.4.1 measurement, and explicitly scoped out by the maintainer.

## Consequences

- A cache field (one slot per stacked accessor — `_A2`, `_A2H`, right-matmul) on
  the block-diagonal / stacked / sum-to-single operators, with pytree-safe
  handling and identity exclusion.
- An amortization heuristic or explicit opt-in, tied to ADR-021 fusion.
- Tests pin: cached stack == the per-call `ops.stack(...)` and cached apply
  result == uncached result; the cache is excluded from `__eq__` / `__hash__` and
  survives a pytree round-trip; a fold containing a matrix-free operand is never
  cache-materialized.

## Contributor invariants

- A cached materialized form is derived and reconstructable: it is excluded from
  `__eq__` / `__hash__` and from the operator's mathematical identity, and a
  pytree round-trip must not depend on it.
- Materialization caching obeys the ADR-016 memory gate and never densifies a
  matrix-free operand ([ADR-008](008_linop_subclasses.md)).
- Decision caching, if ever added, lives in the dispatcher (not operator /
  functional class bodies, per [ADR-016](016_kernel_layers.md)) and respects the
  active `dispatch_mode`.
