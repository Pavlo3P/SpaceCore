# ADR-015: Dtype default versus scalar field

## Status

Accepted

## Context

Current contexts use a backend dtype as a representation default. 0.4.0 conformance needs a clearer distinction between representation dtype and mathematical scalar field.

## Current design

`Context.dtype` is backend-normalized by `ops.sanitize_dtype`. Context array constructors use it by default. Spaces use `space.dtype` for zeros, ones, unflattening, and membership checks. Dtype membership is strict: `DTypeCheck` requires exact equality with `space.dtype`. Constructors and conversions preserve or explicitly convert owned arrays through the target context. Backend helpers expose complex detection and real-dtype derivation for solver workspaces.

Some mathematical constraints are already dtype-sensitive. `EuclideanElementwiseJordanSpace` requires real dtype and Euclidean geometry. `WeightedInnerProduct` weights must match the space dtype while being real-valued, finite, and positive.

## Decision

`Context.dtype` is a representation default, not the full mathematical scalar-field contract. `Space.field` describes whether membership is over a real or complex scalar field. Dtype checks remain representation checks. Field-level checks decide whether complex values are mathematically allowed, while exact dtype checks continue to decide representation compatibility.

For 0.4.0 Stage 1, `Space.field` is derived from `Context.dtype`; no constructor-level override is provided. `FieldCheck` makes scalar-field compatibility explicit while `DTypeCheck` continues to enforce exact representation dtype. Conversion rejects complex-to-real narrowing unless the caller explicitly extracts a real part first.

Exact dtype equality and cross-precision compatibility remain strict until concrete workloads justify relaxing them. Solver workspaces should be operand-driven: real diagnostics for complex operators use `ops.real_dtype(ctx.dtype)`, while vector workspaces use the operator/context dtype.

## Rationale

Dtype and scalar field answer different questions. Dtype controls storage and backend kernels; field controls mathematical membership and valid operations. Conflating them makes complex membership and precision compatibility ambiguous.

## Alternatives considered

Using exact dtype as the only field proxy was rejected because real-vs-complex membership is mathematical, not just storage. Allowing broad cross-precision compatibility now was rejected because it risks silent promotion differences across NumPy, JAX, Torch, and CuPy without a workload-driven policy.

## Consequences

0.4.0 tests should cover representation defaults, dtype-preserving construction, strict dtype membership, real vs complex field membership, weighted geometry checks, Euclidean-Jordan dtype restrictions, and solver workspace dtype choices. See [ADR-002](002_context_and_conversion.md) and [ADR-003](003_space_hierarchy.md).

## Contributor invariants

- `Context.dtype` controls representation defaults and constructors.
- `Space.field` is the mathematical contract for real vs complex membership and is derived from the context dtype in 0.4.0.
- Current dtype membership remains exact and strict.
- Solver scalar diagnostics should use real workspaces when mathematically real, especially for complex operators.
