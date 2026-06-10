# ADR-002: Context and conversion

## Status

Accepted

## Context

Spaces, operators, and functionals need a common execution context for backend operations, default dtype, and validation behavior. Conversion must be predictable because hidden array movement or casting can change performance, autograd behavior, and numerical results.

## Current design

`Context` stores `ops`, `dtype`, and `enable_checks`. It does not own arrays, devices, sparse storage, or gradient state. Context normalization accepts a concrete `Context`, backend family/name, or `None`. Concrete contexts are copied with sanitized dtype. Backend names create a new context through the registered backend. `None` resolves to the process default.

`Context.asarray` and `Context.assparse` convert explicit values into the context. `Context.convert` dispatches dense and sparse inputs through those constructors. `ContextBound.convert(new_ctx)` rebuilds a context-bound object in a target context through `_convert`; it returns `self` when the normalized context is equal. Constructors resolve explicit context first, then compatible inferred contexts from operands, then the default context.

The current design has no separate conversion policy object. Conversion policy lives in explicit `convert(...)`, `Context.asarray(...)`, `Context.assparse(...)`, and constructor resolution.

## Decision

Conversion is explicit and target-context driven. User values passed to operations such as `apply`, `add`, or `inner` are validated when checks are enabled but are not silently converted. Object conversion rebuilds spaces, geometries, stored matrices, and algebraic operands in the requested context; matrix-free callables are preserved and must already be valid for the target backend.

## Rationale

Explicit conversion avoids hidden device transfers, sparse format changes, dtype casts, and backend-specific callable failures. Keeping context small also makes equality and algebra compatibility checks straightforward.

## Alternatives considered

Implicitly converting every operation input was rejected because it hides expensive transfers and can mask caller bugs. A separate conversion policy hierarchy was rejected for now because current behavior is simpler and has only one public mode: explicit target conversion.

## Consequences

New context-bound classes must implement `_convert` when they store context-owned data. Conversions may fail when the target backend lacks sparse support, dtype support, or callable compatibility. Dtype defaults are representation defaults, not mathematical fields; see [ADR-015](015_dtype_default_vs_scalar_field.md).

## Contributor invariants

- Context conversion must be explicit and predictable.
- `Context` owns backend, dtype default, and validation flag only.
- Constructors may convert their owned storage, but runtime operation arguments must not be silently converted.
- Conversion must preserve mathematical structure while rebuilding backend-owned representation.
