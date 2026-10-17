# ADR-014: Check policy

## Status

Accepted and implemented for 0.4.0 Phase B.

## Context

Before 0.4.0, validation was controlled by boolean `Context.enable_checks`. That was useful but too coarse for 0.4.0 test generation, where cheap shape checks, standard membership checks, and strict expensive mathematical checks need distinct expectations.

## Current design

Spaces attach `SpaceCheck` objects (backend, shape, dtype, square-matrix, Hermitian, tree-structure and leaf checks), each with a `minimum_level`. Every public element operation is a `checked_method` wrapper over a check-free `_<op>_core`; the wrapper runs the input and output membership checks the object's `check_level` includes. Batched checks allow leading axes while enforcing trailing element shape. While a backend traces (`jax.jit`, `jax.vmap`, `torch.compile`), the level is capped by `set_traced_check_level`, default `"none"`.

## Decision

The level is ``CheckLevel = Literal["none", "cheap", "standard", "strict"]``.
It was first a field of ``Context``; since 0.4.3 it is a property of each
space, operator and functional (``check_level=`` at construction), seeded from
the ambient level (``get_check_level`` / ``set_check_level`` /
``use_check_level``). A container combines its operands' levels by the minimum.

The intended policy levels are:

- `none`: no SpaceCore validation except unavoidable constructor normalization; backend errors may surface later.
- `cheap`: enforce local non-allocating invariants such as backend family, dense/sparse type, rank/shape, dtype, product arity, and obvious domain/codomain compatibility.
- `standard`: current contributor-facing safety level; include component membership, constructor storage checks, scalar output shape, Hermitian structure when explicitly configured, and conversion consistency.
- `strict`: allow expensive or numerical checks such as full adjoint identities, basis-based Hermiticity, positive-definiteness probes, batched/single consistency, and cross-backend conformance checks.

Boolean `enable_checks` was a deprecated compatibility shim (`False` mapped to
`none`, `True` to `standard`); it was removed in 0.5.0a, and `check_level` is the
only spelling.

## Rationale

A boolean cannot distinguish checks that should run in every debug build from checks that are too expensive for hot loops or require numerical tolerances. Test generation needs stable names for these categories.

## Alternatives considered

Keeping only `enable_checks` was rejected because it cannot express strict conformance tests without penalizing normal runtime. Running all checks whenever enabled was rejected because expensive numerical checks are not appropriate inside core methods.

## Consequences

Every check is classified by level, and tests assert which invariants belong to which level for spaces, LinOps, functionals, batching, context conversion, and dtype/field behavior.

## Contributor invariants

- Do not add an expensive numerical check below `strict` without an explicit level decision.
- Cheap checks must be deterministic and local.
- Strict checks belong in conformance/test helpers or explicit APIs, not implicit inner loops.

### Functional real-output invariant (0.5.0a)

The real-valued Functional contract in ADR-010 is independent of check policy.
Even `none` rejects complex codomains, coefficients and output dtypes, including
batch and fused evaluations. These checks use static type/dtype metadata, not
array values, so they also apply while tracing compiled code. Scalar shape and
domain membership checks continue to follow the selected validation level.
