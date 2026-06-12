# ADR-014: Check policy

## Status

Accepted and implemented for 0.4.0 Phase B.

## Context

Current validation is controlled by boolean `Context.enable_checks`. That is useful but too coarse for 0.4.0 test generation, where cheap shape checks, standard membership checks, and strict expensive mathematical checks need distinct expectations.

## Current design

Spaces attach `SpaceCheck` objects such as backend, shape, dtype, square-matrix, Hermitian, product-structure, and product-component checks. `checked_method` runs input and output membership checks when `_enable_checks` is true. Batched checks allow leading axes while enforcing trailing element shape. Context inference combines boolean check flags with logical `and`.

## Decision

The public API is ``Context.check_level`` with the exported
``CheckLevel = Literal["none", "cheap", "standard", "strict"]`` type. This
keeps the existing context-first constructor architecture intact: spaces,
LinOps, and functionals continue to receive a context rather than duplicating
policy keywords on every constructor.

The intended policy levels are:

- `none`: no SpaceCore validation except unavoidable constructor normalization; backend errors may surface later.
- `cheap`: enforce local non-allocating invariants such as backend family, dense/sparse type, rank/shape, dtype, product arity, and obvious domain/codomain compatibility.
- `standard`: current contributor-facing safety level; include component membership, constructor storage checks, scalar output shape, Hermitian structure when explicitly configured, and conversion consistency.
- `strict`: allow expensive or numerical checks such as full adjoint identities, basis-based Hermiticity, positive-definiteness probes, batched/single consistency, and cross-backend conformance checks.

Boolean `enable_checks` is a deprecated compatibility shim: `False` maps to
`none`; `True` maps to `standard`; passing it together with `check_level` raises
`TypeError`.

## Rationale

A boolean cannot distinguish checks that should run in every debug build from checks that are too expensive for hot loops or require numerical tolerances. Test generation needs stable names for these categories.

## Alternatives considered

Keeping only `enable_checks` was rejected because it cannot express strict conformance tests without penalizing normal runtime. Running all checks whenever enabled was rejected because expensive numerical checks are not appropriate inside core methods.

## Consequences

0.4.0 work should introduce a check-level representation and classify existing checks. Tests should assert which invariants belong to which level for spaces, LinOps, functionals, batching, context conversion, and dtype/field behavior. Until implemented, code remains boolean-gated.

## Contributor invariants

- Do not add expensive numerical checks to the current hot-path boolean gate without an explicit level decision.
- Cheap checks must be deterministic and local.
- Standard checks should match current `enable_checks=True` expectations unless documented otherwise.
- Strict checks belong in conformance/test helpers or explicit APIs, not implicit inner loops.
