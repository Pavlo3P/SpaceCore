# SpaceCore vision

## Purpose

SpaceCore gives typed mathematical structure to linear maps, spaces, inner
products, and geometry-aware algorithms across numerical backends. Its role is
to make domain, codomain, scalar field, shape, context, and adjoint semantics
explicit so code can move between NumPy, JAX, Torch, and related backends
without losing the mathematical contract.

SpaceCore should not hide problem-specific mathematics from the caller. It
should give callers precise places to state that mathematics and reusable
machinery to validate it.

## Non-goals

SpaceCore does not provide problem-domain abstractions such as PDE models,
statistical estimators, optimal-control objects, or semidefinite-programming
frontends. It does not hide mathematical work behind convenience constructors
that guess geometry, infer adjoints, or silently select problem formulations.

It is also not intended to become a general scientific-computing toolkit. Array
libraries and domain packages remain the right place for broad array operations,
dense numerical kernels, plotting, data management, and specialized solver
ecosystems.

## Ecosystem position

SpaceCore is intended to sit between array libraries and domain-specific
scientific or optimization libraries. Its tentative role is to provide a small
core of explicit mathematical contracts: spaces, elements, scalar fields,
contexts, inner products, Riesz maps, linear operators, functionals, and
geometry-aware algorithms.

This position is provisional. The exact boundary should be refined through
examples, interoperability work, and downstream use. In particular, SpaceCore
should avoid copying broad APIs from adjacent projects before there is a clear
mathematical reason to do so. The current priority is to keep the core small 
enough that its invariants remain visible and reviewable.

A likely useful boundary is the following: SpaceCore owns the typed mathematical
structure of linear and functional-analytic objects; downstream packages own
problem-specific modeling, discretization choices, application-level solvers,
and domain terminology. When interoperability adapters are added, they should
document which information is preserved and which information is lost, especially
for non-Euclidean geometry, structured elements, backend context, and generalized
adjoints.

## Release sequencing

Contributor infrastructure in 0.3.2 precedes the test-infrastructure expansion
planned for 0.4.0 because contributors cannot build reliable test generators for
a library they cannot navigate. Before adding broad generated tests, SpaceCore
needs accepted contributor process documents, architecture records, docstring
standards, and release-gate rules.

Those documents define where contracts live, which invariants reviewers must
check, and how later test generators should encode expected behavior. The 0.4.0
test work can then target stable documents instead of reverse-engineering intent
from scattered code.

## 1.0.0 definition

SpaceCore reaches 1.0.0 when its core abstractions are exercised in realistic
internal and external examples; backend contracts are stable and documented;
extension APIs are demonstrated outside the core package; interoperability
adapters are reliable for the supported external ecosystems; benchmarks and
regression tracking are established; a deprecation policy is defined and
followed; and the implementation paper is submitted or publication-ready.
