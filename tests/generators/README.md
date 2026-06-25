# Test generators

This package provides deterministic input and reference-data builders for the
SpaceCore test suite. It is infrastructure for later conformance phases, not a
property-testing framework and not a replacement for focused unit tests.

## Adding a generator

Return a `GeneratedCase` whose `obj` is the backend object under test and whose
`reference` mapping contains NumPy data or other cheap reference results.
Advertise only capabilities that callers can rely on, and give every reusable
case a stable, readable `id`. Keep pytest marks on the case so parameter helpers
can preserve skips without backend-specific branching in tests.

Every random generator accepts either `seed` or `rng`, never both. The default
is `DEFAULT_SEED`; `seeded_rng()` creates an independent NumPy generator. Do not
use `numpy.random` module state or backend-global random state.

## Shared policies

- Check levels are exactly `none`, `cheap`, `standard`, and `strict`.
  `check_level_params()` rejects unknown values instead of accepting them
  silently.
- NumPy `float64` and `complex128` contexts are always generated. Optional JAX,
  Torch, and CuPy contexts are used only when operational. Unavailable optional
  contexts are represented by explicit pytest skip marks.
- Real contexts receive only real-valued input. Complex contexts may receive
  complex-valued input. Generators reject complex data requested for a real
  context, preserving SpaceCore's no-narrowing policy.
- A batch is a leading prefix on dense values. For `TreeSpace`, a batch is a
  tree of leaves that each carry the same leading batch shape, matching ADR-006.
- Structured data uses `TreeSpace` only. Do not add `ProductSpace` compatibility
  paths or assumptions.

Phase F smoke tests validate these builders themselves. Broad Space, LinOp,
Functional, linalg, backend, batching, and check-policy conformance belongs in
the later generated-coverage phases.

## Space cases

`tests.generators.spaces` builds Phase G cases for dense coordinate spaces,
dense vectors, Euclidean and full-SPD inner-product spaces, TreeSpace values,
Jordan spaces, and generic vector-space laws. Each case records its context,
dtype, scalar field, check level, target conversion context, valid values,
invalid values, and leading-axis batches in `GeneratedCase.reference`.

Phase G deliberately uses NumPy `float32`, `float64`, `complex64`, and
`complex128` cases. Optional backend behavior remains backend-conformance work.
`MatrixInnerProduct` and the private coordinate-only leaf are narrow test
helpers: the former exercises non-diagonal SPD Riesz maps, while the latter
allows TreeSpace capability intersections to be tested without claiming an
inner product on every leaf.

## Functional cases

`tests.generators.functionals` builds scalar-valued reference cases for real
and complex NumPy contexts. Cases cover zero and nonzero linear functionals,
quadratic forms, generic and specialized pull-backs, Euclidean and weighted
inner products, and tuple-structured `TreeSpace` domains. The reference mapping
stores the deterministic input, direct value, Riesz gradient when available,
pull-back data when supported, conversion context, and check level. Coordinate
gradients are stored separately for weighted cases so tests cannot accidentally
validate a Euclidean gradient in non-Euclidean geometry.

## LinOp cases

`tests.generators.linops` covers every concrete public LinOp family. Each case
stores scalar and leading-axis batch inputs, direct forward and metric-adjoint
references, an optional coordinate matrix, and conversion/batching capability
flags. Tree-structured cases use `TreeSpace` only and include stacked,
sum-to-single, block-diagonal, and block-matrix operators.

For a coordinate map `A : X -> Y`, non-Euclidean references must use
`G_X^-1 A^H G_Y`. A plain conjugate transpose is valid only when both spaces
are Euclidean. Matrix-free cases must supply a callable that already implements
the true metric adjoint.
