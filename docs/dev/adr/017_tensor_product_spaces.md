# ADR-017: Tensor-product spaces

## Status

Deferred. The direct-product vs tensor-product **boundary** is settled here; the
tensor-product **implementation** is deferred and demand-gated (post-0.4.0 plan
§5.4 and §7).

## Context

[ADR-013](013_tree_structured_spaces.md) added `TreeSpace` as SpaceCore's one
structured finite space and was careful to call it a *direct* (Cartesian) product,
explicitly **not** a tensor product. That left an open question recorded in
`current.md`: does SpaceCore need a separate tensor-product abstraction, and where
is the line between the two? This ADR answers the boundary question without
obligating the implementation, which remains a new-core-abstraction track held to
a high bar (a concrete downstream use case plus an accepted design).

## Current design

SpaceCore has only the direct product. `TreeSpace` composes leaf spaces so that
dimension **adds**, elements are tuples/trees of leaf elements, inner products sum
leaf inner products, and all operations are leafwise (ADR-013). There is no
tensor-product space and no Kronecker/tensor operator. A user who needs a
Kronecker action today composes it by hand as a `MatrixFreeLinOp`.

## Decision

The boundary is fixed by dimension and operator structure:

- **Direct product (`TreeSpace`, shipped).** Use when a variable is a collection
  of blocks combined additively — block vectors, parameter trees, multi-physics
  states. Dimension is the **sum** of leaf dimensions; geometry and operations are
  leafwise.
- **Tensor product (reserved).** Use when a variable is a genuine multi-index
  object whose operators **factor across axes** — separable operators, Kronecker
  covariances, low-rank tensor formats, tensor products of subsystems. Dimension
  is the **product** of factor dimensions; the value is exploiting that
  factorization instead of materializing the product-dimensional operator.

`TreeSpace` must **not** be overloaded to carry tensor semantics: the dimension
and inner-product rules differ, and conflating them would corrupt the
direct-product contract.

When (and only when) the track is activated, the tensor-product abstraction is a
**new core abstraction**, not an extension of `TreeSpace`, and it has **two
variants** mirroring the homogeneous/heterogeneous split already familiar from
batching and the direct product:

- **Uniform tensor power (homogeneous).** `X^⊗n` — `n` copies of one factor
  space. Backed by a single dense `n`-mode array and acted on uniformly along each
  mode. This is the optimized case: identical factors permit contiguous storage
  and vectorized per-mode application, close in spirit to leading-axis batching
  ([ADR-006](006_current_batching_model.md)) but along genuine tensor modes rather
  than a batch axis, and it is where symmetry (symmetric/antisymmetric subspaces)
  would live. This is the "same spaces, optimized" half.
- **Heterogeneous tensor product.** `X₁ ⊗ X₂ ⊗ …` — distinct factor spaces.
  Structured like `TreeSpace` is for the *direct* product: the abstraction tracks
  the ordered, possibly-different factor spaces, their individual geometry, and
  factor-wise operators. This is the general "tree-like" half.

Both store elements as dense tensors with multiplicative dimension; the split is
whether the factor spaces — and therefore their geometry and admissible
optimizations — are identical or differ. The supporting operators are Kronecker /
tensor LinOps (e.g. `KroneckerLinOp`) that apply `A₁ ⊗ A₂ ⊗ …` factor-wise without
forming the Kronecker product, with the metric-aware adjoint being the tensor of
factor adjoints; the homogeneous variant may expose a dedicated optimized path
while the heterogeneous variant carries per-factor metadata. In both, the induced
inner product is the tensor product of factor inners and the Riesz map is the
tensor of factor Riesz maps.

Activation is gated on **both** an accepted design that resolves the geometry,
adjoint, and batching interactions **and** a concrete downstream workload
(typically SDPLab). Until then, no `TensorProductSpace` ships and Kronecker
actions are built by hand as `MatrixFreeLinOp` operators — the documented interim.

## Rationale

A tensor product is mathematically distinct from a direct product (multiplicative
dimension, factorizable operators, multilinear structure), so it cannot be a mode
of `TreeSpace` without breaking that space's semantics. Per the sequencing
principles, a new core abstraction needs an accepted design and a real use case
before implementation; settling the boundary now closes the open `current.md`
question and tells users exactly which existing tool to reach for, without
committing to speculative structure.

The homogeneous/heterogeneous split within the tensor product reuses a distinction
that already pays off elsewhere — uniform stacking vs `TreeSpace` for direct
products, and the leading-axis batch for repeated work. Letting the common
same-factor case (tensor powers, quantum subsystems, symmetric tensors) hit a
dense, vectorized per-mode path keeps it from paying the general structured cost,
while the heterogeneous variant remains available for genuinely mixed factors.

## Alternatives considered

Overloading `TreeSpace` to express tensor products was rejected: the dimension and
inner-product semantics differ and the direct-product contract would be corrupted.
Building `TensorProductSpace` and Kronecker LinOps speculatively now was rejected:
there is no concrete use case yet, and it adds a core abstraction against the
"stop adding structure, start hardening it" stance. Forcing users to flatten to a
`DenseCoordinateSpace` and hand-build Kronecker operators *forever* was rejected as
the long-term answer, but is accepted as the interim until the track activates.

## Consequences

The open tensor-product question in `current.md` moves from "undecided" to
"boundary defined, implementation gated." When activated, the track gets its own
minor-release section and must specify the geometry, metric adjoint, and
batching-axis interactions in full — it touches the space hierarchy
([ADR-003](003_space_hierarchy.md)), geometry ([ADR-004](004_inner_product_and_geometry.md)),
batching ([ADR-006](006_current_batching_model.md)), LinOp subclasses
([ADR-008](008_linop_subclasses.md)), and the direct-product space
([ADR-013](013_tree_structured_spaces.md)). A tensor-product axis is distinct from
the leading batch axis and any future design must keep them separate.

## Contributor invariants

- `TreeSpace` is a finite direct product; it must not be extended to express
  tensor products.
- No `TensorProductSpace` or Kronecker/tensor LinOp implementation lands until
  this ADR's design is accepted and a concrete downstream use case exists.
- Direct-product dimension **adds**; tensor-product dimension **multiplies** — the
  two abstractions are never merged on the basis of surface similarity.
- The uniform tensor power (identical factors) and the heterogeneous tensor
  product are distinct variants: the homogeneous one may assume identical factor
  geometry and use a dense per-mode path; the heterogeneous one must carry
  per-factor spaces and geometry, like `TreeSpace`.
- A tensor-product axis is not a batch axis ([ADR-006](006_current_batching_model.md)),
  even though the homogeneous variant resembles batching.
- Until the track activates, Kronecker actions are expressed as `MatrixFreeLinOp`,
  not as a bespoke space.
