Phase G plan: Space coverage through generated tests.

Goal:
Use the Phase F generator infrastructure to systematically test SpaceCore spaces as mathematical vector spaces over a scalar field (F \in {\mathbb{R}, \mathbb{C}}), with optional inner-product, star, Jordan, spectral, tree, batching, and conversion structure.

Scope:
Implement Phase G only. Do not implement new LinOp generators, Functional generators, linalg tests, or backend-conformance tests except where a minimal helper is needed for space generation.

Important adjustment:
The old task list mentions a `ProductSpace` generator. ProductSpace has now been removed. Replace that item with TreeSpace tuple/direct-product coverage. Do not reintroduce ProductSpace, aliases, shims, tests, or docs.

Use only `.venv`:

* `.venv/bin/python -m pytest ...`
* `.venv/bin/python -m ruff check .`

Inputs:

* Phase F generator package.
* Existing space implementations.
* `docs/dev/docstring_style.md`.
* ADR-006 batching model.
* ADR-014 check policy.
* ADR-015 dtype/field policy.
* Current TreeSpace docs/tests.

Graph navigation:
Use `GRAPH_REPORT.md` and `graph.json` only as navigation aids. If they still mention ProductSpace, treat that as stale unless the current source also contains ProductSpace. Verify against source before editing.

Milestone 1: Generator audit.

1. Inspect `tests/generators/`.
2. Confirm the generator protocol can express:

   * object under test;
   * context;
   * dtype;
   * field;
   * check level;
   * batching support;
   * reference data;
   * expected capabilities.
3. Add narrow helper extensions only if Phase G cannot be expressed with existing Phase F infrastructure.

Milestone 2: DenseCoordinateSpace generator.
Implement generated cases for:

* shapes: scalar-like `()`, vector `(n,)`, matrix `(m, n)`, small tensor shapes;
* fields: real and complex;
* dtypes: supported real/complex backend dtypes;
* batching: unbatched and leading-axis batched values;
* check levels: `none`, `cheap`, `standard`, `strict`.

Tests:

* `zero`;
* `add`;
* `scale`;
* `flatten`;
* `unflatten`;
* `flatten_batch`;
* `unflatten_batch`;
* membership checks;
* conversion to another compatible context;
* field behavior.

Milestone 3: DenseVectorSpace generator.
Specialize DenseCoordinateSpace tests to one-dimensional vector spaces (F^n).

Tests:

* vector-specific shape behavior;
* inherited vector-space operations;
* conversion;
* batched vectors;
* field/dtype consistency;
* invalid shape rejection.

Milestone 4: Inner-product space generator.
Generate both:

* Euclidean inner-product spaces;
* non-Euclidean inner-product spaces with SPD metric data.

Tests:

* inner product is scalar-valued;
* norm agrees with inner product:
  `norm(x)^2 == inner(x, x)` within tolerance;
* conjugate symmetry for complex fields;
* bilinearity/sesquilinearity as appropriate;
* non-Euclidean Riesz maps remain coherent;
* conversion preserves inner-product behavior.

Milestone 5: TreeSpace generator.
Generate TreeSpace cases with leaves from existing generated spaces.

Structures:

* tuple tree;
* nested tuple/list tree if supported;
* dict tree if supported by current TreeSpace contract;
* mixed leaf capabilities;
* homogeneous inner-product leaves;
* heterogeneous leaves where some capabilities must not be advertised.

Tests:

* tree structure validation;
* leaf-count validation;
* path-aware error messages;
* `zero`;
* `add`;
* `scale`;
* `check`;
* `convert`;
* `inner` when all leaves support inner products;
* no false inner-product capability when some leaves lack it;
* batched tree elements where supported.

Milestone 6: Jordan-capable space generator.
Generate spaces supporting Jordan operations and spectral structure.

Cover:

* real Euclidean elementwise Jordan spaces;
* Hermitian spaces;
* stacked/tree structures whose leaves genuinely support required capabilities.

Tests:

* Jordan product closure;
* star/involution behavior;
* spectrum shape and dtype;
* spectral decomposition/reconstruction where supported;
* strict-level spectral checks;
* no false Jordan capability for mixed unsupported TreeSpace leaves.

Milestone 7: Vector-space laws.
For each generated vector space (X) over (F), test with generated (x,y,z \in X) and scalars (a,b \in F):

* additive identity: `x + 0 == x`;
* commutativity: `x + y == y + x`;
* associativity: `(x + y) + z == x + (y + z)`;
* scalar identity: `1 * x == x`;
* scalar distributivity: `a * (x + y) == a*x + a*y`;
* field distributivity: `(a + b) * x == a*x + b*x`;
* scalar associativity: `(a*b) * x == a * (b*x)`.

Use backend-aware tolerances.

Milestone 8: Membership by check level.
For each generated space, create valid and invalid elements.

Invalid cases should include:

* wrong shape;
* wrong backend/context;
* unsafe real/complex field mismatch;
* wrong tree structure;
* wrong leaf count;
* invalid Hermitian/Jordan/spectral element when applicable;
* invalid batched trailing shape.

Tests must show:

* `cheap` catches structural errors;
* `standard` catches ordinary membership errors;
* `strict` catches expensive/spectral/metric errors;
* `none` skips only optional runtime validation and does not corrupt internal construction.

Milestone 9: Conversion tests.
For generated spaces and elements, test conversion across compatible contexts.

Required checks:

* mathematical structure is preserved;
* `Space.field` is preserved unless a deliberate documented conversion says otherwise;
* dtype behavior follows ADR-015 Stage 1 decision;
* capabilities are recomputed, not blindly copied;
* TreeSpace leaf structure and leaf paths are preserved;
* converted elements still pass membership.

Milestone 10: Field behavior tests.
For real and complex spaces:

* real data belongs to real spaces;
* complex data belongs to complex spaces;
* complex data is rejected by real spaces unless explicitly allowed by a named intentional operation;
* real data can enter complex spaces if current conversion policy permits it;
* exact dtype checks are not confused with scalar-field checks.

Milestone 11: Batching tests.
Use ADR-006 as the contract.

Tests:

* leading-axis batch shape is accepted;
* trailing dimensions match `space.shape`;
* batched zero/add/scale agree with pointwise operations;
* batched inner products agree with looping over examples;
* TreeSpace batched leaves preserve tree structure;
* unsupported batching is explicitly skipped or documented, not silently wrong.

Files to create or update:

* `tests/generators/spaces.py` or equivalent.
* `tests/spaces/test_generated_dense_coordinate.py`.
* `tests/spaces/test_generated_dense_vector.py`.
* `tests/spaces/test_generated_inner_product.py`.
* `tests/spaces/test_generated_tree_space.py`.
* `tests/spaces/test_generated_jordan_space.py`.
* `tests/spaces/test_generated_space_laws.py`.
* Generator docs if the Phase G additions extend the Phase F API.

Acceptance criteria:

* DenseCoordinateSpace generator exists.
* DenseVectorSpace generator exists.
* Inner-product space generator exists.
* TreeSpace generator exists.
* ProductSpace is not reintroduced.
* Jordan-capable space generator exists.
* Generated tests cover zero/add/scale laws.
* Generated tests cover membership by check level.
* Generated tests cover conversion.
* Generated tests cover real/complex field behavior.
* Generated tests cover batched elements according to ADR-006.
* All tests are deterministic.
* `.venv/bin/python -m pytest tests/spaces tests/generators -x -q` passes.
* `.venv/bin/python -m ruff check .` passes.
* Any deferred unsupported backend or batching case is recorded narrowly in `docs/dev/current.md`.
