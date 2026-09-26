# Phase F Plan — Test-generator infrastructure

## Goal

Create reusable, deterministic test-generator infrastructure under `tests/generators/`. This phase should make later phases G–J easy to implement without duplicating ad hoc fixtures.

Do not implement broad Space, LinOp, Functional, linalg, or backend conformance coverage in this phase. Add only minimal smoke tests proving the generators work.

## Constraints

* Use only `.venv`.
* Run commands as `.venv/bin/python -m pytest ...` and `.venv/bin/python -m ruff check .`.
* Use `GRAPH_REPORT.md` and `graph.json` only for navigation/token saving. Verify all inferred graph facts against source.
* Since `ProductSpace` has been removed, implement only `TreeSpace`-based structured generators.

## Proposed package layout

```text
tests/generators/
    __init__.py
    _protocol.py
    _seed.py
    _contexts.py
    _arrays.py
    _hermitian.py
    _metrics.py
    _trees.py
    _params.py
    README.md
```

## Step 1 — Define the generator protocol

Create a small public testing protocol, for example:

```python
@dataclass(frozen=True)
class GeneratedCase(Generic[T]):
    obj: T
    reference: Mapping[str, Any]
    capabilities: frozenset[str]
    marks: tuple[Any, ...] = ()
    id: str | None = None
```

The protocol should support:

* generated object;
* reference data;
* expected capabilities;
* pytest id;
* optional pytest marks/skips.

Keep it simple. Do not over-engineer a full property-testing framework.

## Step 2 — Add seed policy

Create deterministic seed utilities:

* one global default seed;
* `seeded_rng(seed: int)`;
* documented convention: every generator accepts `seed` or `rng`;
* no hidden global randomness in tests.

Acceptance:

* repeated calls with the same seed produce identical NumPy reference data.

## Step 3 — Add context generator

Implement context cases for supported backend/dtype combinations.

Minimum:

* NumPy float64;
* NumPy complex128;
* any optional backends only if installed.

The generator should produce `Context` objects and pytest marks for unavailable optional backends.

Do not require JAX/Torch/CuPy to be installed to run the core test suite.

## Step 4 — Add check-level parameterization

Generate/check parameter cases for:

```text
none
cheap
standard
strict
```

The output should be usable in pytest parametrization.

Acceptance:

* generated ids are stable;
* marks/skips are explicit;
* invalid check levels are not silently accepted.

## Step 5 — Add batched/unbatched parameterization

Create a small batch-mode descriptor:

```python
@dataclass(frozen=True)
class BatchCase:
    batched: bool
    batch_shape: tuple[int, ...]
```

Minimum cases:

* unbatched;
* batch size 2 or 3.

Do not redesign batching. Follow ADR-006 behavior.

## Step 6 — Add dense-array data generator

Implement dense array generation for:

* real arrays;
* complex arrays;
* scalar, vector, matrix, and small tensor shapes;
* optional batch prefix.

The generator should return both backend arrays and NumPy reference arrays where useful.

It must respect field/dtype rules from Phase C:

* real field gets real-valued arrays;
* complex field may get complex-valued arrays;
* no silent complex-to-real narrowing.

## Step 7 — Add Hermitian data generator

Implement Hermitian matrix generation for square shapes.

Cases:

* real symmetric;
* complex Hermitian;
* batched Hermitian, if current APIs support it.

Acceptance:

* generated data satisfies `A = A*`;
* tests use numerical tolerance appropriate to dtype.

## Step 8 — Add SPD metric generator

Implement SPD matrix generation for non-Euclidean inner products.

Requirements:

* deterministic;
* small dimensions;
* controlled condition number;
* real SPD by default;
* complex Hermitian positive definite where supported.

Reference data should include:

* matrix;
* inverse or solve helper if cheap;
* condition estimate if available.

## Step 9 — Add TreeSpace data generator

Implement structured tree data generation using `TreeSpace`, not `ProductSpace`.

Minimum tree structures:

* tuple tree;
* nested tuple/list tree;
* dict tree if TreeSpace supports dicts robustly.

Generate:

* leaf spaces;
* matching leaf values;
* reconstructed tree element;
* reference flattened leaves;
* leaf paths.

Acceptance:

* structure mismatch cases can be generated for later negative tests;
* no ProductSpace imports or compatibility assumptions.

## Step 10 — Add pytest parameter helpers

Create helpers such as:

```python
context_params()
check_level_params()
batch_params()
dense_array_cases(...)
tree_space_cases(...)
```

These should return clean `pytest.param(...)` objects with stable readable ids.

## Step 11 — Add smoke tests

Add a small test file, for example:

```text
tests/test_generators_smoke.py
```

Smoke tests should check:

* package imports;
* deterministic seed behavior;
* context generator emits at least NumPy cases;
* dense arrays match requested shape/field/dtype;
* Hermitian generator produces Hermitian matrices;
* SPD generator produces positive eigenvalues;
* TreeSpace generator produces valid TreeSpace elements.

Do not expand this into full Phase G/H/I/J coverage.

## Step 12 — Add contributor documentation

Create `tests/generators/README.md`.

Document:

* purpose of generators;
* how to add a new generator;
* seed policy;
* check-level policy;
* backend skip policy;
* TreeSpace-only structured data policy;
* boundary between Phase F infrastructure and later generated coverage phases.

## Acceptance criteria

* `tests/generators/` package exists.
* Generator protocol exists.
* Context generator exists.
* Check-level parameterization exists.
* Batched/unbatched parameterization exists.
* Seed policy exists and is documented.
* Dense-array generator exists.
* Hermitian data generator exists.
* SPD metric generator exists.
* TreeSpace structured-data generator exists.
* ProductSpace is not used.
* Contributor README exists.
* Smoke tests validate generator infrastructure.
* `.venv/bin/python -m pytest tests/test_generators_smoke.py -q` passes.
* `.venv/bin/python -m pytest tests/ -x -q` passes.
* `.venv/bin/python -m ruff check .` passes.
