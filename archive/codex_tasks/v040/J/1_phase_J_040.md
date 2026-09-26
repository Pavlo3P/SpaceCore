Phase J implementation plan: backend conformance.

Goal:
Build a backend-conformance layer that verifies SpaceCore backend implementations against a clear NumPy reference contract.

Scope:
Implement Phase J only. Do not redesign backend APIs unless a defect is found. Prefer tests, documentation, and narrow bug fixes.

Assumptions:

* Phases B–I are already merged or their missing pieces are recorded.
* Use only `.venv` for execution.
* Use `GRAPH_REPORT.md` and `graph.json` as navigation aids, but verify all conclusions against source files.

Primary files to inspect:

* `spacecore/backend/_ops.py`
* `spacecore/backend/_context.py`
* `spacecore/backend/numpy/_ops.py`
* `spacecore/backend/jax/_ops.py`
* `spacecore/backend/torch/_ops.py`
* `spacecore/backend/cupy/_ops.py`
* `_batching.py`
* `_checks.py`
* `spacecore/_contextual/*`
* existing backend tests under `tests/backend/`
* generator helpers under `tests/generators/`, if present

Step 1: Define the backend-conformance matrix.
Create a document, preferably:

`docs/dev/backend_conformance_matrix.md`

It should list:

* backend families: NumPy, JAX, Torch, CuPy;
* supported dtypes;
* real/complex field behavior;
* dense-array operations;
* sparse support;
* optional arguments;
* batching behavior;
* conversion behavior;
* JAX JIT expectations;
* known deviations from NumPy.

Step 2: Add backend-conformance test helpers.
Create or extend:

`tests/backend/conformance/`

Suggested files:

* `test_numpy_reference.py`
* `test_jax_conformance.py`
* `test_torch_conformance.py`
* `test_cupy_conformance.py`
* `test_backend_optional_args.py`
* `test_backend_dtype_promotion.py`
* `test_backend_scalar_field.py`
* `test_backend_batching.py`
* `test_backend_conversion.py`
* `test_jax_jit.py`

Use NumPy as the reference backend. Optional backends should use `pytest.importorskip`.

Step 3: Implement J1.
The conformance matrix must explicitly cover:

* operations;
* dtypes;
* scalar fields;
* batching;
* JIT;
* conversion;
* sparse behavior;
* deviations.

Step 4: Implement J2.
Compare `NumpyOps` against direct NumPy calls. This pins SpaceCore’s own reference behavior.

Step 5: Implement J3–J5.
Compare JAX, Torch, and CuPy against `NumpyOps`.
Use tolerances by dtype:

* stricter for float64/complex128;
* looser for float32/complex64;
* skip unsupported dtype/backend combinations explicitly.

Step 6: Implement J6.
Test optional arguments consistently:

* `axis`;
* `keepdims`;
* `dtype`;
* `out`, if supported or intentionally unsupported;
* indexing/mutation helper arguments;
* sparse helper arguments.

Unsupported optional arguments should be documented, not silently ignored.

Step 7: Implement J7.
Pin dtype-promotion behavior.
Tests should cover:

* real + real;
* real + complex;
* scalar + array;
* int input normalization;
* complex-to-real refusal from Phase C;
* backend-specific promotion differences.

Do not force all backends to emulate NumPy if that would be unnatural. Instead, document deliberate deviations.

Step 8: Implement J8.
Test scalar-field behavior:

* real contexts reject unsafe complex narrowing;
* complex contexts accept real input by promotion;
* `Space.field` and backend conversion rules remain compatible;
* exact dtype checks remain separate from mathematical field checks.

Step 9: Implement J9.
Test batching behavior:

* batched dense operations match loops over unbatched operations;
* batch-axis conventions follow ADR-006;
* backend-native vectorization is used only where already supported;
* unsupported batching is explicitly skipped or documented.

Step 10: Implement J10.
Test JAX JIT behavior:

* mark which paths are JIT-compatible;
* test representative JIT-compatible paths with `jax.jit`;
* document non-JIT-compatible paths caused by Python-side validation, object construction, sparse limitations, or dynamic tree structure.

Step 11: Implement J11.
Test conversion behavior:

* `ctx.asarray`;
* `ctx.assparse`;
* `ctx.convert`;
* backend-to-backend conversion;
* dtype preservation or coercion according to ADR-015;
* field/narrowing rules.

Step 12: Implement J12.
Add backend-deviation docs.
Preferred location:

`docs/dev/backend_deviations.md`

Include:

* NumPy reference contract;
* JAX deviations;
* Torch deviations;
* CuPy deviations;
* sparse limitations;
* dtype promotion differences;
* JIT limitations;
* optional argument limitations.

Acceptance criteria:

* Backend-conformance matrix exists.
* NumPy reference tests pass.
* JAX/Torch/CuPy conformance tests pass or skip with clear reasons.
* Optional argument behavior is pinned.
* Dtype promotion behavior is pinned.
* Scalar-field behavior is tested.
* Batching behavior is tested.
* JAX JIT-compatible paths are tested.
* Conversion behavior is tested.
* Backend deviations are documented.
* Run:

`.venv/bin/python -m pytest tests/backend/ -x -q`

`.venv/bin/python -m pytest tests/ -x -q`

`.venv/bin/python -m ruff check .`

If any backend cannot be tested locally, record the exact backend, missing dependency/device, and intended CI behavior in `docs/dev/current.md`.
