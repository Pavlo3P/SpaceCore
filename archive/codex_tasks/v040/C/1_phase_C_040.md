Implement SpaceCore 0.4.0 Phase C: dtype defaults and scalar-field contract.

Goal:
Separate mathematical scalar field from representation dtype.

The mathematical space (X) should expose a field:

* `Space.field == "real"` for real vector spaces over (\mathbb{R})
* `Space.field == "complex"` for complex vector spaces over (\mathbb{C})

`Context.dtype` remains the default array representation dtype, not the mathematical scalar-field contract.

Inputs:

* `task_list.md`
* `GRAPH_REPORT.md`
* `graph.json`
* ADR-015, if present in the repo

Use `GRAPH_REPORT.md` first to navigate the repo and save context. Verify all inferred graph edges against source code before editing. Use `graph.json` only when exact file/function locations are needed.

Relevant graph-navigation hints:

* Start around `BackendOps`, `Context`, `DenseCoordinateSpace`, `DenseVectorSpace`, `ElementwiseJordanSpace`, `ProductSpace`, checks, and linalg.
* Inspect dtype-related communities: `DType`, `sanitize_dtype`, `is_complex_dtype`, `real_dtype`, `asarray`, `astype`, `DTypeCheck`.
* Inspect `space/checks/*`, `backend/_context.py`, `backend/_ops.py`, concrete spaces, and solver workspace allocation.

Scope:
Implement Phase C only. Do not implement TreeSpace, block LinOps, or full backend-conformance generators unless a minimal local test helper is needed.

Required implementation:

1. Add `Space.field`.

   * Public spaces expose `field: Literal["real", "complex"]`.
   * Default field is derived from the context dtype:

     * real dtype -> `"real"`
     * complex dtype -> `"complex"`
   * Keep behavior explicit and documented.

2. Decide explicit field override.

   * Read ADR-015.
   * If ADR-015 permits constructor-level field override, implement it.
   * If ADR-015 does not permit it or is ambiguous, do not invent a broad API. Record the decision in docs/current notes.

3. Rewrite scalar-field guards.

   * Replace precision-bearing dtype logic where it is being used to mean scalar field.
   * Capability guards should ask whether the space is over (\mathbb{R}) or (\mathbb{C}), not whether the representation dtype is exactly `float64`, `complex128`, etc.
   * Keep exact dtype checks only where representation dtype genuinely matters.

4. Forbid silent complex-to-real narrowing.

   * `ctx.asarray(...)`, space construction helpers, and conversion paths must reject unsafe conversion from complex-valued data into real dtype/real field unless the imaginary part is explicitly discarded by a named, intentional user action.
   * Error messages must say what was rejected and how to fix it.

5. Backend-specific narrowing behavior.

   * NumPy: add tests proving complex-to-real narrowing is refused.
   * JAX: add tests or document backend limitation.
   * Torch: add tests or document backend limitation.
   * CuPy: add tests when installed, otherwise use optional-backend skips with documented reason.

6. Update membership checks.

   * Membership should distinguish:

     * field compatibility: real vs complex;
     * representation dtype compatibility: exact dtype only if the chosen check policy requires it.
   * Integrate with Phase B check policy if already implemented.
   * Avoid broad behavioral regressions in existing tests.

7. Update docs.

   * Explain that `Context.dtype` is the default representation dtype.
   * Explain that `Space.field` is the mathematical scalar-field contract.
   * Document examples:

     * (X = \mathbb{R}^n) with `float32` or `float64`;
     * (X = \mathbb{C}^n) with `complex64` or `complex128`;
     * why complex input cannot silently enter a real space.

8. Decide ADR-015 Stage 2.

   * Either implement Stage 2 or explicitly defer it to `0.5.0`.
   * If deferred, record the decision in `docs/dev/current.md`.
   * If implemented:

     * preserve input dtype in `ctx.asarray` unless explicit dtype is passed;
     * add field-level `DTypeCheck`;
     * add exact dtype opt-in;
     * make solver workspaces follow operand dtype where possible.

9. Record open mixed-precision questions.

   * Add or update `docs/dev/current.md`.
   * Include unresolved questions about mixed precision, solver workspace dtype, backend promotion, and cross-backend dtype compatibility.

Tests:

* Add unit tests for `Space.field`.
* Add tests for field derivation from dtype.
* Add tests for field override only if implemented.
* Add tests that scalar-field guards use `Space.field`.
* Add narrowing refusal tests for NumPy, JAX, Torch, and optionally CuPy.
* Add membership tests separating field checks from exact dtype checks.
* Add regression tests for concrete spaces, ProductSpace, functionals, LinOps, and linalg paths affected by dtype/field changes.

Acceptance criteria:

* `Space.field` exists and is stable.
* Real/complex field behavior is tested.
* `Context.dtype` is documented as representation default.
* Silent complex-to-real narrowing is refused.
* Backend-specific narrowing behavior is tested or explicitly documented.
* ADR-015 Stage 1 is implemented.
* Stage 2 is either implemented or explicitly deferred.
* Open mixed-precision questions are recorded.
* `pytest tests/ -x -q` passes.
* `ruff check .` passes.
* No unrelated Phase D+ changes are introduced.
