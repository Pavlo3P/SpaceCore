# SpaceCore 0.3.2 — Release gate fix tasks

Five blockers were identified during the post-implementation audit of the
0.3.2 archive. This document specifies the fixes required before tagging.

---

## Summary of blockers

| ID | Area                                                                           | Severity           |
| -- | ------------------------------------------------------------------------------ | ------------------ |
| R1 | `from_coordinate_adjoint` incoherence: code, test, ADR-009, and bench disagree | **Release-gating** |
| R2 | `docs/dev/vision.md` absent — E1–E6 entirely unlanded                          | **Release-gating** |
| R3 | `## [0.3.1]` carries no release date (Z4 half-done)                            | Gate               |
| R4 | E11 branch policy not recorded in any accepted document                        | Gate               |
| R5 | `MatrixFreeLinOp` docstrings do not reference ADR-009 (C15 weak)               | Gate               |

---

## Phase R1 — Resolve `from_coordinate_adjoint` incoherence

### Background

Three artifacts are mutually incoherent:

* **`_algebra.py`** stores `coordinate_rapply` directly, unchanged. No Riesz wrapping is applied.
* **`test_adjoint_identity.py`** pins this by asserting that the adjoint identity fails for an operator built with `from_coordinate_adjoint`.
* **ADR-009** states that `from_coordinate_adjoint` exists for the case where the user has a Euclidean coordinate adjoint and wants SpaceCore to wrap it.
* **`bench/cases.py`** builds the weighted matrix-free benchmark case via `from_coordinate_adjoint`, then asserts `op.rapply(y)` equals (R_X^{-1} A^\dagger R_Y y). Both `rapply` and `rvapply` correctness assertions currently fail.

The coherent resolution is **option A**: make `from_coordinate_adjoint` actually perform the metric wrapping. ADR-009 and the benchmark are the intended contract; the code and the current negative test are wrong.

Option B — deleting the constructor and changing ADR/bench — would also be coherent, but it would discard a useful and self-documenting API.

### Design

For a linear map (A : X \to Y), if the user supplies a Euclidean coordinate adjoint (A^\dagger), then SpaceCore must construct the metric adjoint

[
A^\sharp = R_X^{-1} A^\dagger R_Y.
]

Therefore

```python
from_coordinate_adjoint(
    apply,
    coordinate_rapply,
    dom,
    cod,
    ctx,
    vapply=None,
    coordinate_rvapply=None,
)
```

must return a `MatrixFreeLinOp` whose stored scalar reverse callable is

```python
rapply_fn = lambda y: dom.riesz_inverse(
    coordinate_rapply(cod.riesz(y))
)
```

For the batched analog, when `coordinate_rvapply` is supplied, the stored batched reverse callable must represent

```python
rvapply_fn = lambda ys: dom.riesz_inverse(
    coordinate_rvapply(cod.riesz(ys))
)
```

provided that `cod.riesz` and `dom.riesz_inverse` support batched inputs. If they do not, use the public batched Riesz path or vectorize the scalar wrapper consistently. Do not assume scalar Riesz methods are automatically batch-aware unless the current space contract guarantees it.

If `coordinate_rvapply` is `None`, the stored `rvapply_fn` remains `None`, and the base `rvapply` fallback over scalar `rapply_fn` is used.

**Precondition:** both `dom` and `cod` must have usable Riesz maps or be Euclidean. Raise `ValueError` at construction time, not on first application, if either space fails this requirement. Use the existing helper:

```python
_requires_euclidean_or_riesz(
    dom,
    cod,
    "MatrixFreeLinOp.from_coordinate_adjoint",
)
```

Euclidean spaces have identity Riesz maps, so the formula degenerates to the Euclidean coordinate adjoint.

| ID  | Task                                                   | Done condition                                                                                                                                                                                                                                                                                                                                                              |
| --- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1a | Implement metric wrapping in `from_coordinate_adjoint` | Stored `rapply_fn` computes `dom.riesz_inverse(coordinate_rapply(cod.riesz(y)))`; no path stores `coordinate_rapply` directly.                                                                                                                                                                                                                                              |
| R1b | Implement batched analog                               | When `coordinate_rvapply` is supplied, stored `rvapply_fn` computes the batched metric adjoint using batched Riesz operations or a vectorized scalar wrapper. When `coordinate_rvapply` is `None`, stored `rvapply_fn` remains `None`.                                                                                                                                      |
| R1c | Add Riesz-availability guard                           | `_requires_euclidean_or_riesz(dom, cod, "MatrixFreeLinOp.from_coordinate_adjoint")` is called before constructing wrapped callables; invalid spaces raise `ValueError`.                                                                                                                                                                                                     |
| R1d | Fix the regression test                                | Rename `test_matrix_free_from_coordinate_adjoint_stores_reverse_callable_directly` to `test_matrix_free_from_coordinate_adjoint_wraps_coordinate_adjoint_to_metric_adjoint`. Test the weighted non-Euclidean fixture; verify `apply` equals raw forward, adjoint identity passes, and `rapply(y)` differs from raw `coordinate_rapply(y)` when the metric is non-Euclidean. |
| R1e | Add side-by-side constructor test                      | On the same non-Euclidean fixture, verify direct `MatrixFreeLinOp` construction preserves a user-supplied metric `rapply` unchanged, while `from_coordinate_adjoint` wraps `coordinate_rapply` into the metric adjoint.                                                                                                                                                     |
| R1f | Update conversion tests                                | `test_matrix_free_conversion_preserves_direct_reverse_callable_without_riesz_maps` and `test_matrix_free_compatibility_constructor_preserves_batched_reverse_on_convert` use the metric-adjoint contract; adjoint identity holds before and after `convert()`.                                                                                                              |
| R1g | Update fallback test                                   | `test_matrix_free_compatibility_constructor_without_rvapply_uses_fallback` verifies `rvapply_fn is None` when `coordinate_rvapply` is omitted and that the fallback matches rowwise scalar `rapply`.                                                                                                                                                                        |
| R1h | Confirm bench cases green                              | `bc._weighted_dense_cases(..., matrix_free=True)` `assert_equal()` passes for both `rapply` and `rvapply`. Add a minimal inline test if no CI path exercises these assertions.                                                                                                                                                                                              |
| R1i | Update `from_coordinate_adjoint` docstring             | State that the constructor wraps a coordinate adjoint with Riesz maps to produce the metric adjoint; cite ADR-009; document the Riesz-availability precondition and batched `coordinate_rvapply` behavior.                                                                                                                                                                  |

### Acceptance criteria

* Direct `MatrixFreeLinOp` construction still treats user-supplied `rapply` as the metric adjoint and leaves it unchanged.
* `from_coordinate_adjoint` treats user-supplied `coordinate_rapply` as a Euclidean coordinate adjoint and wraps it into the metric adjoint.
* The side-by-side non-Euclidean fixture proves the distinction.
* `test_matrix_free_non_euclidean_callables_are_untouched` still passes.
* `test_matrix_free_from_coordinate_adjoint_wraps_coordinate_adjoint_to_metric_adjoint` passes.
* Bench `rapply` and `rvapply` correctness assertions pass.
* No matrix-backed adjoint tests change.

### Risk

`from_coordinate_adjoint` with `dom` or `cod` lacking Riesz maps will now raise at construction time rather than silently returning a wrong operator. Any call site passing non-Riesz-capable custom spaces must either supply Riesz maps or use the direct constructor with a precomputed metric adjoint callable.

---

## Phase R2 — Write `docs/dev/vision.md`

All of E1–E6 are unlanded. This is an explicit release-gate requirement.

The document must live at `docs/dev/vision.md`. It must be reachable from `docs/source/dev/index.rst`.

| ID  | Task                                              | Done condition                                                                                                                                                                                                                                                                                                                                 |
| --- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R2a | Create `docs/dev/vision.md` with section skeleton | File exists; five sections present: **Purpose**, **Non-goals**, **Ecosystem position**, **Release sequencing**, **1.0.0 definition**.                                                                                                                                                                                                          |
| R2b | Write Purpose section                             | States that SpaceCore gives typed mathematical structure to linear maps, spaces, inner products, and geometry-aware algorithms across backends, without hiding problem-specific mathematics from the caller.                                                                                                                                   |
| R2c | Write Non-goals section                           | States that SpaceCore does not provide problem-domain abstractions, does not hide mathematical work behind convenience constructors, and does not aim to be a general scientific-computing toolkit.                                                                                                                                            |
| R2d | Write Ecosystem position section                  | Compressed comparison versus SciPy `LinearOperator`, PyLops, CoLA, Pyxu, ODL, and Geomstats. For each: overlapping abstractions, key structural difference, adapter plausibility, and tutorial-comparison suitability. Close with SpaceCore’s intended niche.                                                                                  |
| R2e | Write Release sequencing section                  | Explains why contributor infrastructure in 0.3.2 precedes test infrastructure in 0.4.0: contributors cannot build test generators for a library they cannot navigate.                                                                                                                                                                          |
| R2f | Write 1.0.0 definition section                    | Criteria: core abstractions exercised by SpaceCore and SDPLab; backend contracts stable and documented; extension APIs demonstrated; interoperability adapters reliable; benchmarks and regression tracking established; generic conic abstraction validated; deprecation policy defined; implementation paper submitted or publication-ready. |
| R2g | Add `vision.md` to docs navigation                | `docs/source/dev/index.rst` includes a reference or `literalinclude`; file is reachable from built developer docs.                                                                                                                                                                                                                             |

### Acceptance criteria

* `docs/dev/vision.md` exists, all five sections present and non-placeholder.
* Ecosystem position covers all six named libraries.
* File is reachable from Sphinx dev docs.
* Docs build with warnings as errors.

---

## Phase R3 — Add release date to `[0.3.1]`

Z4 required both the date and the link. The link exists; the date does not.

| ID  | Task                                               | Done condition                                                               |
| --- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| R3a | Add release date to `## [0.3.1]` CHANGELOG heading | Heading reads `## [0.3.1] — 2026-06-10`; no other CHANGELOG content changes. |

---

## Phase R4 — Record branch policy

E11 specified: “Contributor docs may be drafted on a branch but must merge to default branch before release.” This rule is not written anywhere in the accepted documents.

| ID  | Task                                                    | Done condition                                                                                                                                                                                                                                                                                                            |
| --- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R4a | Add branch policy to `docs/dev/contributing/process.md` | Under “Branch naming” or new “Branch and merge policy” subsection: contributor infrastructure and documentation changes may be drafted on a feature branch but must be merged into the default branch before the release tag is applied; stranded contributor docs on side branches do not count toward the release gate. |

---

## Phase R5 — Add ADR-009 reference to `MatrixFreeLinOp` docstrings

C15 required correctness docs to point to ADR-009. The `prerequisites.md` pointer is present but the class and method docstrings do not reference it.

| ID  | Task                                                         | Done condition                                                                                                                            |
| --- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| R5a | Add ADR-009 reference to `MatrixFreeLinOp` class docstring   | The class docstring ends with a `See also` or inline note pointing to `docs/dev/adr/009_metric_adjoint.md` for the full design rationale. |
| R5b | Add ADR-009 reference to `from_coordinate_adjoint` docstring | After R1i, the updated docstring includes a reference to ADR-009.                                                                         |

---

## Release gate re-check

After all phases are complete, re-run:

1. `pytest --co -q` — zero collection errors; JAX/Torch cases skip when optional backends are absent.
2. `pytest tests/ -q` — all non-optional tests pass.
3. `pytest tests/linops/test_adjoint_identity.py -v` — matrix-free non-Euclidean tests and `from_coordinate_adjoint` tests pass.
4. `ruff check .` and `python scripts/docstring_audit.py --check` — clean.
5. Bench `from_coordinate_adjoint` correctness assertions pass.
6. `docs/dev/vision.md` exists with all five sections.
7. CHANGELOG `[0.3.1]` carries the release date.
8. Direct `MatrixFreeLinOp` docstring and `from_coordinate_adjoint` docstring reference ADR-009.
9. `from_coordinate_adjoint` docstring describes wrapping, not passthrough.
10. Sphinx docs build without warnings.
