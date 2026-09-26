# SpaceCore 0.3.2 — Follow-up release blocker fix task

The current `0.3.2` archive is not ready to tag. This task fixes the remaining release blockers identified after the latest audit.

Do not expand the release scope. Fix only the blockers below, add the minimal tests needed to prevent regression, and re-run the release-gate checks.

---

## Summary of remaining blockers

| ID | Area                                               | Required outcome                                                                                   |
| -- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| F1 | `MatrixFreeLinOp.from_coordinate_adjoint` contract | Constructor wraps coordinate adjoints into metric adjoints using Riesz maps.                       |
| F2 | `docs/dev/vision.md`                               | Vision document exists, is tentative, avoids named-library comparison, and is reachable from docs. |
| F3 | `CHANGELOG.md`                                     | `[0.3.1]` heading carries release date.                                                            |
| F4 | Branch policy                                      | Contributor-doc branch policy is recorded in accepted docs.                                        |
| F5 | ADR-009 docstring references                       | `MatrixFreeLinOp` and `from_coordinate_adjoint` docstrings point to ADR-009.                       |
| F6 | GitHub label mismatch                              | Bug report template labels match label setup script.                                               |

---

## Phase F1 — Fix `from_coordinate_adjoint`

### Problem

`MatrixFreeLinOp.from_coordinate_adjoint` currently stores the supplied reverse callable directly. That is wrong for this constructor.

There are two distinct contracts:

1. Direct `MatrixFreeLinOp(...)` construction:

   * user supplies `rapply`;
   * `rapply` is already the metric adjoint (A^\sharp : Y \to X);
   * SpaceCore must not wrap it.

2. `MatrixFreeLinOp.from_coordinate_adjoint(...)` construction:

   * user supplies a Euclidean coordinate adjoint (A^\dagger);
   * SpaceCore must wrap it into the metric adjoint

[
A^\sharp = R_X^{-1} A^\dagger R_Y
]

for (A : X \to Y).

### Required implementation

In the implementation of `from_coordinate_adjoint`, rename internal parameters or local variables to make the contract explicit:

```python
coordinate_rapply
coordinate_rvapply
```

The scalar reverse callable stored on the returned operator must be equivalent to:

```python
rapply_fn = lambda y: dom.riesz_inverse(
    coordinate_rapply(cod.riesz(y))
)
```

The batched reverse callable, when `coordinate_rvapply` is supplied, must be equivalent to:

```python
rvapply_fn = lambda ys: dom.riesz_inverse(
    coordinate_rvapply(cod.riesz(ys))
)
```

Only use this direct batched formula if `cod.riesz` and `dom.riesz_inverse` support batched inputs under the current space contract. Otherwise, implement the batched path by vectorizing the scalar wrapped `rapply_fn` through the same public batching mechanism used elsewhere.

If `coordinate_rvapply` is omitted, store `rvapply_fn=None` and rely on the existing base fallback over scalar `rapply`.

Before constructing the wrapped callables, call:

```python
_requires_euclidean_or_riesz(
    dom,
    cod,
    "MatrixFreeLinOp.from_coordinate_adjoint",
)
```

Invalid spaces must raise `ValueError` at construction time.

### Required tests

Add or update tests in `tests/linops/test_adjoint_identity.py` or the closest existing LinOp test module.

| ID  | Test                                                       | Required assertion                                                                                                                             |
| --- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| F1a | Direct constructor preserves metric callable               | On a weighted non-Euclidean fixture, direct `MatrixFreeLinOp(..., rapply=metric_rapply)` returns exactly the supplied metric-adjoint behavior. |
| F1b | Coordinate constructor wraps                               | On the same fixture, `from_coordinate_adjoint(..., coordinate_rapply=coordinate_rapply)` satisfies the adjoint identity.                       |
| F1c | Coordinate constructor differs from raw coordinate adjoint | In the non-Euclidean case, `op.rapply(y)` is not equal to `coordinate_rapply(y)` for a nontrivial test vector.                                 |
| F1d | `.H` behavior                                              | `.H.apply(y)` agrees with wrapped `rapply(y)`; `.H.rapply(x)` agrees with forward `apply(x)`; `.H.H` round-trips behavior.                     |
| F1e | Batched coordinate adjoint                                 | If `coordinate_rvapply` is supplied, `rvapply(ys)` agrees with rowwise scalar wrapped `rapply`.                                                |
| F1f | Batched fallback                                           | If `coordinate_rvapply` is omitted, stored `rvapply_fn is None`, and public `rvapply(ys)` agrees with rowwise scalar wrapped `rapply`.         |
| F1g | Riesz availability guard                                   | Non-Euclidean spaces without usable Riesz maps fail at construction time with `ValueError`.                                                    |

Also update any existing negative test that asserts the adjoint identity fails for `from_coordinate_adjoint`; after this task, that test is wrong and must be replaced by a positive adjoint-identity test.

### Bench check

Run or add a minimal test for the weighted matrix-free benchmark case:

```python
bc._weighted_dense_cases(..., matrix_free=True)
```

The case must pass `assert_equal()` for both `rapply` and `rvapply`.

### Acceptance criteria

* Direct `MatrixFreeLinOp` construction remains untouched.
* `from_coordinate_adjoint` wraps coordinate adjoints into metric adjoints.
* Non-Euclidean adjoint identity holds for `from_coordinate_adjoint`.
* Batched and scalar paths agree.
* Matrix-backed adjoint tests are not changed.

---

## Phase F2 — Add `docs/dev/vision.md`

Create `docs/dev/vision.md`.

The document must have exactly these sections:

```markdown
# SpaceCore vision

## Purpose

## Non-goals

## Ecosystem position

## Release sequencing

## 1.0.0 definition
```

### Content requirements

**Purpose**

State that SpaceCore gives typed mathematical structure to spaces, elements, scalar fields, contexts, inner products, Riesz maps, linear operators, functionals, and geometry-aware algorithms across numerical backends.

State that SpaceCore provides precise places to express problem-specific mathematics; it does not hide that mathematics.

**Non-goals**

State that SpaceCore does not provide domain-level abstractions such as PDE models, statistical estimators, optimal-control objects, or semidefinite-programming frontends.

State that SpaceCore is not a general scientific-computing toolkit.

**Ecosystem position**

Keep this section tentative. Do not include named comparisons with SciPy, PyLops, CoLA, Pyxu, ODL, Geomstats, Julia libraries, or any other specific library.

Do not claim a final niche. Use language such as:

```text
tentative role
provisional
likely useful boundary
to be refined through examples and interoperability work
```

The section may say that SpaceCore probably sits between array libraries and domain-specific packages, but it must not contain a detailed current niche analysis.

**Release sequencing**

Explain why contributor infrastructure in `0.3.2` precedes generated test infrastructure in `0.4.0`.

**1.0.0 definition**

Do not mention SDPLab.

Use generic wording such as:

```text
realistic internal and external examples
downstream packages
external use
```

### Docs navigation

Add `vision.md` to `docs/source/dev/index.rst` using the same style as the other dev documents. The file must be reachable from built developer docs.

### Acceptance criteria

* `docs/dev/vision.md` exists.
* No occurrence of `SDPLab` or `sdplab` appears in `vision.md`.
* `vision.md` does not contain named comparisons against external libraries.
* The ecosystem-position language is tentative.
* Sphinx docs build with warnings as errors.

---

## Phase F3 — Fix `[0.3.1]` CHANGELOG date

Edit only the `0.3.1` heading.

Required heading:

```markdown
## [0.3.1] — 2026-06-10
```

Do not modify unrelated changelog content.

---

## Phase F4 — Record branch policy

Edit `docs/dev/contributing/process.md`.

Add a short subsection under branch naming, or immediately after it:

```markdown
### Branch and merge policy

Contributor infrastructure and documentation may be drafted on a feature branch.
They count toward a release only after they are merged into the default branch.
Contributor docs stranded on side branches do not satisfy the release gate.
```

Equivalent wording is acceptable, but all three points must be present.

---

## Phase F5 — Add ADR-009 references to docstrings

Update the `MatrixFreeLinOp` class docstring and the `from_coordinate_adjoint` docstring.

Required content:

* class docstring says direct matrix-free `rapply` is a user assertion of the metric adjoint;
* class docstring points to `docs/dev/adr/009_metric_adjoint.md`;
* `from_coordinate_adjoint` docstring says it wraps a coordinate adjoint with Riesz maps;
* `from_coordinate_adjoint` docstring points to `docs/dev/adr/009_metric_adjoint.md`;
* no docstring says that `from_coordinate_adjoint` stores the supplied coordinate adjoint unchanged.

---

## Phase F6 — Fix label mismatch

Current bug report template and label setup script must agree.

Choose one of these two options:

### Preferred option

Add a `bug` label to `scripts/setup_labels.sh`.

Keep `bug_report.md` using `bug`.

### Alternative option

Change `bug_report.md` to use `fix`.

Only use the alternative if the repository intentionally avoids a separate `bug` label.

Acceptance criterion:

* Running `scripts/setup_labels.sh` creates every label referenced by GitHub issue templates.

---

## Final release-gate checks

Run:

```bash
pytest --co -q
pytest tests/ -q
pytest tests/linops/test_adjoint_identity.py -v
ruff check .
python scripts/docstring_audit.py --check
sphinx-build -W docs/source docs/_build/html
```

Also run the benchmark correctness assertion path that covers:

```python
bc._weighted_dense_cases(..., matrix_free=True)
```

Final acceptance:

* zero pytest collection errors;
* all non-optional tests pass;
* matrix-free direct and coordinate-adjoint constructors are tested side by side;
* benchmark weighted matrix-free `rapply` and `rvapply` assertions pass;
* `vision.md` exists and satisfies the content restrictions;
* `[0.3.1]` changelog has date;
* branch policy is recorded;
* ADR-009 is referenced from relevant docstrings;
* labels referenced by issue templates are created by the label script.
