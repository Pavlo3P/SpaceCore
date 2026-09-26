# Codex task: Fulfil Phase C — SpaceCore 0.3.1 documentation baseline

## Objective

Complete the SpaceCore `0.3.1` documentation baseline.

The goal is not to add new functionality. The goal is to make the documentation accurately describe the current `0.3.x` API, remove stale concepts from older designs, ensure tutorials execute, and make the public API documentation usable for new users.

This task should be treated as a documentation stabilization pass before release.

Two quality bars are normative for this task and are specified in full below:

* **Docstring standard** — every reviewed or added docstring follows the NumPy (numpydoc) format already used in the codebase, with a tiny executable example wherever feasible.
* **Tutorial quality standard** — every public tutorial is extensive enough to be self-contained, user-friendly, and step-by-step followable by a reader who has never used SpaceCore.

## Scope

Work only on documentation, examples, tutorials, docstrings, documentation indexes, and documentation build configuration when needed.

Do not change library behavior unless a documentation example reveals a small obvious bug that prevents the documented public API from working. In that case, keep the fix minimal and document it in the final report.

## General constraints

* Target the current `0.3.x` API.
* Do not document removed APIs as current APIs.
* Do not invent future APIs.
* If an API is planned but not implemented, mark it clearly as planned or omit it.
* Prefer small, reviewable commits.
* Keep each subtask bounded. If one page or tutorial requires more than roughly one focused hour, stop and create a precise follow-up note instead of expanding the task indefinitely.
* All examples included in public docs must either execute or be explicitly marked as conceptual/pseudocode.
* Mathematical statements must specify the spaces involved. For example, write `A : X -> Y` where `X` and `Y` are SpaceCore spaces, not just "an operator".
* When discussing adjoints, distinguish the coordinate conjugate transpose from the metric adjoint.
* When discussing batching, describe current vectorized application mechanisms such as `vapply` or backend vectorization. Do not refer to removed batch-space abstractions.
* All docstrings must conform to the **Docstring standard** section below. Do not mix formats.
* All tutorials must conform to the **Tutorial quality standard** section below.

---

## Docstring standard (numpydoc, with tiny examples)

The codebase already contains the target style. Treat these two as the canonical exemplars and bring everything else to their level:

* `spacecore/linop/_base.py` — class `LinOp` (class-style exemplar);
* `spacecore/linalg/_cg.py` — function `cg` and `CGResult` (function/result-style exemplar).

### Required format

Use **numpydoc** sections, in this fixed order (omit sections that do not apply, never reorder):

1. One-line summary — imperative or declarative, one sentence, states the mathematical object or action. For raw-LaTeX docstrings use `r"""`.
2. Extended summary — 1–4 sentences. For operators and solvers this is where the spaces appear: `A : X -> Y`, `:math:` roles for formulas.
3. `Parameters`
4. `Returns` / `Yields` (or `Attributes` for classes)
5. `Raises` — only errors a user can realistically trigger.
6. `See Also` — cross-link siblings (`cg` ↔ `lsqr`, `DenseLinOp` ↔ `MatrixFreeLinOp`).
7. `Notes` — mathematical contract: adjoint semantics, geometry restrictions, convergence criteria, backend caveats. State which inner product is used (`A.domain.inner`, etc.).
8. `References` — only for algorithms with a citable source (Lanczos, LSQR, CG variants).
9. `Examples` — see below.

### Tiny-example rules

Every public class, constructor, and linalg function gets an `Examples` section unless it is genuinely impossible to demonstrate in ≤ 10 lines (record such cases in the C20 checklist with a reason).

* **Tiny means tiny**: 3–10 lines, one idea, smallest possible space (`(2,)` or `(3,)` shapes, `2x2` matrices).
* **Doctests must be deterministic and backend-stable**:
  * always build on `sc.Context(sc.NumpyOps(), dtype=np.float64)` — never JAX/Torch/CuPy in doctests, since their array reprs (`Array(...)`, `tensor(...)`) differ and break `--doctest-modules`;
  * never use random data in a doctest; use small integer-valued floats so printed output is stable (`array([3., 8.])`, as in the existing `LinOp` example);
  * if an example needs a non-NumPy backend (JIT, GPU), write it as a `.. code-block:: python`, not a doctest, and say why.
* Examples demonstrate the *contract*, not just the call: a `LinOp` example should show `apply` **and** `rapply`; a weighted-geometry example should show that `inner` differs from the Euclidean dot; a solver example should show reading the result object (`result.converged`, `result.num_iters`).
* Every doctest must pass under `python -m pytest --doctest-modules spacecore`.

### Reference templates

Class (space/operator) skeleton:

```python
r"""
Represent <mathematical object> as a SpaceCore <Space/LinOp>.

<1-3 sentences. For operators: the map :math:`A : X \to Y`, element
representation, what ``rapply`` computes (metric adjoint
:math:`A^\sharp`, equal to the conjugate transpose iff both geometries
are Euclidean), and any geometry restrictions.>

Parameters
----------
...

Attributes
----------
...

Raises
------
...

See Also
--------
...

Notes
-----
<Adjoint/geometry contract. Matrix-backed operators: state how the
matrix is interpreted in domain/codomain coordinates.>

Examples
--------
>>> import numpy as np
>>> import spacecore as sc
>>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
>>> X = sc.DenseCoordinateSpace((2,), ctx)
>>> A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 2.0]]), X, X, ctx)
>>> A.apply(ctx.asarray([3.0, 4.0]))
array([3., 8.])
>>> A.rapply(ctx.asarray([1.0, 1.0]))
array([1., 2.])
"""
```

Solver skeleton (model: `cg`): summary with :math:`A x = b`; preconditions in the extended summary (`A.domain == A.codomain`, Hermitian, positive-definite **with respect to** `A.domain.inner`); `Parameters`; `Returns` naming the result NamedTuple; `Notes` stating the inner product used for residuals and the convergence criterion; tiny doctest on a 2x2 SPD system printing `result.converged` and `result.num_iters`.

---

## Tutorial quality standard

Applies to every tutorial that remains public after the C9 inventory. A tutorial is *extensive, user-friendly, and easy to follow* when it meets all of the following.

### Structure (fixed skeleton)

1. **Header block**: what you will build/learn (one sentence), prerequisites (Python level, math level, which earlier tutorial), estimated time, backends used.
2. **Setup**: complete imports and context construction, runnable as-is; state required extras (`pip install "spacecore[jax]"`) when relevant.
3. **Stepwise body**: short numbered steps. Each step = a few sentences of *why*, then a code block, then the expected output. Code blocks are cumulative: pasting them top-to-bottom into a fresh interpreter reproduces the tutorial exactly. No hidden state, no "assume we have...".
4. **Checkpoints**: after each major step, a one-line verification the reader can run (`print(result.converged)  # True`, a norm check, a shape check) so they know immediately whether they are on track.
5. **"What can go wrong" boxes**: at known stumbling points, an admonition showing the actual error message the user would see and the fix (wrong dtype, mismatched context, non-Hermitian matrix passed to `cg`, missing optional backend).
6. **Recap**: 3–5 bullet summary of the concepts exercised.
7. **Next steps**: links to the logical next tutorial and to the relevant design notes / API index pages.

### Rules

* **One coherent idea per tutorial.** If a tutorial currently teaches two, split it (and record the split in C9).
* **Progressive disclosure.** Start with the simplest correct usage (NumPy context, Euclidean geometry); introduce geometry, structure, or alternative backends only after the basic flow works. Never open with the most general form.
* **A single running example per tutorial**, carried through all steps, rather than disconnected fragments.
* **Plain language first, notation second.** Every formula is preceded by a sentence saying what it means; every SpaceCore term (context, space, metric adjoint, capability) is either defined inline on first use or linked to its design note.
* **Show expected outputs** for every code block that prints, so readers can self-verify without running anything.
* **No stale API, no future API.** Tutorials demonstrate `0.3.x` exactly; planned features may appear only in "Next steps" as plans.
* **Executable end-to-end** in a clean environment with documented extras; this is verified in C9 and again in the final validation pass.

### Per-tutorial acceptance checklist (apply in every C10+ subtask)

```text
[ ] Header: goal, prerequisites, estimated time, backends
[ ] Setup block runs as-is in a clean environment
[ ] Steps are numbered; each has why -> code -> expected output
[ ] Code blocks are cumulative and reproduce top-to-bottom
[ ] At least one checkpoint per major step
[ ] At least one "what can go wrong" box at a real stumbling point
[ ] Adjoint language distinguishes metric adjoint from conjugate transpose
[ ] Batching language uses vapply/vectorization, no removed abstractions
[ ] Recap and next-steps sections present
[ ] Executes start to finish; outputs match the shown outputs
```

---

## Main deliverables

1. Updated README opening and feature summary.
2. Updated documentation landing page.
3. Reviewed and corrected conceptual notes:

   * context ownership / normalization;
   * context conversion;
   * dtype handling;
   * batching;
   * capability dispatch.
4. Tutorial inventory with status for every tutorial.
5. Updated tutorials using the current `0.3.x` API, each meeting the Tutorial quality standard.
6. Public docstring audit.
7. Reviewed docstrings for:

   * spaces;
   * linear operators;
   * functionals;
   * linalg;
   * backend/context objects.
8. A short docstring-writing guide codifying the numpydoc standard above.
9. Index pages for:

   * spaces;
   * linear operators;
   * functionals;
   * linalg;
   * backend/context.
10. Final documentation build with no warnings, and passing doctests.

## Detailed tasks

### C1 — Inventory documentation problems

Create a documentation inventory file, for example:

```text
docs/dev/0.3.1-docs-inventory.md
```

The inventory should list every stale, incomplete, broken, or suspicious page.

For each item, record:

* file path;
* problem type:

  * stale API;
  * broken example;
  * missing page;
  * incomplete explanation;
  * incorrect mathematical statement;
  * missing docstring;
  * broken link;
  * obsolete tutorial;
* proposed action:

  * rewrite;
  * delete;
  * update;
  * mark conceptual;
  * split into follow-up;
* status:

  * todo;
  * done;
  * follow-up.

Done condition: every known documentation problem has an entry.

### C2 — Rewrite README opening and feature summary

Update the README opening so that it accurately describes SpaceCore `0.3.x`.

The opening should say, in substance:

* SpaceCore provides typed vector spaces, structured elements, linear operators, functionals, and linalg utilities.
* Operators are maps between spaces, not merely arrays.
* Spaces may carry inner products and structure.
* Adjoint semantics are geometry-aware.
* Backends are handled through explicit contexts.
* The native solver set is intentionally small and should be understood as a correctness baseline and substrate layer, not a replacement for mature solver ecosystems.
* External adapters / fast paths may be used later for solver breadth.

Avoid overstating current capabilities.

Do not claim that all future planned features already exist.

Additionally: every code snippet in the README must execute against `0.3.x` (the known-broken CG and `ops.fft` examples are covered by Phase A; verify they were fixed, otherwise fix them here and note the overlap in the report).

Done condition: README first section gives an accurate, concise, technically honest description of the library, and all README snippets execute.

### C3 — Improve documentation landing page

Update the main documentation landing page (`docs/source/index.rst`) so that a new reader can find:

* Spaces;
* elements;
* linear operators;
* functionals;
* linalg;
* backend/context;
* tutorials;
* migration notes;
* API reference.

The landing page should briefly explain the core mental model:

```text
A SpaceCore operator is a typed map A : X -> Y between spaces X and Y.
The spaces carry the rules needed to validate elements, compute inner products,
and interpret adjoints.
```

Add a short "where do I start?" routing paragraph: first-time users -> spaces tutorial; users with an existing NumPy algorithm -> backend/context tutorial; users with non-Euclidean problems -> geometry design note.

Done condition: the landing page has clear navigation, a routing paragraph, and no stale references.

### C4 — Review context ownership / normalization note

Find the existing note or page explaining context ownership and normalization.

Update it to match current implementation.

It should clarify:

* what owns the backend context;
* how arrays are normalized or converted;
* when user input is accepted as-is;
* when SpaceCore converts input;
* what guarantees a constructed space/operator makes about context;
* what is not guaranteed.

Remove references to removed context-policy APIs.

Done condition: the note matches the current implementation and examples execute.

### C5 — Review context conversion note

Update the context conversion documentation (`docs/source/design/conversion_policy.rst` and the matching tutorial).

It should explain:

* how spaces/operators/elements move across contexts, if currently supported;
* what is copied;
* what is reconstructed;
* what may fail;
* how backend-specific behavior is handled;
* how JAX/Torch/NumPy/CuPy differences are exposed.

Do not describe removed APIs.

Done condition: context conversion docs are current and technically precise.

### C6 — Review dtype note

Update dtype documentation (`docs/source/design/dtype_policy.rst`).

It should explain:

* how dtype is stored or inferred;
* how dtype interacts with spaces;
* how dtype interacts with backend context;
* what happens for real vs complex spaces;
* what assumptions linalg functions make;
* what errors users should expect from invalid dtype usage.

Remove stale dtype-policy documentation.

Done condition: dtype docs match current implementation.

### C7 — Review batching note

Update batching documentation.

The page should describe current batching/vectorization mechanisms.

It should avoid removed abstractions such as old batch-space language.

Recommended framing:

* a space describes one element type;
* batched computation is handled by vectorized application mechanisms, e.g. `vapply`, backend `vmap`, or explicit batched operators where available;
* batching should not change the mathematical domain/codomain of an operator unless the operator itself is explicitly batched.

Include a small example if the current API supports it.

Done condition: batching docs are accurate and do not mention stale batch APIs.

### C8 — Review capability-dispatch note

Update the capability-dispatch documentation.

It should explain current structural capabilities, for example:

* plain vector-space behavior;
* inner-product capability;
* adjoint capability;
* star/involution capability if present;
* Jordan/spectral capability if present;
* how conversions preserve or drop capabilities;
* when operations refuse because a capability is missing.

Include at least one concrete refusal example with the actual error message (e.g. a matrix-backed operator constructed on a non-Euclidean space without usable Riesz maps).

Important: distinguish current implemented capabilities from planned future capabilities.

Done condition: the note explains actual capability selection and failure behavior, including one demonstrated refusal.

## Tutorials

### C9 — Inventory and execute all tutorials

Create a tutorial inventory table. The current tutorial set to inventory (verify against the repository; do not assume this list is complete):

```text
docs/source/tutorials/spaces.rst
docs/source/tutorials/linops.rst
docs/source/tutorials/context.rst
docs/source/tutorials/conversion_policy.rst
docs/source/tutorials/backend_ops.rst
docs/source/tutorials/regularized_ot.rst
docs/source/tutorials/index.rst
```

For each tutorial, record:

* file path;
* purpose (the one coherent idea it teaches — if there are two, plan the split);
* API version it appears to target;
* whether it executes from start to finish;
* failure point if it fails;
* gap against the Tutorial quality standard (missing header, missing checkpoints, non-cumulative code, missing expected outputs, etc.);
* whether it should be:

  * updated;
  * deleted;
  * split;
  * marked obsolete;
  * turned into a conceptual note.

Also record gaps in the *learning path*: with the current set, a reader should be able to go from "never seen SpaceCore" to "solved a small problem with cg on a custom geometry". If a step in that path has no tutorial, file it as a follow-up note (do not write new tutorials in this phase beyond the Phase D worked example).

Run each tutorial in a clean environment if possible.

Done condition: every tutorial is marked passing, failing, or obsolete, with a quality-standard gap list and an ordered learning path.

### C10+ — Update tutorials one by one to the `0.3.x` API

For each tutorial that should remain public, bring it to the Tutorial quality standard. Mechanical pass:

* update imports;
* update class/function names;
* update construction of spaces;
* update construction of operators;
* update element creation;
* update linalg calls;
* update adjoint language;
* update batching language;
* update expected outputs;
* make the tutorial executable from start to finish.

Quality pass (same subtask, same hour budget):

* add the header block (goal, prerequisites, estimated time, backends);
* number the steps and ensure each is why -> code -> expected output;
* make code blocks cumulative;
* add checkpoints and at least one "what can go wrong" box;
* add recap and next-steps sections;
* run the per-tutorial acceptance checklist and paste the filled checklist into the PR description.

Each tutorial should be handled as a separate subtask.

If a tutorial has multiple independent sections and cannot be repaired quickly, split the work by section.

Done condition for each tutorial: it runs from start to finish, teaches one coherent idea, and the acceptance checklist is fully checked.

## Public API documentation

### C20 — Audit missing public docstrings

Produce a module-by-module checklist of public objects missing adequate docstrings. "Adequate" means conforming to the Docstring standard above, including the tiny example.

Include at least:

* public classes;
* public functions;
* public methods likely to be used directly;
* public module-level helpers;
* public exceptions if any;
* result types (`CGResult`, `LSQRResult`, `LanczosResult`, `PowerIterationResult`, `ExpmMultiplyResult`).

For each object, mark:

* missing;
* incomplete;
* stale;
* mathematically unclear;
* missing example;
* okay.

Where a tiny example is infeasible, record the reason.

Done condition: checklist exists, covers the public API, and tracks example coverage separately from prose coverage.

### C21 — Review space docstrings

Review docstrings for all public space classes and constructors against the Docstring standard.

Each space docstring should explain:

* mathematical object represented;
* element space;
* array representation;
* shape conventions;
* dtype assumptions;
* context assumptions;
* inner product, if any;
* validation behavior;
* a tiny example constructing the space, creating one element, and (for inner-product spaces) computing one inner product.

Examples of preferred mathematical clarity:

```text
DenseVectorSpace represents K^n with elements stored as arrays of shape (n,).
```

```text
ProductSpace represents X_1 × ... × X_k, not a tensor product.
```

For Jordan-capable spaces, the docstring must state what `spectrum` / `spectral_decompose` return and demonstrate `spectral_apply` on a trivial element if it fits in a tiny example.

Done condition: space docstrings are complete, mathematically correct, numpydoc-conformant, and carry tiny examples.

### C22 — Review linear-operator docstrings

Review docstrings for public linear-operator classes and constructors against the Docstring standard.

Each docstring should explain:

* domain `X`;
* codomain `Y`;
* action `A : X -> Y`;
* expected element representation;
* adjoint semantics;
* whether `rapply` means metric adjoint;
* restrictions on geometry;
* backend/context behavior;
* a tiny example showing **both** `apply` and `rapply` with printed outputs.

Important: if an operator is matrix-backed, document how the matrix is interpreted with respect to domain and codomain coordinates, and state the refusal behavior on non-Euclidean spaces without usable Riesz maps.

The algebra types (`SumLinOp`, `ComposedLinOp`, `ScaledLinOp`, and the `make_*` helpers) must each show the one-line algebraic construction they correspond to (`A + B`, `A @ B`, `2 * A`).

Done condition: LinOp docstrings are complete, accurate, explicit about adjoints, and carry tiny apply/rapply examples.

### C23 — Review functional docstrings

Review public functional docstrings against the Docstring standard.

Each docstring should explain:

* domain space `X`;
* scalar field, usually `R`;
* value semantics;
* gradient semantics;
* whether the gradient is a Riesz-represented gradient in `X` (i.e. satisfies `<grad f(x), v>_X = Df(x)[v]`);
* pullback behavior if applicable;
* differentiability assumptions;
* a tiny example computing `value` and `grad` of a quadratic on a 2-dimensional space.

Done condition: functional docstrings are complete, mathematically precise, and carry tiny examples.

### C24 — Review linalg docstrings

Review public linalg function docstrings against the Docstring standard, using `cg` as the model and the Krylov.jl documentation discipline as the bar: every method states the exact problem class it solves.

Each solver/function should document:

* accepted operator type;
* required domain/codomain properties;
* whether the operator must be square;
* whether it must be self-adjoint;
* whether it must be positive definite / semidefinite;
* inner product used in residuals and adjoints (`A.domain.inner` / `A.codomain.norm` — name it explicitly);
* convergence criteria;
* return type (the result NamedTuple, field by field);
* failure modes;
* backend/JIT caveats;
* a tiny deterministic doctest on a 2x2 or 3x3 system showing how to read the result object.

For example, `cg` should clearly say that it solves an equation on a space `X`, and should state the assumptions on `A : X -> X`.

Done condition: linalg docstrings state preconditions and guarantees accurately, with passing doctests.

### C25 — Review backend/context docstrings

Review docstrings for backend and context objects (`Context`, `BackendOps`, `NumpyOps`, `JaxOps`, `TorchOps`, `CuPyOps`) against the Docstring standard.

Each docstring should explain:

* what the context owns;
* what backend operations are available;
* array API assumptions;
* conversion behavior;
* dtype/device caveats;
* backend-specific limitations;
* examples — `Context`/`NumpyOps` get doctests; `JaxOps`/`TorchOps`/`CuPyOps` get `.. code-block::` examples (optional dependencies must not break `--doctest-modules`).

Done condition: backend/context docs are complete and accurate, and the doctest suite passes without optional backends installed.

### C26 — Add or finalize docstring-writing guide

Create or update a short contributor-facing guide, for example:

```text
docs/dev/docstring_style.md
```

The guide codifies the Docstring standard section of this task as the single project style. It must:

* mandate numpydoc with the fixed section order above;
* point to `spacecore/linop/_base.py::LinOp` and `spacecore/linalg/_cg.py::cg` as the canonical exemplars;
* always state mathematical spaces;
* always state array representation;
* always state backend/context assumptions;
* always state adjoint semantics for operators (metric adjoint vs conjugate transpose);
* always state gradient semantics for functionals;
* include the tiny-example rules verbatim (size limits, NumPy-only doctests, determinism, no randomness, `code-block` for non-NumPy backends);
* include both reference templates (class skeleton, solver skeleton);
* avoid documenting future APIs as current APIs.

Done condition: one clear docstring style exists, matches what the reviewed docstrings actually do, and is the file `prerequisites.md` will link to in 0.3.2.

## API index pages

### C27 — Add `space` index page

Create or update an index page summarizing public space classes.

It should group objects by role, for example:

* coordinate/vector spaces;
* product spaces;
* structured spaces (e.g. `HermitianSpace`, `StackedSpace`);
* inner-product spaces and geometries (`InnerProduct`, `WeightedInnerProduct`);
* spectral/Jordan-capable spaces, if implemented;
* helper constructors.

Each entry: one sentence stating the mathematical object and the element representation, then a link.

Done condition: all public space objects are summarized and linked.

### C28 — Add `linop` index page

Create or update an index page summarizing public linear-operator classes.

Group by role:

* matrix-backed operators (`DenseLinOp`, `SparseLinOp`, `DiagonalLinOp`);
* matrix-free operators (`MatrixFreeLinOp`);
* identity/zero operators;
* adjoint views;
* algebraic compositions (`SumLinOp`, `ComposedLinOp`, `ScaledLinOp`, `make_*`);
* product/block operators (`BlockDiagonalLinOp`, `StackedLinOp`, `SumToSingleLinOp`).

Each entry: one sentence with the map it represents and its adjoint behavior, then a link.

Done condition: public LinOp objects are summarized and linked.

### C29 — Add `functional` index page

Create or update an index page summarizing public functionals.

Group by role:

* base classes (`Functional`);
* differentiable functionals;
* linear functionals (`LinearFunctional`, `InnerProductFunctional`, `MatrixFreeLinearFunctional`);
* quadratic functionals (`QuadraticForm`, `LinOpQuadraticForm`);
* composed/pulled-back functionals (`ComposedFunctional`, `make_functional_composed`);
* utilities.

Done condition: public functional objects are summarized and linked.

### C30 — Add `linalg` index page

Create or update an index page summarizing linalg methods.

Group by role:

* linear solves (`cg`);
* least squares (`lsqr`);
* eigenvalue/spectral methods (`lanczos_smallest`, `power_iteration`);
* matrix functions (`expm_multiply`);
* result types;
* helper routines.

Each entry should briefly state the mathematical assumptions (one line: problem class, operator requirements, geometry awareness).

Done condition: public linalg methods are summarized and linked.

### C31 — Add backend/context index page

Create or update an index page summarizing backend/context APIs.

It should explain:

* Context;
* BackendOps;
* available backend families (NumPy always; JAX/Torch/CuPy as extras);
* how users choose a backend;
* what backend portability does and does not mean.

Done condition: public backend/context APIs are summarized and linked.

## Validation commands

Run the relevant project commands. Adjust exact commands to the repository setup.

Suggested minimum:

```bash
python -m pytest
python -m pytest --doctest-modules spacecore
python -m sphinx -W -b html docs/source docs/_build/html
python -m ruff check .
```

Doctest strategy:

* run `--doctest-modules` in two environments if feasible: NumPy-only, and full extras — doctests must pass in **both** (this is why doctests are NumPy-only and optional-backend examples are code-blocks);
* if `--doctest-modules` is not currently wired into CI, wire it in as part of this task and note it in the report.

If the project has a dedicated documentation command, use that instead.

The documentation build must pass with warnings treated as errors.

## Final report

At the end, provide a concise report containing:

1. Files changed.
2. Pages rewritten.
3. Tutorials executed and their status.
4. Tutorials updated, with their filled acceptance checklists.
5. Tutorials marked obsolete or split into follow-ups, and learning-path gaps filed.
6. Public docstring coverage summary, including tiny-example coverage (count with examples / count total, with reasons for exceptions).
7. Remaining documentation risks, if any.
8. Commands run and results (including both doctest environments).
9. Any follow-up issues created.

## Acceptance criteria

This task is complete when:

* README accurately describes SpaceCore `0.3.x` and all its snippets execute.
* Documentation landing page provides clear navigation and a routing paragraph.
* Context, conversion, dtype, batching, and capability notes match current implementation; the capability note demonstrates one refusal.
* Every tutorial has been inventoried with quality-standard gaps recorded.
* Public tutorials either execute and meet the Tutorial quality standard (filled checklist per tutorial), or are explicitly marked obsolete/conceptual.
* Public API docstring audit exists and tracks example coverage.
* Spaces, LinOps, functionals, linalg, and backend/context docstrings are numpydoc-conformant with tiny examples (or recorded exceptions).
* `python -m pytest --doctest-modules spacecore` passes, in NumPy-only and full-extras environments.
* Docstring-writing guide exists and matches the reviewed docstrings.
* Index pages exist for space, linop, functional, linalg, and backend/context APIs.
* Strict documentation build passes.
* No stale removed APIs are presented as current APIs.