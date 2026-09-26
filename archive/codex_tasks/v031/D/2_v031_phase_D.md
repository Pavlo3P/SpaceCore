# Task: Add tutorial notebook for the weighted Tikhonov worked example

## Goal

Add a documentation-quality Jupyter notebook tutorial for the SpaceCore `0.3.1` weighted Tikhonov worked example.

The notebook should explain the mathematics, run the existing executable example code, and demonstrate why SpaceCore’s space-aware operator model matters for non-Euclidean inner products and metric adjoints.

This task is a follow-up to the implemented weighted Tikhonov example. It should not replace the executable script. The notebook should reuse the existing implementation as much as possible.

---

## Context

The worked example solves a weighted Tikhonov inverse problem on finite-dimensional real vector spaces

```text
X = R^n,
Y = R^m,
```

but with non-Euclidean inner products

```text
<x, z>_X = x^T G_X z,
<y, w>_Y = y^T G_Y w,
```

where `G_X` and `G_Y` are symmetric positive-definite metric matrices.

The forward operator is

```text
A : X -> Y
```

with coordinate matrix `M`.

The optimization problem is

```text
minimize over x in X:

    1/2 ||A x - b||_Y^2 + lambda/2 ||x||_X^2.
```

The correct optimality equation is

```text
(A^sharp A + lambda I_X) x = A^sharp b,
```

where `A^sharp : Y -> X` is the metric adjoint.

In coordinates,

```text
A^sharp = G_X^{-1} M^T G_Y.
```

The notebook must make clear that the coordinate transpose `M.T` is not the adjoint unless both spaces are Euclidean coordinate spaces.

---

## Main deliverable

Add a notebook, for example:

```text
tutorials/weighted_tikhonov.ipynb
```

or the equivalent location used by the repository.

The notebook should be suitable for:

* reading as a tutorial;
* execution from a clean environment;
* inclusion in the documentation;
* use as a release-candidate artifact for `0.3.1`.

---

## Relationship to existing code

Do not duplicate the whole example implementation in the notebook.

Instead, import and reuse the existing example module, for example:

```python
from examples.weighted_tikhonov import (
    make_weighted_tikhonov_problem,
    dense_reference_solve,
    solve_with_spacecore,
    adjoint_diagnostics,
    run_example,
)
```

If the examples directory is not importable from the notebook context, add the minimal packaging or path adjustment needed according to the repository’s documentation conventions.

Avoid copying large blocks of logic from the script into the notebook. The notebook should explain and call the implementation, not become a second independent implementation that can drift.

---

## Notebook structure

The notebook should contain the following sections.

### 1. Title and motivation

Title:

```text
Weighted Tikhonov inverse problem on non-Euclidean spaces
```

Explain the purpose:

* this is not a generic Euclidean least-squares example;
* the point is that spaces carry inner products;
* the adjoint depends on the domain and codomain geometry;
* SpaceCore’s `A.H` / `rapply` should represent the metric adjoint, not blindly the coordinate transpose.

State the central lesson:

```text
The matrix M does not determine the adjoint by itself. The adjoint is determined by M together with the inner products on X and Y.
```

### 2. Mathematical problem

Introduce the spaces:

```text
X = R^n,
Y = R^m.
```

Define the inner products:

```text
<x, z>_X = x^T G_X z,
<y, w>_Y = y^T G_Y w.
```

Specify the objects and their spaces:

```text
G_X in R^{n x n},
G_Y in R^{m x m},
M in R^{m x n},
A : X -> Y,
b in Y,
lambda > 0.
```

Define the objective:

```text
J(x) =
    1/2 (M x - b)^T G_Y (M x - b)
    + lambda/2 x^T G_X x.
```

Define the metric adjoint:

```text
<Ax, y>_Y = <x, A^sharp y>_X.
```

Then derive the coordinate formula:

```text
A^sharp = G_X^{-1} M^T G_Y.
```

Finally derive the dense normal equation:

```text
(M^T G_Y M + lambda G_X) x = M^T G_Y b.
```

The derivation should be short but explicit enough for a user to understand where every term comes from.

### 3. Generate a deterministic problem

Call the existing helper:

```python
problem = make_weighted_tikhonov_problem(
    n=32,
    m=48,
    lam=1e-2,
    seed=0,
)
```

Print or display:

* `M.shape`;
* `Gx.shape`;
* `Gy.shape`;
* `lam`;
* a short check that the metric weights are nonconstant.

Add a sentence explaining that the metrics are diagonal SPD matrices in this example to keep the implementation simple while still non-Euclidean.

### 4. Dense NumPy reference solve

Call:

```python
x_ref, ref_diag = dense_reference_solve(problem)
```

Display:

* objective value;
* residual norm in `Y`;
* regularization norm in `X`;
* first-order residual norm.

Explain that this dense reference solve is independent of SpaceCore and acts as the correctness oracle.

The notebook should show the reference linear system explicitly:

```python
lhs = problem.M.T @ problem.Gy @ problem.M + problem.lam * problem.Gx
rhs = problem.M.T @ problem.Gy @ problem.b
```

but it should not need to reimplement the full reference solve if the helper already exists.

### 5. SpaceCore solve

Call:

```python
sc_solve = solve_with_spacecore(problem)
```

Explain what the SpaceCore path does:

* constructs `X` with the `G_X`-induced inner product;
* constructs `Y` with the `G_Y`-induced inner product;
* constructs `A : X -> Y`;
* forms the normal operator using `A.H @ A + lambda * I`;
* solves the system with SpaceCore `cg`.

The notebook should include a compact code cell that either calls the helper or shows the essential SpaceCore construction.

Prefer a compact pedagogical version such as:

```python
ctx, X, Y, A = build_spacecore_operator(problem)

normal = A.H @ A + problem.lam * sc.IdentityLinOp(X)
rhs = A.H.apply(ctx.asarray(problem.b))
result = sc.cg(normal, rhs, tol=1e-12, atol=0.0, maxiter=2 * problem.M.shape[1])
```

If exact names differ, use the current public API.

Do not rely on private APIs.

### 6. Compare reference and SpaceCore solutions

Compute:

```python
relative_solution_error =
    norm(x_sc - x_ref) / max(1, norm(x_ref))
```

Display a comparison table with at least:

```text
quantity                         reference        SpaceCore        difference
objective value                  ...              ...              ...
relative solution error           --              --               ...
first-order residual norm         ...              ...              ...
CG converged                      --              ...
CG iterations                     --              ...
```

Use a pandas DataFrame if pandas is already a documentation dependency. Otherwise print a simple Markdown table.

State the expected tolerance, for example:

```text
The relative solution error should be near machine precision for the default NumPy run.
```

Do not overpromise exact numbers if they may vary slightly across environments.

### 7. Demonstrate the adjoint identity

Call:

```python
adj = adjoint_diagnostics(problem)
```

Display:

* metric-adjoint identity error;
* wrong-transpose identity error.

Explain the two identities.

Correct metric adjoint:

```text
<Ax, y>_Y = <x, A^sharp y>_X.
```

Wrong coordinate transpose test:

```text
<Ax, y>_Y ?= <x, M^T y>_X.
```

Explain that the first error should be near floating-point precision, while the second should be visibly nonzero for the chosen non-Euclidean metrics.

This is the most important conceptual section of the notebook.

### 8. Summary

End with a concise summary:

* the same coordinate matrix `M` has different adjoints depending on the spaces;
* the dense reference system contains `G_X` and `G_Y`;
* SpaceCore keeps these metrics attached to the spaces;
* `A.H` is therefore the mathematically correct adjoint;
* this prevents a silent but serious bug common in weighted inverse problems.

---

## Execution requirements

The notebook must run from a clean environment.

It should not require:

* SDPLab;
* external solvers;
* GPU;
* JAX;
* Torch;
* internet access.

Default execution should use NumPy only.

The notebook should run quickly enough for CI or documentation execution. Keep the default problem size small.

---

## Documentation integration

Add the notebook to the documentation navigation.

Depending on the docs system, update one of:

```text
docs/index.md
docs/tutorials/index.md
docs/examples/index.md
mkdocs.yml
docs/conf.py
```

The notebook should be discoverable from the documentation landing page or tutorials page.

Use a title consistent with other tutorials.

---

## CI or release-candidate integration

Add the notebook to the release-candidate tutorial execution path.

Acceptable options:

1. execute notebooks during docs build;
2. add a dedicated notebook test using `nbconvert`;
3. add a lightweight test that imports the notebook’s underlying example module and verifies the same numerical quantities.

Prefer notebook execution if the project already supports it. Otherwise, create a follow-up issue for full notebook execution and at least ensure the underlying example is tested.

The release-candidate task should be able to say:

```text
The weighted Tikhonov tutorial notebook executes successfully from a clean environment.
```

---

## Style requirements

The notebook should be explanatory but compact.

Avoid long walls of text.

Every mathematical object should have its space stated explicitly.

Use notation consistently:

* `X` for the domain;
* `Y` for the codomain;
* `G_X` for the domain metric;
* `G_Y` for the codomain metric;
* `A : X -> Y` for the operator;
* `A^sharp : Y -> X` for the metric adjoint;
* `M` for the coordinate matrix.

Do not use `A.T` as notation for the adjoint unless explicitly discussing the wrong Euclidean-coordinate transpose.

Prefer `A^sharp` in prose and equations.

---

## Tests and validation

The notebook or its companion test must verify:

1. The dense reference solution has small first-order residual.
2. The SpaceCore solution agrees with the dense reference solution.
3. The SpaceCore objective agrees with the reference objective.
4. The metric-adjoint identity error is small.
5. The wrong-transpose identity error is nontrivial.
6. The notebook can be executed without manual state.

Suggested tolerances:

```python
relative_solution_error <= 1e-8
objective_difference <= 1e-8
metric_identity_error <= 1e-10
wrong_transpose_identity_error >= 1e-3
```

Adjust only if justified by the current implementation.

---

## Acceptance criteria

This task is complete when:

* a weighted Tikhonov tutorial notebook exists;
* the notebook imports or reuses the existing executable example;
* the notebook explains the mathematical problem;
* the notebook states the spaces of all main objects;
* the notebook derives the metric adjoint formula;
* the notebook runs the dense reference solve;
* the notebook runs the SpaceCore solve;
* the notebook compares the two solutions numerically;
* the notebook demonstrates that the coordinate transpose is the wrong adjoint;
* the notebook is linked from the docs navigation;
* the notebook or its underlying computations are covered by CI;
* all lint, tests, and docs checks pass.

---

## Non-goals

Do not add new SpaceCore APIs.

Do not add new solvers.

Do not add SDPLab integration.

Do not benchmark performance.

Do not introduce SciPy, PyLops, CoLA, or other external-library adapters.

Do not turn the notebook into a large application example.

The goal is clarity: one small problem showing why metric-aware spaces and adjoints matter.

---

## Final PR summary

The final PR should summarize:

* where the notebook was added;
* how it is linked in the documentation;
* whether the notebook itself is executed in CI;
* achieved relative solution error;
* achieved objective difference;
* achieved metric-adjoint identity error;
* achieved wrong-transpose identity error;
* any follow-up issues opened.
