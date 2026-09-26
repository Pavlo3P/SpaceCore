# Task: Implement Phase D end-to-end worked example for SpaceCore 0.3.1

## Goal

Implement one complete, documented, automatically tested SpaceCore worked example for the `0.3.1` release.

The example must demonstrate why SpaceCore’s abstractions matter. It should not be a generic Euclidean `Ax = b` example that could be written just as clearly with NumPy or SciPy. It must exercise:

* first-class spaces;
* non-Euclidean inner products or another structured-space feature;
* metric-correct adjoints;
* `LinOp` construction and algebra;
* one SpaceCore algorithm;
* comparison against an independent dense/reference implementation;
* documentation-quality explanation.

## Proposed worked example

Implement a small weighted Tikhonov inverse problem on non-Euclidean finite-dimensional spaces.

Let

[
X = \mathbb{R}^n,
\qquad
Y = \mathbb{R}^m,
]

but do not equip them with the coordinate Euclidean inner products. Instead define

[
\langle x, z\rangle_X = x^\top G_X z,
\qquad
\langle y, w\rangle_Y = y^\top G_Y w,
]

where

[
G_X \in \mathbb{R}^{n \times n},
\qquad
G_Y \in \mathbb{R}^{m \times m}
]

are symmetric positive-definite matrices.

Let

[
A : X \to Y
]

be represented in coordinates by a dense matrix

[
M \in \mathbb{R}^{m \times n}.
]

Given data

[
b \in Y
]

and regularization strength

[
\lambda > 0,
]

solve

[
\min_{x \in X}
\frac{1}{2}|Ax - b|_Y^2
+
\frac{\lambda}{2}|x|_X^2.
]

The first-order optimality equation is

[
(A^\sharp A + \lambda I_X)x = A^\sharp b,
]

where

[
A^\sharp : Y \to X
]

is the metric adjoint, not the coordinate transpose. In coordinates,

[
A^\sharp = G_X^{-1}M^\top G_Y.
]

Equivalently, the dense reference system is

M^\top G_Y b.
]

This example should show that using the ordinary transpose (M^\top) as the adjoint is mathematically wrong unless (G_X = I_n) and (G_Y = I_m).

## Why this example is appropriate

This example is small, deterministic, and easy to validate. It also demonstrates SpaceCore’s central value proposition:

* `A.rapply` should mean the metric adjoint (A^\sharp), not merely the coordinate transpose (M^\top).
* The algorithm should be written against the spaces (X) and (Y), not against raw arrays alone.
* The same mathematical code should remain readable even when the geometry changes.
* The reference dense solve gives an independent correctness oracle.

The example should be suitable for documentation, CI, and release validation.

## Files to add or modify

Use the repository’s existing documentation and example layout. If there is already an examples directory, place the executable example there. Otherwise create a clear minimal structure.

Suggested files:

```text
examples/
  weighted_tikhonov.py

docs/
  examples/
    weighted_tikhonov.md

tests/
  examples/
    test_weighted_tikhonov.py
```

Adapt the exact paths to the current project conventions.

## Implementation requirements

### 1. Problem generation

Create a deterministic helper, for example:

```python
def make_weighted_tikhonov_problem(
    n: int = 32,
    m: int = 48,
    lam: float = 1e-2,
    seed: int = 0,
):
    ...
```

It should return all data needed for both the reference solve and SpaceCore solve:

* coordinate matrix `M` with shape `(m, n)`;
* SPD domain metric matrix `Gx` with shape `(n, n)`;
* SPD codomain metric matrix `Gy` with shape `(m, m)`;
* data vector `b` with shape `(m,)`;
* regularization parameter `lam`.

Use a stable SPD construction, for example:

```python
Q = rng.normal(size=(n, n))
Gx = Q.T @ Q + alpha * np.eye(n)
```

Use moderate conditioning. Do not make the example numerically fragile.

Default dimensions should be small enough for CI.

### 2. Independent reference solution

Implement a dense NumPy reference solve that does not depend on SpaceCore.

It should solve

```python
lhs = M.T @ Gy @ M + lam * Gx
rhs = M.T @ Gy @ b
x_ref = np.linalg.solve(lhs, rhs)
```

Also compute useful diagnostic values:

* objective value;
* residual norm in (Y);
* regularization norm in (X);
* first-order residual.

The reference objective is

```python
0.5 * (M @ x - b).T @ Gy @ (M @ x - b)
+ 0.5 * lam * x.T @ Gx @ x
```

The first-order residual is

```python
(M.T @ Gy @ M + lam * Gx) @ x - M.T @ Gy @ b
```

### 3. SpaceCore solution

Implement the same problem using SpaceCore abstractions.

The code should construct:

* a domain space (X) with inner product induced by `Gx`;
* a codomain space (Y) with inner product induced by `Gy`;
* a `LinOp` (A : X \to Y) from the coordinate matrix `M`;
* the metric adjoint (A^\sharp : Y \to X) using SpaceCore’s `rapply`/adjoint mechanism;
* the normal-equation operator

```python
normal = A.H @ A + lam * IdentityLinOp(X)
```

or the equivalent idiomatic SpaceCore expression.

Then solve

```python
normal x = A.H b
```

with a SpaceCore iterative algorithm, preferably `cg`, because the normal operator is self-adjoint positive-definite on (X).

Do not shortcut the SpaceCore solution by directly forming the dense NumPy reference system. The SpaceCore path must actually use spaces, LinOps, metric adjoints, and the solver.

### 4. Demonstrate the adjoint issue

Add a small check or explanatory computation showing that the coordinate transpose is not the correct adjoint for the chosen spaces.

For example, for random

[
x \in X,
\qquad
y \in Y,
]

verify that SpaceCore’s metric adjoint satisfies

\langle x, A^\sharp y\rangle_X
]

up to numerical tolerance.

Also compute what happens if one incorrectly uses

[
M^\top y
]

as the adjoint. Show that the identity generally fails.

This should appear both in the executable example and in the documentation explanation.

### 5. Validation criteria

The test suite must verify:

1. The reference solution satisfies the dense optimality system.
2. The SpaceCore solution agrees with the reference solution:

```python
np.linalg.norm(x_sc - x_ref) / max(1.0, np.linalg.norm(x_ref)) <= tolerance
```

Use a reasonable tolerance, for example `1e-8` for NumPy dense execution or a slightly looser tolerance if backend behavior requires it.

3. The SpaceCore objective value agrees with the reference objective value.
4. The metric-adjoint identity holds.
5. The coordinate-transpose adjoint fails for the non-Euclidean example, unless the generated metrics accidentally reduce to identity-like matrices. Make the metrics nontrivial enough that this test is robust.
6. The example script runs start to finish in CI.

### 6. Documentation page

Write a documentation page explaining the example.

The page should include:

* the mathematical problem statement;
* definitions of (X), (Y), (G_X), (G_Y), (A), (A^\sharp);
* the dense reference normal equation;
* the SpaceCore implementation route;
* why `rapply`/`.H` must use the metric adjoint;
* a compact code listing or linked source;
* expected output;
* comparison table between dense reference and SpaceCore solution.

Suggested comparison table:

```text
quantity                         reference        SpaceCore        difference
objective value                  ...              ...              ...
relative solution error           --              ...              ...
first-order residual norm         ...             ...              ...
metric-adjoint identity error     --              ...              ...
wrong-transpose identity error    --              ...              ...
```

The documentation should explicitly say:

* In Euclidean coordinates, the coordinate transpose is not the adjoint unless the spaces have Euclidean inner products.
* SpaceCore prevents this mistake by associating the operator with its domain and codomain spaces.
* The solver code is written in terms of the mathematical operator (A : X \to Y), not just the array `M`.

### 7. CI integration

Integrate the example into CI or the documentation test pipeline.

At minimum:

* add a unit test that imports and runs the example;
* keep dimensions small enough for quick execution;
* avoid randomness outside controlled seeded generation;
* avoid optional GPU/JAX/Torch requirements unless the existing test matrix already covers them;
* make the default path NumPy-only unless the project has a standard backend parametrization pattern.

The example should not require external solvers.

### 8. Public API discipline

Use only public SpaceCore APIs.

Do not rely on private modules or private methods unless no public API exists. If a private API is needed, open a separate follow-up issue explaining which public API is missing.

The example should be written in a style suitable for users to copy.

### 9. Acceptance criteria

This task is complete when:

* `examples/weighted_tikhonov.py` or the chosen equivalent exists and runs from the command line;
* the example constructs non-Euclidean spaces;
* the example constructs a SpaceCore `LinOp`;
* the example uses the metric adjoint through SpaceCore;
* the example solves the problem with a SpaceCore algorithm;
* an independent dense reference solution is implemented;
* tests compare SpaceCore and reference outputs;
* the documentation page explains the mathematics and the code;
* the example is included in the docs navigation;
* the example is executed in CI or by the docs test suite;
* all tests, linting, and docs build pass.

## Non-goals

Do not add new solver algorithms.

Do not add SDPLab integration.

Do not benchmark performance beyond basic runtime sanity.

Do not introduce adapters to SciPy, PyLops, CoLA, or other libraries.

Do not make the example large or application-heavy. The goal is correctness, clarity, and demonstration of SpaceCore’s abstraction boundary.

## Suggested final output

When complete, summarize:

* which files were added or changed;
* which public APIs the example uses;
* the achieved numerical tolerance;
* the observed metric-adjoint identity error;
* the observed wrong-transpose identity error;
* how the example is run locally;
* how CI executes it.
