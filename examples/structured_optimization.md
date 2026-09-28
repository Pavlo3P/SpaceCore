# Build the mathematics once; change the algorithm and representation

[Runnable example](/Users/pavlopelikh/Desktop/Dev/SpaceCore/examples/structured_optimization.py)

From the repository root, run:

```sh
.venv/bin/python -m examples.structured_optimization
```

The example uses the installed SpaceCore dependencies only. It solves the same
problem using proximal gradient, PDHG, and ADMM, with both dense NumPy values and
custom dictionary-backed coefficient values. ADMM then runs with a faster
subproblem implementation without changing its iteration.

## 1. Construct the mathematical problem

We solve

\[
\min_x\; g(x)+f(Kx),\qquad
f(y)=\tfrac12\|y-b\|_2^2,\qquad g(x)=\lambda\|x\|_1.
\]

`K` is a nontrivial lower bidiagonal operator with positive diagonal entries
between 1 and 2 and subdiagonal -0.25. The example uses 32 coefficients and
`lambda=0.15`. Its Euclidean operator norm is at most 2.25, by the triangle
inequality applied to the diagonal and shift operators.

The construction is ordinary SpaceCore algebra:

```python
smooth = f @ K
objective = g + smooth
```

`f` and `g` subclass `Functional`; `K` is a `LinOp`. SpaceCore builds the lazy
composition/sum and computes the smooth pullback gradient through `K.rapply`.
The individual terms remain accessible to the algorithms.

The proximal methods below are local extensions on these example functionals.
They predate the public optional `Functional.prox` and `conjugate_prox` contract.
The base `Functional` now accepts general `Space` domains. See
[`proximal_contract.py`](proximal_contract.py) for registration through the public
rule table and unchanged algorithms across weighted geometry and compiled JAX.

## 2. Supply the operations each algorithm needs

`SquaredDistance` supplies value, gradient, and the conjugate proximal action:

\[
\nabla f(y)=y-b,\qquad
f^*(p)=\tfrac12\|p\|^2+\langle p,b\rangle,\qquad
\operatorname{prox}_{\sigma f^*}(p)=\frac{p-\sigma b}{1+\sigma}.
\]

These methods use space arithmetic, so the same implementation serves both
representations. `L1Penalty` supplies value and soft thresholding through native
adapter operations. This coordinate-defined norm belongs to the coefficient
basis; it is not being claimed to be invariant under arbitrary basis changes.

| Method | Mathematical update |
|---|---|
| Proximal gradient | Gradient of `f @ K`, followed by prox of `g` |
| PDHG | Prox of `f*`, application of `K` and its adjoint, prox of `g`, extrapolation |
| ADMM | Introduce `x=z`; alternate the quadratic x solve and prox of `g`, then update the scaled multiplier |

For example, the actual PDHG iteration is:

```python
y = f.conjugate_prox(Y.axpy(sigma, K.apply(x_bar), y), sigma)
previous = x
x = g.prox(X.axpy(-tau, K.rapply(y), x), tau)
x_bar = X.axpy(1, X.axpy(-1, previous, x), x)
```

It uses no element indexing, NumPy conversion, or representation-dependent branch.
The code fixes `tau=sigma=0.9/2.25`, so the standard convex step-size condition
`tau*sigma*||K||² < 1` holds. Proximal gradient uses `0.95/2.25²`.
See the [Chambolle–Pock formulation](https://optimization-online.org/2010/06/2646/).

ADMM uses the splitting

\[
\min_{x,z} f(Kx)+g(z),\qquad x=z,
\]

and the x subproblem is

\[
(K^*K+\rho I)x=K^*b+\rho(z-u).
\]

The loop takes a prepared `solve_x(rhs, initial)` callback. Its penalty `rho` must
match the penalty used to prepare that callback. The implementation warm-starts
iterative solves and checks their convergence. See the
[ADMM reference](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html).

## 3. Replace the element representation

`DenseCoefficients` uses NumPy arrays and a `DenseLinOp`.

`SparseCoefficients` uses `Coefficients(entries={index: value})` and a
`MatrixFreeLinOp`. Its values have no arithmetic overloads and deliberately raise
if converted to a NumPy array. Its space has no `shape`, `flatten`, or `unflatten`.
Addition, scaling, inner products, thresholding, and operator actions are
implemented natively.

Both adapters describe the same Euclidean coefficient space and mathematical
operator. The problem and algorithm definitions are unchanged. Operator actions
and the prepared solve may visit every coefficient; this demonstrates native
representation support, not an optimized sparse-library implementation.

The dictionary wrapper is frozen, but its dictionary is mutable. The example
treats input values as owned, read-only data and constructs fresh outputs; callers
must not mutate values during a solve. The example implements real NumPy float64
execution only and does not claim JAX tracing or cross-backend conversion.

## 4. Optimize the implementation after defining the algorithm

The first ADMM execution uses existing SpaceCore CG on the lazy operator

```python
system = K.H @ K + rho * sc.IdentityLinOp(X)
```

The second uses the known bidiagonal structure of `K`. Its normal system is
tridiagonal, so the adapter prepares an LDL-transpose factorization once and
performs linear-time triangular substitutions for each right-hand side.
This path never constructs the dense normal matrix or converts a custom value
to a dense backend array. Representation-specific coefficient access lives inside
the prepared solve, not inside ADMM.

```python
cg_solve, cg_work = prepare_cg(problem, rho=1.0, maxiter=64)
baseline = admm(problem, cg_solve)

direct_solve, direct_work = prepare_banded_solve(X, diagonal, lower, rho=1.0)
optimized = admm(problem, direct_solve)
```

The diagonal, subdiagonal, and penalty are fixed for this run. A changed operator
or penalty requires a newly prepared solve. There is no automatic mutable cache.

## 5. Check the mathematics independently

The demonstration constructs a sparse target `x*` and a valid L1 subgradient `s`,
then chooses `b` so that

\[
K^*(Kx^*-b)+\lambda s=0.
\]

Since `K` is invertible, the least-squares term is strictly convex: this condition
certifies the unique optimum. The algorithms receive only the problem data, not
the target solution.

All methods stop using the relative proximal-gradient mapping norm. ADMM also
checks relative primal and dual residuals for its splitting. The shared
stationarity check evaluates the objective gradient and prox at each iteration;
that overhead is included in the printed times.

The default run produced:

| Method | Iterations, both representations | Distance to certified optimum |
|---|---:|---:|
| Proximal gradient | 75 | approximately 2.89e-8 |
| PDHG | 65 | approximately 2.98e-8 |
| ADMM / CG | 59 | approximately 4.27e-9 |
| ADMM / prepared factorization | 59 | approximately 4.27e-9 |

All objectives agree at approximately **1.71493206**. CG used 615 total inner
iterations; the optimized variant used one factorization and 59 substitutions.
Printed timings are illustrative single runs, exclude preparation, and are not a
performance benchmark or a claim that one outer algorithm is always faster.

Run the regression checks:

```sh
.venv/bin/python -m pytest -q tests/test_structured_optimization_example.py
```

They verify the native actions against independent dense matrices, the adjoint
identity, a gradient directional derivative, the KKT certificate, convergence of
all variants, reuse of the factorization, and unchanged input values.

## What this establishes

SpaceCore already supports a useful part of the intended architecture: a problem
can retain its mathematical structure, algorithms can use operations on spaces,
and representations can supply different computations. This example demonstrates
that path with a bounded, verifiable problem.

MPS and neural-state implementations would supply their own valid operations or
parameterized subproblem solves. They do not automatically inherit the L1 prox,
convexity, exact vector arithmetic, or convergence guarantees used here. The next
library work is to formalize these optional operation contracts and finish
removing coordinate assumptions at generic boundaries.
