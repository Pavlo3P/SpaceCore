# SpaceCore tutorials

An executable, visual learning path for SpaceCore. The same notebooks are rendered in the
documentation under `docs/source/tutorials/` (see
[the tutorials index](../docs/source/tutorials/index.rst)).

## Reading order

The four **foundations** build on each other; the four **worked examples** can be read in any
order once the foundations are in place.

### Foundations

1. [`01_backend_and_context.ipynb`](01_backend_and_context.ipynb) — why the backend is an
   explicit `Context`; running one routine on NumPy **and** JAX; check levels; `convert`.
2. [`02_linear_algebra.ipynb`](02_linear_algebra.ipynb) — spaces with geometry, operators
   `A : X -> Y`, and a conjugate-gradient solve.
3. [`03_functionals.ipynb`](03_functionals.ipynb) — scalar objectives, metric-aware
   gradients, and gradient descent.
4. [`04_tree_spaces.ipynb`](04_tree_spaces.ipynb) — structured (tuple and **named**) unknowns,
   block operators, and a block solve over a `TreeSpace`.

### Worked examples

5. [`05_weighted_tikhonov.ipynb`](05_weighted_tikhonov.ipynb) — a deblurring inverse problem
   solved with **metric adjoints** and weighted (non-Euclidean) geometry.
6. [`06_optimal_transport.ipynb`](06_optimal_transport.ipynb) — marginalisation as a
   matrix-free operator; Sinkhorn powered by its adjoint.
7. [`07_manifold_descent.ipynb`](07_manifold_descent.ipynb) — a custom non-Euclidean geometry
   and Riemannian gradient descent on a manifold.
8. [`08_pdhg_conic_program.ipynb`](08_pdhg_conic_program.ipynb) — a primal–dual (PDHG) solver
   for a conic program using a Jordan-algebra cone projection.

### Performance and internals

9. [`09_kernels_and_fusion.ipynb`](09_kernels_and_fusion.ipynb) — optimized-kernel
   **dispatch** (ADR-016) and operator **fusion** (ADR-021): running the fastest bit-exact
   kernel with `dispatch_mode`, and multiplying operators together with `fuse()`. Both are
   opt-in and neither densifies a matrix-free operator.

## Running

Every notebook runs on the NumPy backend with only `numpy` and `matplotlib`; tutorial 1 also
shows the JAX backend when `jax` is installed. To execute a notebook in place:

```bash
jupyter nbconvert --to notebook --execute --inplace tutorials/01_backend_and_context.ipynb
```

The release-candidate script `scripts/verify_release_candidate.sh` executes the full set.
