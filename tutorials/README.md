# SpaceCore tutorials

This folder contains the executable notebook set referenced by `docs/source/tutorials/index.rst`.

## Recommended reading order

1. `3_Space.ipynb` - define dense coordinate spaces and check their geometry.
2. `4_LinOp.ipynb` - build maps `A : X -> Y`, use `apply`, and use metric adjoints.
3. `2_Context.ipynb` - make backend ownership explicit.
4. `1_BackendOps.ipynb` - write backend-agnostic helper code.
5. `5_Conversion_Policy.ipynb` - rebuild spaces and operators under another context.
6. `weighted_tikhonov.ipynb` - worked inverse problem with non-Euclidean adjoints.
7. `7_Quadratic_Program.ipynb` - optional SciPy optimization example.
8. `8_Linalg_MatrixFree.ipynb` - matrix-free iterative-solver example.
9. `9_Linalg_Comparison.ipynb` - compare against NumPy, SciPy, and optional JAX references.
10. `6_Regularized_Opt_Transport.ipynb` - retained advanced OT example, outside the 0.3.1 release gate.

## Notebook index

| Notebook | Purpose | Status | Optional dependencies |
| --- | --- | --- | --- |
| `1_BackendOps.ipynb` | Backend operation interface and its relation to contexts. | active | JAX |
| `2_Context.ipynb` | Context ownership, dtype policy, and runtime checks. | active | JAX optional |
| `3_Space.ipynb` | Concrete spaces, abstract `VectorSpace` role, and product geometry. | active | None |
| `4_LinOp.ipynb` | Dense, sparse, product, and matrix-free linear operators. | active | SciPy |
| `5_Conversion_Policy.ipynb` | Explicit target-context conversion and dtype behavior. | active | None |
| `6_Regularized_Opt_Transport.ipynb` | Entropy-regularized OT with reusable SpaceCore objects. | retained | JAX, Optax, Matplotlib |
| `7_Quadratic_Program.ipynb` | Small constrained quadratic program using SpaceCore objects and SciPy. | advanced | SciPy, JAX optional |
| `8_Linalg_MatrixFree.ipynb` | Matrix-free CG, LSQR, power iteration, Lanczos, and expm actions. | active | SciPy |
| `9_Linalg_Comparison.ipynb` | Iterative solver comparisons against dense and external references. | advanced | SciPy, JAX optional |
| `weighted_tikhonov.ipynb` | Official 0.3.1 SpaceCore-native worked example. | active | None |

The active 0.3.1 notebook gate executes every active and advanced notebook listed above except the retained regularized OT notebook. Regularized OT is kept as an illustrative advanced example with optional dependencies, but it is not part of the 0.3.1 release-candidate gate.
