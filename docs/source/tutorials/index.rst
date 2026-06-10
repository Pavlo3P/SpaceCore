Tutorials
=========

This learning path takes a new user from contexts and spaces to a small
space-aware solve. It also records the status of every notebook in
``tutorials/`` so the release gate is explicit.

Recommended order
-----------------

1. :doc:`spaces` - define dense coordinate spaces and check their geometry.
2. :doc:`linops` - build maps :math:`A : X \to Y`, use ``apply`` and metric adjoints.
3. :doc:`context` - make backend ownership explicit.
4. :doc:`backend_ops` - write backend-agnostic helper code.
5. :doc:`conversion_policy` - rebuild spaces and operators under another context.
6. :doc:`weighted_tikhonov` - worked inverse problem with non-Euclidean adjoints.
7. ``7_Quadratic_Program.ipynb`` - optional SciPy optimization example.
8. ``8_Linalg_MatrixFree.ipynb`` - matrix-free iterative-solver example.
9. ``9_Linalg_Comparison.ipynb`` - comparison against NumPy, SciPy, and optional JAX references.
10. :doc:`regularized_ot` - retained advanced OT example, outside the 0.3.1 release gate.

Notebook status for the 0.3.1 docs baseline
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 34 16 18

   * - Notebook or page
     - Purpose
     - Status
     - Optional dependencies
   * - ``1_BackendOps.ipynb`` / :doc:`backend_ops`
     - Backend operation interface and its relation to contexts
     - active
     - JAX
   * - ``2_Context.ipynb`` / :doc:`context`
     - Context ownership and checks
     - active
     - JAX optional
   * - ``3_Space.ipynb`` / :doc:`spaces`
     - Concrete spaces, abstract ``VectorSpace`` role, and product geometry
     - active
     - None
   * - ``4_LinOp.ipynb`` / :doc:`linops`
     - Dense, sparse, product, and matrix-free linear operators
     - active
     - SciPy
   * - ``5_Conversion_Policy.ipynb`` / :doc:`conversion_policy`
     - Explicit target-context conversion and dtype behavior
     - active
     - None
   * - ``6_Regularized_Opt_Transport.ipynb`` / :doc:`regularized_ot`
     - Entropy-regularized OT with reusable SpaceCore objects
     - retained
     - JAX, Optax, Matplotlib
   * - ``7_Quadratic_Program.ipynb``
     - Small constrained quadratic program using SpaceCore objects and SciPy
     - advanced
     - SciPy, JAX optional
   * - ``8_Linalg_MatrixFree.ipynb``
     - Matrix-free CG, LSQR, power iteration, Lanczos, and expm actions
     - active
     - SciPy
   * - ``9_Linalg_Comparison.ipynb``
     - Iterative solver comparisons against dense and external references
     - advanced
     - SciPy, JAX optional
   * - ``weighted_tikhonov.ipynb`` / :doc:`weighted_tikhonov`
     - Official 0.3.1 SpaceCore-native worked example
     - active
     - None

The active 0.3.1 notebook gate executes every active and advanced notebook
listed above except the retained regularized OT notebook. Regularized OT is kept
as an illustrative advanced example with optional dependencies, but it is not
part of the 0.3.1 release-candidate gate.

.. toctree::
   :maxdepth: 1

   spaces
   linops
   context
   backend_ops
   conversion_policy
   weighted_tikhonov
   regularized_ot
