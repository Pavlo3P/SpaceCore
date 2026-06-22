SpaceCore
=========

SpaceCore provides typed vector spaces, structured elements, linear operators,
functionals, and small linear-algebra utilities for backend-aware numerical
code.

A SpaceCore operator is a typed map :math:`A : X \to Y` between spaces
:math:`X` and :math:`Y`. The spaces carry the rules needed to validate elements,
compute inner products, flatten structured values, and interpret adjoints.
``rapply`` is the metric adjoint with respect to the domain and codomain inner
products; it is the coordinate conjugate transpose only in Euclidean geometry.

Core navigation
---------------

* :doc:`api/spaces` - spaces, geometries, structure, and validation.
* :doc:`api/linops` - linear operators and algebraic compositions.
* :doc:`api/functionals` - scalar-valued maps and gradients.
* :doc:`api/linalg` - CG, LSQR, Lanczos, power iteration, and exponential action.
* :doc:`api/backend` and :doc:`api/context` - backend operations and contexts.
* :doc:`tutorials/index` - step-by-step learning path.
* :doc:`design/index` - conversion, dtype, batching, capability, and geometry notes.
* :doc:`dev/index` - contributor-facing architecture decision records.
* :doc:`release_notes` - migration and release information.

Where do I start?
-----------------

First-time users should start with :doc:`tutorials/01_backend_and_context`, then
work through :doc:`tutorials/02_linear_algebra` and :doc:`tutorials/03_functionals`.
For structured unknowns continue with :doc:`tutorials/04_tree_spaces`. The worked
examples — a :doc:`Tikhonov inverse problem <tutorials/05_weighted_tikhonov>`,
:doc:`optimal transport <tutorials/06_optimal_transport>`,
:doc:`manifold descent <tutorials/07_manifold_descent>`, and
:doc:`conic optimisation <tutorials/08_pdhg_conic_program>` — apply these pieces to
real problems. If your problem uses non-Euclidean geometry, read
:doc:`design/geometry` before relying on adjoints or solver preconditions.

What SpaceCore is not
---------------------

SpaceCore is not a full solver suite and not a replacement for NumPy, SciPy,
JAX, Torch, CuPy, PETSc, Krylov.jl, or PyLops. Its native solvers are a small
correctness baseline and substrate layer for space-aware algorithms. External
adapters and backend-specific fast paths can be added where a project needs
solver breadth or production performance.

Installation
------------

.. code-block:: bash

   pip install spacecore
   pip install "spacecore[jax]"
   pip install "spacecore[torch]"
   pip install "spacecore[cupy]"

Minimal example
---------------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx)
   A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, ctx)
   b = ctx.asarray([4.0, 9.0])

   result = sc.cg(A, b, tol=1e-12, maxiter=10)
   print(result.x)
   print(result.converged)

Expected output:

.. code-block:: text

   [2. 3.]
   True

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
   design/index
   dev/index
   api/index
   release_notes
