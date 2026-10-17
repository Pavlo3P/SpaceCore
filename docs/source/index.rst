SpaceCore
=========

Typed vector spaces and linear operators for NumPy, JAX, Torch, and CuPy.
Spaces define inner products; operators know their domain and codomain.
Adjoints and gradients follow the spaces' geometry.

Install
-------

Requires Python 3.11+.

.. code-block:: bash

   pip install spacecore          # NumPy backend
   pip install "spacecore[jax]"   # or [torch], [cupy]

Solve a linear system
---------------------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx)
   A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, ctx)
   b = ctx.asarray([4.0, 9.0])

   result = sc.cg(A, b, tol=1e-12, maxiter=10)
   print(result.x)          # [2. 3.]
   print(result.converged)  # True

``A.apply(x)`` computes the forward map; ``A.rapply(y)`` applies its metric
adjoint. The adjoint equals the coordinate conjugate transpose only in
Euclidean geometry. See :doc:`design/geometry` for weighted inner products
and solver assumptions.

Next steps
----------

* **Look up an object:** use the :doc:`api/index`.
* **Understand behavior:** read :doc:`design/index` for geometry, validation,
  batching, dtype, and backend rules.
* **Contribute:** see :doc:`dev/contributing` and the :doc:`dev/adr`.
* **Upgrade:** check :doc:`release_notes` for changes and migration steps.

SpaceCore includes a small set of native solvers and adapters for SciPy and
Optax. The API is experimental (0.x) and may change before 1.0.

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
   design/index
   dev/index
   api/index
   release_notes
   release_notes_archive
