Tutorials
=========

A guided path from SpaceCore's core abstractions to three complete algorithms. Every
page below is a runnable Jupyter notebook in ``tutorials/`` — read it here or open the
notebook and execute it yourself. The foundations build on each other; the worked
algorithms can be read in any order once the foundations are in place.

Foundations
-----------

* :doc:`01_backend_and_context` — Why the backend is an explicit ``Context``; one routine on NumPy and JAX; check levels.
* :doc:`02_linear_algebra` — Spaces with geometry, operators :math:`A:X\to Y`, and a conjugate-gradient solve.
* :doc:`03_functionals` — Scalar objectives, metric-aware gradients, and gradient descent.
* :doc:`04_tree_spaces` — Structured unknowns, block operators, and a block solve over a ``TreeSpace``.

Worked examples
---------------

* :doc:`05_weighted_tikhonov` — An inverse problem solved with metric adjoints and weighted geometry.
* :doc:`06_optimal_transport` — Marginalisation as a matrix-free operator; Sinkhorn powered by its adjoint.
* :doc:`07_manifold_descent` — A custom non-Euclidean geometry and Riemannian gradient descent.
* :doc:`08_pdhg_conic_program` — A primal--dual solver for a conic program with a Jordan-cone projection.

.. toctree::
   :hidden:
   :maxdepth: 1

   01_backend_and_context
   02_linear_algebra
   03_functionals
   04_tree_spaces
   05_weighted_tikhonov
   06_optimal_transport
   07_manifold_descent
   08_pdhg_conic_program
