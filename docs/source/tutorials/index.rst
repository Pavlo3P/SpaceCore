Tutorials
=========

Start with the four foundations, then choose a worked example. Each page
states prerequisites, explains the mathematical choices, and ends with
checks and a small experiment. Examples use real float64 arrays unless
stated otherwise.

Run the examples
----------------

From a checkout of the version you want to learn, create an environment
with Python 3.11 or newer, then install the package and documentation tools:

.. code-block:: console

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[docs]"

On Windows, activate with ``.venv\Scripts\activate`` instead. The docs
extra includes Matplotlib for the plots. Run each page's Python blocks
from top to bottom in a fresh Python session or notebook. NumPy is the
baseline; the optional JAX comparison in tutorial 1 is skipped if JAX is
not installed. To enable it, install ``python -m pip install -e ".[jax]"``.

The examples target the API in this checkout. Check ``spacecore.__version__``
if an installed release behaves differently. Numerical tolerances matter;
exact iteration counts, timing results, and final printed digits can vary.
The explanatory loops use NumPy for plotting and diagnostics and are not
JIT-ready solvers merely because their operators support other backends.

To execute every tutorial and its assertions without opening plot windows:

.. code-block:: console

   python scripts/check_tutorials.py

Choose a path
-------------

New to SpaceCore? Follow 1 → 2 → 3 → 4. For inverse problems, continue to 5;
for smooth constrained optimisation, choose 6 or 7; for primal-dual methods,
choose 8. Read 9 after you can build and verify an operator expression.

Foundations
-----------

* :doc:`01_backend_and_context` — Why the backend is an explicit ``Context``; one routine on NumPy and JAX; check levels.
* :doc:`02_linear_algebra` — Spaces with geometry, operators :math:`A:X\to Y`, and a conjugate-gradient solve.
* :doc:`03_functionals` — Scalar objectives, metric-aware gradients, and gradient descent.
* :doc:`04_tree_spaces` — Structured unknowns, block operators, and a block solve over a ``TreeSpace``.

Worked examples
---------------

* :doc:`05_weighted_tikhonov` — An inverse problem solved with metric adjoints and weighted geometry.
* :doc:`06_optimal_transport` — Marginalisation as a matrix-free operator; named dual potentials, functional pullbacks, and gradient ascent.
* :doc:`07_manifold_descent` — An operator-based non-Euclidean geometry and Riemannian gradient descent.
* :doc:`08_pdhg_conic_program` — A primal--dual solver for a conic program with a Jordan-cone projection.

Performance and internals
-------------------------

* :doc:`09_kernels_and_fusion` — Lazy operator algebra and explicit fusion with ``fuse()``.

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
   09_kernels_and_fusion
