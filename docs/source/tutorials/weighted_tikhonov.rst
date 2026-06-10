Weighted Tikhonov Example
=========================

Goal
   Solve a small inverse problem where the domain and codomain have non-Euclidean
   weighted inner products, and compare SpaceCore against an independent dense
   NumPy reference solve.
Prerequisites
   Know :doc:`spaces`, :doc:`linops`, and the meaning of a Hilbert-space adjoint.
Backends used
   NumPy only.

Problem statement
-----------------

Let

.. math::

   X = \mathbb{R}^n,
   \qquad
   Y = \mathbb{R}^m,

but equip the coordinate vectors with weighted inner products

.. math::

   \langle x, z\rangle_X = x^T G_X z,
   \qquad
   \langle y, w\rangle_Y = y^T G_Y w.

In this example ``G_X`` and ``G_Y`` are non-identity diagonal SPD matrices. The
operator :math:`A : X \to Y` is represented in coordinates by a dense matrix
:math:`M \in \mathbb{R}^{m \times n}`. Given data :math:`b \in Y` and
:math:`\lambda > 0`, solve

.. math::

   \min_{x \in X}
   \frac{1}{2}\|Ax - b\|_Y^2
   +
   \frac{\lambda}{2}\|x\|_X^2.

The optimality equation is

.. math::

   (A^\sharp A + \lambda I_X)x = A^\sharp b,

where :math:`A^\sharp : Y \to X` is the metric adjoint. In coordinates,

.. math::

   A^\sharp = G_X^{-1} M^T G_Y.

Multiplying the SpaceCore normal equation by :math:`G_X` gives the independent
reference system

.. math::

   (M^T G_Y M + \lambda G_X)x = M^T G_Y b.

Why the coordinate transpose is wrong
-------------------------------------

The adjoint is defined by the spaces, not by the stored matrix alone:

.. math::

   \langle Ax, y\rangle_Y = \langle x, A^\sharp y\rangle_X.

For weighted spaces, using :math:`M^T y` in place of :math:`A^\sharp y` omits
both metrics. It is correct only when both spaces use Euclidean coordinate inner
products. SpaceCore prevents this mistake by storing the operator together with
its domain and codomain spaces; ``A.rapply(y)`` and ``A.H.apply(y)`` therefore
use the metric adjoint.

Implementation route
--------------------

The executable source lives at ``examples/weighted_tikhonov.py``. It does four
separate things:

1. Creates deterministic ``M``, ``Gx``, ``Gy``, ``b``, and ``lam``.
2. Solves the dense NumPy reference system
   ``M.T @ Gy @ M + lam * Gx``.
3. Constructs SpaceCore weighted spaces, a ``DenseLinOp`` ``A : X -> Y``, the
   lazy normal operator ``A.H @ A + lam * IdentityLinOp(X)``, and solves it with
   ``cg``.
4. Verifies the metric-adjoint identity and shows that the coordinate transpose
   identity fails.

Compact SpaceCore solve
-----------------------

.. code-block:: python

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
   X = sc.DenseVectorSpace(
       (M.shape[1],),
       ctx,
       geometry=sc.WeightedInnerProduct(ctx.asarray(np.diag(Gx))),
   )
   Y = sc.DenseVectorSpace(
       (M.shape[0],),
       ctx,
       geometry=sc.WeightedInnerProduct(ctx.asarray(np.diag(Gy))),
   )

   A = sc.DenseLinOp(ctx.asarray(M), X, Y, ctx)
   normal = A.H @ A + lam * sc.IdentityLinOp(X)
   rhs = A.H.apply(ctx.asarray(b))
   result = sc.cg(normal, rhs, tol=1e-12, maxiter=2 * M.shape[1], check_every=1)

The solver code is written in terms of the mathematical map
:math:`A : X \to Y`, not in terms of raw array transposes.

Full source listing
-------------------

.. literalinclude:: ../../../examples/weighted_tikhonov.py
   :language: python
   :caption: examples/weighted_tikhonov.py

Expected output
---------------

Run from the repository root:

.. code-block:: bash

   python examples/weighted_tikhonov.py

When running from a source checkout in this repository's development virtualenv,
use ``.venv/bin/python`` if the system ``python`` is not Python 3.11 or newer.

The default deterministic problem prints output like:

.. code-block:: text

   Weighted Tikhonov inverse problem on non-Euclidean spaces
   CG converged: True in 37 iterations

   quantity                                  reference      SpaceCore     difference
   ----------------------------------------------------------------------------------
   objective value                        2.540443e-01   2.540443e-01   0.000000e+00
   relative solution error                          --             --   1.480233e-14
   first-order residual norm              4.368925e-15   3.981618e-13             --
   metric-adjoint identity error                    --   0.000000e+00             --
   wrong-transpose identity error                   --   1.419222e+01             --

Comparison summary
------------------

.. list-table:: Default deterministic run
   :header-rows: 1
   :widths: 38 20 20 20

   * - Quantity
     - Reference
     - SpaceCore
     - Difference
   * - Objective value
     - ``2.540443e-01``
     - ``2.540443e-01``
     - ``0.000000e+00``
   * - Relative solution error
     - ``--``
     - ``--``
     - ``1.480233e-14``
   * - First-order residual norm
     - ``4.368925e-15``
     - ``3.981618e-13``
     - ``--``
   * - Metric-adjoint identity error
     - ``--``
     - ``0.000000e+00``
     - ``--``
   * - Wrong-transpose identity error
     - ``--``
     - ``1.419222e+01``
     - ``--``

CI coverage
-----------

``tests/examples/test_weighted_tikhonov.py`` imports the example, runs the
script as a subprocess, checks the dense optimality residual, compares the
SpaceCore and dense solutions, and verifies both adjoint identity diagnostics.
