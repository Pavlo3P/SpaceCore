SpaceCore
=========

Building blocks for first-order optimization: spaces, geometry, linear operators,
functionals and proximal maps, on NumPy, JAX, Torch and CuPy.

The geometry lives in the space. Gradients, adjoints and proxes are taken in the
space's inner product, so an algorithm written once stays correct in a weighted or
operator metric, on matrices, Jordan algebras or nested trees. SpaceCore ships no
optimization loops: you write them in a few lines, or use
`spacecore-optimize <https://github.com/Pavlo3P/spacecore-optimize>`_.

Install
-------

.. code-block:: bash

   pip install spacecore            # NumPy backend
   pip install "spacecore[jax]"     # or [torch], [cupy]

Python 3.11+.

Example
-------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   X = sc.DenseCoordinateSpace((2,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0])))
   A = sc.DenseLinOp(ctx.asarray([[1.0, 2.0], [0.0, 1.0]]), X, X, ctx)
   b = ctx.asarray([1.0, 2.0])

   print(X.inner(b, b))   # 22.0: the weighted inner product 2*1 + 5*4
   print(A.rapply(b))     # [1.  2.8]: the adjoint in this geometry, not A.T @ b = [1. 4.]

   # Weighted lasso, minimize 1/2 ||A x - b||^2 + 0.1 ||x||_1, by proximal gradient.
   f = sc.SquaredL2NormFunctional(X) @ A - sc.InnerProductFunctional(A.rapply(b), X, ctx)
   g = 0.1 * sc.L1NormFunctional(X)
   x, t = X.zeros(), 0.05
   for _ in range(500):
       x = g.prox(X.axpy(-t, f.grad(x), x), t)
   print(x)               # about [-2.83  1.94]

The loop is the algorithm; the space supplies the geometry it runs in.

What is in it
-------------

* **Spaces:** dense vectors and matrices, Hermitian matrices and Jordan algebras,
  stacks, and trees of any of these; Euclidean, weighted and operator metrics.
* **Operators:** dense, sparse, diagonal, matrix-free and block operators, with
  lazy sums, scaling and composition; ``rapply`` is always the metric adjoint.
* **Functionals:** linear, quadratic and norm functionals, sums, products and
  pull-backs; gradients in the space's geometry; ``prox`` and ``conjugate_prox``;
  sets and indicators.
* **Sub-step solvers:** CG, LSQR, Lanczos, power iteration, ``expm_multiply``.
* **Backends:** NumPy, JAX, Torch, CuPy; the objects run inside ``jax.jit`` and
  ``torch.compile``.

Extending
---------

A new space, operator or functional implements check-free cores (``_add_core``,
``_inner_core``, ``_apply_core``, ``_grad_core``, ...). The public methods (``add``,
``inner``, ``apply``, ``grad``, ...) are inherited wrappers that run the checks and
call the cores, and composites call cores, so each object checks once. See
:doc:`design/extending`.

Checks
------

Every object has a ``check_level``: ``"none"``, ``"cheap"`` (shape, dtype),
``"standard"`` (membership, e.g. Hermiticity) or ``"strict"`` (+ probes). Set it per
object, or once with ``sc.set_check_level`` / ``sc.use_check_level``. Failures raise
``sc.SpaceValidationError``. Inside ``jax.jit`` / ``torch.compile`` checks are
capped by ``sc.set_traced_check_level`` (default ``"none"``). See
:doc:`design/checking_policy`.

Where to go next
----------------

* **Learn by building:** :doc:`tutorials/index`.
* **Understand the rules:** :doc:`design/index`.
* **Look up an object:** :doc:`api/index`.
* **Upgrade:** :doc:`release_notes`.
* **Contribute:** :doc:`dev/index`.

Experimental (0.x): the API may change before 1.0. Apache 2.0.

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
   design/index
   dev/index
   api/index
   release_notes
