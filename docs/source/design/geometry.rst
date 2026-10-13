Geometry and Riesz Maps
=======================

An ``InnerProductSpace`` has an ``InnerProduct`` geometry. The geometry defines the
inner product and the Riesz maps used to convert between coordinate elements
and dual coordinates. Matrix-backed adjoints use those maps:

.. math::

   A^\sharp y = R_X^{-1} A^\dagger R_Y y.

Here ``A^\dagger`` is the Euclidean coordinate adjoint, while ``A^\sharp`` is
the true adjoint for the declared domain and codomain geometries. Expanding the
inner-product identity gives ``A^dagger R_Y = R_X A^sharp``. Trefethen & Bau,
*Numerical Linear Algebra*, Lecture 2, pp. 11-12 (orthonormal coordinates).

Matrix-Free Adjoints
--------------------

``MatrixFreeLinOp`` does not own a coordinate matrix, so it does not derive or
correct adjoints with Riesz maps. Its ``rapply`` callable is the user-supplied
implementation of the metric adjoint :math:`A^\sharp`. The adjoint view
``A.H`` delegates directly to those callables: ``A.H.apply(y)`` calls
``A.rapply(y)``, and ``A.H.rapply(x)`` calls ``A.apply(x)``.

If a matrix-free reverse callable is only a Euclidean coordinate transpose, it
can still pass construction and shape validation. It will fail the mathematical
adjoint dot-test on non-Euclidean spaces:

.. math::

   \langle A x, y\rangle_Y = \langle x, A^\sharp y\rangle_X.

Automatic application of :math:`R_X^{-1} A^\dagger R_Y` is reserved for
matrix-backed operators where SpaceCore owns the coordinate matrix.

Weighted Geometry
-----------------

The common diagonal-metric case is built in:

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   weights = ctx.asarray([2.0, 5.0, 11.0])
   X = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(weights))

   x = ctx.asarray([1.0, 2.0, 3.0])
   y = ctx.asarray([4.0, 5.0, 6.0])
   value = X.inner(x, y)        # vdot(x, weights * y)
   dual = X.riesz(x)            # weights * x
   primal = X.riesz_inverse(dual)

Pairing and Covectors
---------------------

A covector is a coordinate array of the space's shape. ``X.pair(xi, x)`` pairs it
with an element and uses no metric, so it is the same under every geometry;
``X.vpair(xi, xs)`` pairs one covector with a batch. The inner product is the
pairing of a Riesz image, ``real_inner(x, y) = pair(riesz(x), y)``.

.. math::

   \langle \xi, x\rangle = \operatorname{Re}\,\mathrm{vdot}(\xi, x),
   \qquad Df(x)[h] = \langle df(x), h\rangle.

``F.differential(x)`` is the covector ``df(x)``, and the gradient is
``X.riesz_inverse(F.differential(x))``. Metric-free methods (Frank-Wolfe steps,
mirror maps, quasi-Newton curvature pairs, Sinkhorn) need only these.

Dual Norm
---------

``X.dual_norm(xi)`` is the norm of a covector:

.. math::

   \|\xi\|_* = \sup_{\|x\|_G \le 1} \langle \xi, x\rangle
   = \sqrt{\langle \xi, R_G^{-1}\xi\rangle}.


Operator Metrics
----------------

``OperatorInnerProduct(M, M_inverse)`` takes the metric from a self-adjoint
positive-definite operator ``M`` on the Euclidean coordinates, so block,
Kronecker or matrix-free metrics need no new geometry class:

.. math::

   \langle x, y\rangle_X = \langle x, M y\rangle_E, \qquad R_X = M.

``M_inverse`` applies the inverse Riesz map and may be inexact, for example a
``MatrixFreeLinOp`` that runs ``cg`` on ``M``. Without it, operations that need
the inverse raise ``CapabilityError``. ``M`` must be self-adjoint and positive definite.
From ``standard`` checks upward a known non-self-adjoint ``M``, or one with
non-positive curvature on a probe vector, is rejected; ``strict`` also requires a
positive Lanczos estimate of its smallest eigenvalue. CG on a space with this geometry is
preconditioned CG: for ``A = M^{-1} B`` with ``B`` symmetric positive definite,
``cg(A, M^{-1} b)`` solves ``B x = b``.

.. code-block:: python

   E = sc.DenseCoordinateSpace((n,), ctx)
   M = sc.DenseLinOp(M_matrix, E, E, ctx)
   X = sc.DenseCoordinateSpace((n,), ctx,
                               geometry=sc.OperatorInnerProduct(M, M_inverse))

A Metric Lives in a Space
-------------------------

A geometry is reachable only through the space that holds it: no solver,
operator or functional takes a geometry argument. A different metric is a
different space, and operators and functionals are built on it. A generalized
eigenproblem ``K x = lambda M x`` is ``lanczos_smallest`` on the operator
``M^{-1} K`` built on the space with metric ``M``, in which it is self-adjoint.
Preconditioning is not a metric change: ``cg(A, b, preconditioner=K)`` takes an
operator ``K`` on ``A.domain`` approximating the inverse of ``A``, self-adjoint
and positive definite in ``A.domain``.

Custom Geometry
---------------

To define a custom geometry, subclass ``InnerProduct`` and implement
``inner``. For non-Euclidean geometries, also implement ``riesz`` and
``riesz_inverse`` and return ``False`` from ``is_euclidean``:

.. code-block:: python

   class MyGeometry(sc.InnerProduct):
       def inner(self, ops, x, y):
           ...

       def riesz(self, ops, x):
           ...

       def riesz_inverse(self, ops, x):
           ...

       @property
       def is_euclidean(self):
           return False

       def convert(self, ctx):
           # Convert any stored arrays into ctx.
           return self

The Riesz maps must broadcast over leading batch axes: if ``x`` has shape
``(N,) + space.shape``, ``riesz(ops, x)`` returns the batch of dual coordinates.
Batched adjoints and stacked spaces rely on it.

Matrix-backed operators refuse non-Euclidean spaces that do not provide usable
Riesz maps. This avoids silently using a Euclidean coordinate adjoint as if it
were a metric adjoint. If a geometry cannot expose Riesz maps, use
``MatrixFreeLinOp`` and provide an explicit metric adjoint instead.

Solvers and Functionals
-----------------------

Solvers such as ``cg``, ``lsqr``, ``power_iteration``, and
``lanczos_smallest`` use ``domain.inner`` and ``domain.norm`` (or
``codomain.norm`` for residuals that live in the codomain). Gradients returned
by real-valued functionals are space elements: they satisfy

.. math::

   \operatorname{Re}\langle \nabla f(x), v \rangle_X = Df(x)[v].

This contract is correct on non-Euclidean geometries when spaces supply Riesz
maps and operators use metric-aware adjoints. Beck, *First-Order Methods in
Optimization*, Remark 3.32, gives the real-coordinate formula; expanding
``Re(g^H R v)`` gives the complex real-valued case.
