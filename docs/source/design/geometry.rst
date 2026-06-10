Geometry and Riesz Maps
=======================

Every ``Space`` has an ``InnerProduct`` geometry. The geometry defines the
inner product and the Riesz maps used to convert between coordinate elements
and dual coordinates. Matrix-backed adjoints use those maps:

.. math::

   A^\sharp y = R_X^{-1} A^\dagger R_Y y.

Here ``A^\dagger`` is the Euclidean coordinate adjoint, while ``A^\sharp`` is
the true adjoint for the declared domain and codomain geometries.

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

The Riesz maps should broadcast over leading batch axes. For example, if
``x`` can have shape ``(N,) + space.shape``, then ``riesz(ops, x)`` should
return the corresponding batch of dual coordinates. This lets batched adjoints
stay on the fast path.

Matrix-backed operators refuse non-Euclidean spaces that do not provide usable
Riesz maps. This avoids silently using a Euclidean coordinate adjoint as if it
were a metric adjoint. If a geometry cannot expose Riesz maps, use
``MatrixFreeLinOp`` and provide an explicit metric adjoint instead.

Solvers and Functionals
-----------------------

Solvers such as ``cg``, ``lsqr``, ``power_iteration``, and
``lanczos_smallest`` use ``domain.inner`` and ``domain.norm`` (or
``codomain.norm`` for residuals that live in the codomain). Gradients returned
by functionals are space elements: they satisfy

.. math::

   \langle \nabla f(x), v \rangle_X = Df(x)[v].

This contract is correct on non-Euclidean geometries when spaces supply Riesz
maps and operators use metric-aware adjoints.
