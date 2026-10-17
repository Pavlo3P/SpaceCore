Geometry and Riesz Maps
=======================

A space's ``geometry`` (an ``InnerProduct``) defines its inner product and its
Riesz maps, and algorithms reach it only through the space.

* No solver, operator or functional takes a geometry argument. A different
  metric is a different space; build operators and functionals on it.
* A preconditioner is an operator, not a metric: ``cg(A, b, preconditioner=K)``
  takes ``K`` mapping ``A.domain`` to itself, approximating the inverse of ``A``,
  self-adjoint and positive definite in ``A.domain``.
* The generalized eigenproblem ``K x = lambda M x`` is ``lanczos_smallest`` on
  ``M^{-1} K`` built on the space with metric ``M``, where it is self-adjoint.
* ``cg``, ``lsqr``, ``power_iteration`` and ``lanczos_smallest`` use
  ``domain.inner``/``domain.norm`` (``codomain.norm`` for codomain residuals);
  their Hermitian and positive-definite assumptions refer to those inner
  products, not to the coordinate matrix.

Elements, covectors and gradients
---------------------------------

An element and a covector are both coordinate arrays of the space's shape;
``X.riesz`` maps an element to its covector ``R x`` and ``X.riesz_inverse`` maps
back.

==========================  ===================================================
call                        returns
==========================  ===================================================
``X.inner(x, y)``           :math:`\langle x, y\rangle_X = \langle R x, y\rangle`
``X.riesz(x)``              the covector :math:`R x`
``X.riesz_inverse(xi)``     the element :math:`R^{-1}\xi`
``X.pair(xi, x)``           :math:`\operatorname{Re}\,\mathrm{vdot}(\xi, x)`, metric-free
``X.vpair(xi, xs)``         ``pair(xi, x)`` for each ``x`` in a batch
``X.dual_norm(xi)``         :math:`\|\xi\|_* = \sup_{\|x\|_X\le 1}\langle\xi,x\rangle = \|R^{-1}\xi\|_X`
``F.grad(x)``               the Riesz gradient, an element of ``F.domain``
``F.differential(x)``       the covector :math:`df(x)`, by default ``X.riesz(F.grad(x))``
==========================  ===================================================

``X.real_inner(x, y)`` equals ``X.pair(X.riesz(x), y)``. The gradient and the
differential satisfy

.. math::

   \operatorname{Re}\langle \nabla f(x), v\rangle_X = Df(x)[v]
   = \langle df(x), v\rangle.

Beck, *First-Order Methods in Optimization*, Remark 3.32, gives the
real-coordinate formula; expanding ``Re(g^H R v)`` gives the complex case.

Metric-free methods (Frank-Wolfe steps, mirror maps, quasi-Newton curvature
pairs, Sinkhorn) need only ``pair`` and ``differential``.

Adjoints
--------

``A.H`` and ``A.rapply`` are the metric adjoint :math:`A^\sharp`,
:math:`\langle A x, y\rangle_Y = \langle x, A^\sharp y\rangle_X`; ``A.H.apply``
calls ``A.rapply`` and ``A.H.rapply`` calls ``A.apply``.

* Matrix-backed operators (``DenseLinOp``, ``SparseLinOp``, ``DiagonalLinOp``)
  compute :math:`A^\sharp = R_X^{-1} A^\dagger R_Y` from the coordinate adjoint
  :math:`A^\dagger` (Trefethen & Bau, *Numerical Linear Algebra*, Lecture 2).
  On a non-Euclidean space without Riesz maps their constructor raises
  ``TypeError``.
* ``MatrixFreeLinOp(apply, rapply, dom, cod)`` stores ``rapply`` as
  :math:`A^\sharp`. If ``rapply`` is the coordinate adjoint, pass
  ``euclidean_adjoint=True`` (or use ``MatrixFreeLinOp.from_coordinate_adjoint``)
  and it is wrapped in the Riesz maps once. Omitting the flag on a
  non-Euclidean space warns; ``euclidean_adjoint=False`` asserts ``rapply`` is
  already the metric adjoint. At ``strict`` the adjoint identity is dot-tested
  at construction.

Built-in geometries
-------------------

* ``EuclideanInnerProduct()``: the default, :math:`R = I`.
* ``WeightedInnerProduct(weights)``: :math:`R = \operatorname{diag}(w)`, with
  real, finite, positive ``weights`` broadcasting to the space shape.
* ``OperatorInnerProduct(M, M_inverse=None)``: :math:`R = M` for a self-adjoint
  positive-definite ``LinOp`` ``M`` on the Euclidean coordinate space of the same
  shape, backend and dtype. ``M_inverse`` applies :math:`R^{-1}` and may be
  inexact (for example a ``MatrixFreeLinOp`` running ``cg`` on ``M``); without it
  ``riesz_inverse`` raises ``CapabilityError``. From ``standard`` checks a known
  non-self-adjoint ``M`` or non-positive curvature on a probe vector is rejected;
  ``strict`` also requires a positive Lanczos estimate of the smallest eigenvalue
  and that ``M_inverse`` inverts ``M`` on a probe. ``cg`` on such a space is
  preconditioned CG: for ``A = M^{-1} B`` with ``B`` symmetric positive definite,
  ``cg(A, M^{-1} b)`` solves ``B x = b``.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
   x = ctx.asarray([1.0, 2.0, 3.0])
   y = ctx.asarray([4.0, 5.0, 6.0])

   W = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.WeightedInnerProduct(ctx.asarray([2.0, 5.0, 11.0])))
   print(W.inner(x, y), W.riesz(x))                   # vdot(x, w * y), w * x

   E = sc.DenseCoordinateSpace((3,), ctx)
   M_matrix = ctx.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
   M = sc.DenseLinOp(M_matrix, E, E, ctx)
   M_inverse = sc.DenseLinOp(np.linalg.inv(M_matrix), E, E, ctx)
   X = sc.DenseCoordinateSpace((3,), ctx, geometry=sc.OperatorInnerProduct(M, M_inverse))
   print(np.allclose(X.riesz_inverse(X.riesz(x)), x))

Custom geometry
---------------

Subclass ``InnerProduct`` and implement ``inner(ops, x, y)``, ``riesz(ops, x)``
and ``riesz_inverse(ops, x)`` directly. A geometry is not a checked object; the
space's public ``inner``/``riesz``/``riesz_inverse`` are the checked wrappers.

* The Riesz maps must broadcast over leading batch axes: for ``x`` of shape
  ``(N,) + space.shape``, ``riesz(ops, x)`` returns ``N`` covectors.
* ``is_euclidean`` defaults to ``False``; set it ``True`` only for the coordinate
  inner product.
* Define ``__eq__``: equal spaces need equal geometries, and the default is
  identity.
* ``convert(ctx)`` returns ``self`` by default; override it when the geometry
  stores arrays. ``validate_for(space)`` is an optional hook that raises when the
  geometry does not fit ``space``.
* Without ``riesz``/``riesz_inverse``, matrix-backed operators refuse the space;
  use ``MatrixFreeLinOp`` with an explicit metric adjoint instead.

.. code-block:: python

   class Scaled(sc.InnerProduct):
       """<x, y> = c * vdot(x, y)."""

       def __init__(self, c):
           self.c = c

       def inner(self, ops, x, y):
           return self.c * ops.vdot(x, y)

       def riesz(self, ops, x):
           return self.c * x

       def riesz_inverse(self, ops, x):
           return x / self.c

   S = sc.DenseCoordinateSpace((3,), ctx, geometry=Scaled(4.0))
   A = sc.DenseLinOp(ctx.asarray(np.arange(9.0).reshape(3, 3)), S, S, ctx)
   print(np.isclose(S.inner(A.apply(x), y), S.inner(x, A.H.apply(y))))

A geometry that stores weights, with equality, conversion and ``validate_for``, on a
custom space with its own membership check: :doc:`extending`.
