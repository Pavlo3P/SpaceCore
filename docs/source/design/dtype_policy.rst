Dtype Behavior
==============

Each ``Context`` stores a backend-normalized ``dtype``. ``Context.asarray`` and
``Context.assparse`` use that dtype unless a backend method receives an explicit
dtype argument.

Stored and inferred dtype
-------------------------

A concrete ``Context`` sanitizes its dtype through ``ops.sanitize_dtype``.
Constructors that receive several context-bearing inputs resolve one backend
family and one dtype through backend promotion. Explicit ``ctx=...`` takes
priority over inference.

Spaces use ``space.dtype`` for zeros, ones, unflattened arrays, and membership
checks. Operators use their context dtype for stored matrices and scalar helper
values.

Real and complex spaces
-----------------------

Most coordinate spaces accept real or complex floating dtypes. Geometry can add
extra restrictions. ``WeightedInnerProduct`` requires real, positive, finite
weights even in a complex coordinate space. ``EuclideanElementwiseJordanSpace``
requires a real dtype and Euclidean geometry; complex elementwise Jordan spaces
use ``ElementwiseJordanSpace`` instead.

Linear algebra assumptions
--------------------------

Solvers use the operator domain and codomain inner products. Hermitian and
positive-definite assumptions are with respect to those inner products, not just
the coordinate matrix. Complex problems require complex-compatible dtypes when
inputs or scalars are complex.

Expected errors
---------------

Invalid dtype usage usually appears as ``TypeError`` from membership checks,
backend dtype conversion, or sparse conversion. Invalid Euclidean-Jordan dtype
usage raises ``ValueError``.

Example
-------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, enable_checks=True)
   try:
       sc.EuclideanElementwiseJordanSpace((2,), sc.Context(sc.NumpyOps(), dtype=np.complex128))
   except ValueError as exc:
       print(exc)

Expected output:

.. code-block:: text

   EuclideanElementwiseJordanSpace requires a real dtype.
