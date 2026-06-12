Conversion Tutorial
===================

Goal
   Convert spaces and operators to an explicit target context and understand
   what is rebuilt.
Prerequisites
   Complete :doc:`context` and :doc:`linops`.
Estimated time
   10 minutes.
Backends used
   NumPy only, converting dtype within the same backend.

1. Build an object in one context
---------------------------------

The source context uses NumPy float32.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   src = sc.Context(sc.NumpyOps(), dtype=np.float32, check_level="standard")
   X = sc.DenseCoordinateSpace((2,), src)
   A = sc.DiagonalLinOp(src.asarray([2.0, 3.0]), X, src)

   print(A.ctx.dtype == np.dtype("float32"))

Expected output:

.. code-block:: text

   True

Checkpoint: ``A.domain.dtype`` is also ``float32``.

2. Convert to a target context
------------------------------

Explicit conversion is target-context driven. The target dtype and check level win.

.. code-block:: python

   dst = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="none")
   B = A.convert(dst)

   print(B.ctx.dtype == np.dtype("float64"))
   print(B.ctx.check_level)
   print(B.apply(dst.asarray([1.0, 2.0])))

Expected output:

.. code-block:: text

   True
   none
   [2. 6.]

Checkpoint: ``B is A`` should be false because the context changed.

3. Know what is copied or preserved
-----------------------------------

Matrix-backed operators convert stored arrays. Spaces are reconstructed.
Matrix-free callables are preserved, so their Python functions must be valid for
the target backend.

.. code-block:: python

   print(B.domain == X.convert(dst))

Expected output:

.. code-block:: text

   True

Checkpoint: conversion changes representation policy, not the mathematical map.

What can go wrong
-----------------

.. admonition:: Optional backend missing

   ``sc.Context(sc.JaxOps(), ...)`` works only when JAX is installed and
   ``JaxOps`` is exported. Install ``spacecore[jax]`` or use a backend that is
   available in the current environment.

Recap
-----

* ``convert`` is explicit and target-context driven.
* Spaces and matrix-backed operators are rebuilt in the target context.
* Matrix-free callables are not rewritten for another backend.
* Backend-specific dtype, sparse, device, and tracing behavior remains visible.

Next steps
----------

Read :doc:`../design/conversion_policy` for the full policy and
:doc:`../design/dtype_policy` for dtype rules.
