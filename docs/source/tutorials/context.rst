Context Tutorial
================

Goal
   Understand what a ``Context`` owns and how checks affect user input.
Prerequisites
   Basic Python and NumPy.
Estimated time
   10 minutes.
Backends used
   NumPy only; optional backend notes are conceptual.

1. Create a context
-------------------

A context packages backend operations, dtype, and validation policy.

.. code-block:: python

   import numpy as np
   import spacecore as sc

   ctx = sc.Context(sc.NumpyOps(), dtype=np.float64, check_level="standard")
   print(ctx.ops.family)
   print(ctx.dtype == np.dtype("float64"))
   print(ctx.check_level)

Expected output:

.. code-block:: text

   numpy
   standard
   True

Checkpoint: ``ctx.asarray([1]).dtype`` should be ``float64``.

2. Use the context to create arrays and spaces
----------------------------------------------

Constructors store the normalized context. Arrays created through ``ctx`` use
that context's backend and dtype.

.. code-block:: python

   X = sc.DenseCoordinateSpace((2,), ctx)
   x = ctx.asarray([1.0, 2.0])

   print(X.ctx == ctx)
   print(x.dtype == np.dtype("float64"))

Expected output:

.. code-block:: text

   True
   True

Checkpoint: ``X.check_member(x)`` should not raise.

3. See what checks catch
------------------------

SpaceCore does not silently convert operation inputs. At ``cheap`` and higher,
a wrong shape is rejected before numerical work begins.

.. code-block:: python

   try:
       X.check_member(ctx.asarray([1.0, 2.0, 3.0]))
   except Exception as exc:
       print(type(exc).__name__)

Expected output:

.. code-block:: text

   SpaceValidationError

Checkpoint: create valid inputs through the same ``ctx`` whenever possible.

What can go wrong
-----------------

.. admonition:: Assuming ``Context`` owns devices

   A context records backend operations and dtype. It does not move existing
   Torch or CuPy arrays between devices unless the backend conversion you call
   does so explicitly. Treat device placement as backend-specific behavior.

Recap
-----

* A context owns backend operations, dtype, and validation policy.
* It does not own array memory or device placement.
* Constructors store normalized contexts.
* Operation inputs are validated when checks are enabled, not silently converted.

Next steps
----------

Read :doc:`conversion_policy` for explicit conversion. See
:doc:`../design/context_ownership` for the precise policy.
