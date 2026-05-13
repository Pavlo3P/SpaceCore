SpaceCore documentation
=======================

SpaceCore is a lightweight Python library for backend-agnostic vector spaces
and linear operators. It is designed for numerical algorithms that should work
with structured spaces and multiple array backends without hard-coding NumPy or
JAX throughout the algorithm.

The documentation has three layers:

* :doc:`tutorials/index` explain the main ideas with examples.
* :doc:`design/index` describes the policies and design choices behind the API.
* :doc:`api/index` provides explicit object-level API reference pages.
* :doc:`release_notes` records user-visible changes by release.

Core model
----------

SpaceCore code is organized around three questions:

* Which backend executes array operations?
* Which space does each value belong to?
* Which linear map connects one space to another?

In notation, a linear operator maps one space to another:

.. math::

   A : X \to Y.

Quick example
-------------

.. code-block:: python

   import numpy as np
   import spacecore as sc

   sc.set_context("numpy", dtype="float64", enable_checks=True)

   X = sc.VectorSpace((3,))
   Y = sc.VectorSpace((2,))

   A = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
   op = sc.DenseLinOp(A, dom=X, cod=Y)

   x = X.ctx.asarray([1.0, 0.0, -1.0])
   y = op.apply(x)

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
   design/index
   api/index
   release_notes
