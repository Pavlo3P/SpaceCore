SpaceCore
=========

SpaceCore exists for writing numerical algorithms once, independently of the
array backend.

For example, the same algorithm can run with NumPy for debugging, CuPy for
eager GPU execution, JAX for JIT/autodiff, and Torch for tensor workflows,
while preserving the same
mathematical spaces and linear operators.

What problem does SpaceCore solve?
----------------------------------

Numerical algorithms often start as clear NumPy code and later need to move to
CuPy, JAX, Torch, or another array system. Without a backend boundary, that
migration usually leaks through the whole implementation: array constructors,
dtype handling, inner products, sparse support, and linear-operator conventions
all become backend-specific.

SpaceCore keeps those choices in a ``Context``, while algorithms work with
mathematical objects:

* a ``Space`` knows the structure and geometry of its elements;
* a ``LinOp`` maps one space to another;
* a ``Functional`` maps a space element to a scalar;
* backend-specific array creation and operations live behind ``BackendOps``.

The result is ordinary Python code whose core numerical logic is not tied to
one array library.

Mental model:

.. code-block:: text

   BackendOps -> Context -> Space/LinOp/Functional -> Algorithm

Write once, run twice
---------------------

This gradient descent loop uses only the ``Space`` and ``LinOp`` APIs. It does
not know whether the arrays are NumPy arrays, CuPy arrays, JAX arrays, or Torch
tensors.

.. code-block:: python

   import numpy as np
   import spacecore as sc


   def as_numpy(x):
       if hasattr(x, "detach"):
           return x.detach().cpu().numpy()
       return np.asarray(x)


   def make_problem(ctx):
       X = sc.DenseCoordinateSpace((3,), ctx)
       Y = sc.DenseCoordinateSpace((2,), ctx)

       A = sc.DenseLinOp(
           ctx.asarray([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]),
           dom=X,
           cod=Y,
           ctx=ctx,
       )
       x = ctx.asarray([1.0, 0.0, -1.0])
       b = ctx.asarray([0.5, 0.25])
       return X, Y, A, x, b


   def gradient_step(X, A, x, b, eta):
       r = A.apply(x) - b
       grad = A.rapply(r)
       return X.axpy(-eta, grad, x)


   def run_gradient_descent(X, A, x, b, eta, steps):
       for _ in range(steps):
           x = gradient_step(X, A, x, b, eta)
       return x

Run it with NumPy:

.. code-block:: python

   np_ctx = sc.Context(sc.NumpyOps(), dtype="float64")
   X, Y, A, x, b = make_problem(np_ctx)
   x_numpy = run_gradient_descent(X, A, x, b, eta=0.1, steps=5)
   print(as_numpy(x_numpy))

Later, run the same problem and the same ``run_gradient_descent`` with JAX:

.. code-block:: python

   import jax

   jax.config.update("jax_enable_x64", True)

   jax_ctx = sc.Context(sc.JaxOps(), dtype="float64")
   X, Y, A, x, b = make_problem(jax_ctx)
   x_jax = run_gradient_descent(X, A, x, b, eta=0.1, steps=5)
   print(as_numpy(x_jax))

   print(np.allclose(as_numpy(x_numpy), as_numpy(x_jax)))

Run it the same way with Torch:

.. code-block:: python

   torch_ctx = sc.Context(sc.TorchOps(), dtype="float64")
   X, Y, A, x, b = make_problem(torch_ctx)
   x_torch = run_gradient_descent(X, A, x, b, eta=0.1, steps=5)
   print(as_numpy(x_torch))

   print(np.allclose(as_numpy(x_numpy), as_numpy(x_torch)))

All three backends produce the same result:

.. code-block:: text

   [ 1.184125   0.3411875 -0.447625 ]
   [ 1.184125   0.3411875 -0.447625 ]
   True
   [ 1.184125   0.3411875 -0.447625 ]
   True

If you do not want to enable JAX 64-bit mode, use a supported dtype such as
``"float32"``.

What SpaceCore is not
---------------------

SpaceCore is not an optimizer and not a NumPy/JAX/Torch replacement. It
provides backend-aware spaces, operators, and context handling so you can write
your own algorithms without wiring them to one array library.

Core concepts
-------------

``Context``
~~~~~~~~~~~

A ``Context`` specifies how objects are represented:

* backend operations (``NumpyOps``, ``CuPyOps``, ``JaxOps``, ``TorchOps``, etc.);
* default dtype;
* runtime validation behavior.

Constructors resolve contexts in priority order: explicit ``ctx=...``, then
contexts inferred from inputs, then the global default context. Advanced code
that needs this resolution step directly can call
``spacecore.resolve_context_priority(...)``.

``Space``
~~~~~~~~~

A ``Space`` describes the structure and geometry of values:

* ``DenseCoordinateSpace`` for dense arrays with Euclidean, weighted, or custom inner-product geometry;
* ``ElementwiseJordanSpace`` for dense arrays with Euclidean elementwise star, Jordan, and spectral operations;
* ``HermitianSpace`` for Hermitian or symmetric matrices;
* ``ProductSpace`` for Cartesian products of spaces;
* ``StackedSpace`` for repeated copies of a leaf space.
* Batched spaces for elements such as ``X.batch((B,), (0,))``,
  representing ``B`` independent copies of ``X``.

Algorithms should use space methods such as ``zeros``, ``add``, ``scale``,
``axpy``, ``inner``, ``norm``, ``flatten``, and ``unflatten`` instead of
hard-coding backend array operations.

``LinOp``
~~~~~~~~~

A ``LinOp`` represents a linear operator between spaces:

* ``DenseLinOp`` for dense matrix or tensor operators;
* ``DiagonalLinOp`` for coordinatewise diagonal operators;
* ``SparseLinOp`` for sparse operators;
* ``MatrixFreeLinOp`` for callable-backed operators without stored matrices;
* ``IdentityLinOp`` and ``ZeroLinOp`` for canonical identity and zero maps;
* ``ScaledLinOp``, ``SumLinOp``, and ``ComposedLinOp`` for lazy operator
  algebra;
* ``BlockDiagonalLinOp`` for block-diagonal product-space operators;
* ``StackedLinOp`` for operators from one space into a product space;
* ``SumToSingleLinOp`` for operators from a product space into one space.

Operators expose ``apply`` and ``rapply``, so algorithms can use a linear map
and its adjoint without depending on the storage format.

For batched inputs, ``vapply(xs)`` and ``rvapply(ys)`` lift the operator over
the leading batch axis:

.. code-block:: python

   XB = X.batch(batch_shape=(B,), batch_axes=(0,))
   YB = Y.batch(batch_shape=(B,), batch_axes=(0,))

   ys = A.vapply(xs, batch_space=XB)    # xs in XB, ys in YB
   xs2 = A.rvapply(ys, batch_space=YB)  # ys in YB, xs2 in XB

The fallback uses backend ``vmap``; dense, sparse, diagonal, identity, zero,
algebraic, and product-structured operators provide specialized batched paths.

``Functional``
~~~~~~~~~~~~~~

A ``Functional`` represents a scalar-valued map on a space.
``LinearFunctional`` covers maps such as ``<c, x>``,
``MatrixFreeLinearFunctional`` wraps a callable without storing a representer,
and ``LinOpQuadraticForm`` represents objectives such as
``0.5 * <x, Qx> + ell(x) + a``.

For batched inputs, ``vvalue(xs)`` evaluates independently over leading batch
axes. Quadratic forms that define gradients also expose ``grad(x)`` and
``vgrad(xs)``.

Who should use this?
--------------------

SpaceCore is aimed at people writing optimization, inverse-problem, optimal
transport, semidefinite programming, or scientific ML algorithms that should not
be tied to one backend.

It is most useful when you want the mathematical model to stay stable while the
execution backend changes.

Installation
------------

Base install:

.. code-block:: bash

   pip install spacecore

With JAX support:

.. code-block:: bash

   pip install "spacecore[jax]"

With CuPy support:

.. code-block:: bash

   pip install "spacecore[cupy]"

With PyTorch support:

.. code-block:: bash

   pip install "spacecore[torch]"

* ``spacecore[jax]`` installs optional JAX support.
* GPU users should install the appropriate CUDA-enabled JAX build first,
  following the official JAX installation guide.
* ``spacecore[cupy]`` installs optional CuPy support for ``cupy.ndarray`` and
  ``cupyx.scipy.sparse`` backends.
* GPU users should install the appropriate CUDA-enabled CuPy package first,
  following the official CuPy installation guide.
* ``spacecore[torch]`` installs optional PyTorch support for ``torch.Tensor``
  backends.
* GPU users should install the appropriate CUDA-enabled PyTorch build first,
  following the official PyTorch installation guide.

For local development:

.. code-block:: bash

   python -m pip install -e ".[dev]"

Full example
------------

For a complete example of regularized optimal transport problem,
`see <https://pavlo3p.github.io/SpaceCore/tutorials/regularized_ot.html>`_ the
model is written once and solved with NumPy/JAX backends and its
`notebook <https://github.com/Pavlo3P/SpaceCore/blob/master/tutorials/6_Regularized_Opt_Transport.ipynb>`_.

Documentation
-------------

The hosted documentation is available
`here <https://pavlo3p.github.io/SpaceCore/>`_.

The documentation website is built with Sphinx from ``docs/source``.

Install the documentation dependencies:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Build the local HTML documentation:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Status
------

SpaceCore is currently experimental and under active development. The public API
may still evolve.

License
-------

Apache License 2.0

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/index
   design/index
   api/index
   release_notes
