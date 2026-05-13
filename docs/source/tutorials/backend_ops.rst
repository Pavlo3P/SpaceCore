BackendOps guide
================

This tutorial follows ``tutorials/1_BackendOps.ipynb``. It explains what
``BackendOps`` represents in SpaceCore, how it relates to ``Context``, and how
to use the predefined backends.

Current predefined implementations are ``NumpyOps`` and ``JaxOps``. Natural
future extensions include ``CuPyOps`` and ``TorchOps``.

Motivation
----------

SpaceCore works with mathematical objects such as spaces and linear operators,
but those objects still need a concrete numerical backend in order to be
represented and manipulated in code. This is the role of ``BackendOps``.

``BackendOps`` provides a unified numerical interface for array creation,
reshaping, linear algebra, sparse operations, and other backend-dependent
routines. Its purpose is not to replace the mathematical layer, but to provide
the computational realization of that layer.

SpaceCore separates two concerns:

* the mathematical structure of the objects being manipulated;
* the numerical backend used to store and compute with them.

The same mathematical object may be represented using NumPy arrays for eager CPU
work, JAX arrays for JIT compilation and automatic differentiation, or future
backends such as CuPy or PyTorch. Without a backend abstraction, spaces and
operators would need backend-specific branches throughout their implementations.

The design is:

.. math::

   \texttt{BackendOps} \to \texttt{Context} \to \texttt{Space} \to \texttt{LinOp}.

``BackendOps`` provides concrete numerical operations, ``Context`` selects a
backend with dtype and checking policy, ``Space`` describes geometry, and
``LinOp`` describes maps between spaces.

What BackendOps signifies
-------------------------

``BackendOps`` is the backend-agnostic numerical interface used by SpaceCore
internals. It mostly wraps NumPy-like methods, while normalizing the minimal
signatures that SpaceCore relies on.

For example, NumPy and JAX expose different optional arguments for matrix
multiplication, but SpaceCore's portable interface only needs the common core:

.. code-block:: python

   def matmul(self, a, b):
       ...

Concrete backends may accept additional backend-specific keyword arguments, but
higher-level SpaceCore objects can depend on the shared contract.

.. code-block:: python

   from spacecore.backend import BackendOps, Context, NumpyOps, JaxOps

   numpy_ops = NumpyOps()
   jax_ops = JaxOps()

   print(type(numpy_ops).__name__, numpy_ops.family, numpy_ops.allow_sparse)
   print(type(jax_ops).__name__, jax_ops.family, jax_ops.allow_sparse)

Why the abstraction is useful
-----------------------------

Without ``BackendOps``, the rest of the library would need to branch everywhere:

* use NumPy here;
* use JAX there;
* use SciPy sparse in one case;
* use JAX sparse in another case.

Instead, SpaceCore writes to one interface. Backend-specific classes implement
that interface.

This lets SpaceCore express backend-independent operations such as

.. math::

   x \mapsto \operatorname{reshape}(x),
   \qquad
   (A, x) \mapsto Ax,
   \qquad
   x \mapsto \operatorname{eigh}(x),

without hard-coding whether the actual arrays are NumPy or JAX arrays.

BackendOps versus Context
-------------------------

``BackendOps`` describes a backend family and the available numerical
operations. ``Context`` packages a backend operations object together with
runtime policy:

* backend operations object;
* dtype;
* checking policy.

In practice, users often work with ``Context`` when building spaces and
operators, while lower-level helper functions may accept an ``ops`` object.

.. code-block:: python

   import numpy as np
   from spacecore.backend import Context, NumpyOps, JaxOps

   ctx_np = Context(NumpyOps(), dtype=np.float64, enable_checks=True)
   ctx_jax = Context(JaxOps())

Predefined backends
-------------------

Use ``NumpyOps`` for standard eager CPU arrays, simple debugging, and close
interoperability with SciPy sparse matrices.

.. code-block:: python

   from spacecore.backend import NumpyOps

   ops = NumpyOps()
   x = ops.arange(6, dtype=float)
   X = ops.reshape(x, (2, 3))
   I = ops.eye(3)

Use ``JaxOps`` for the JAX execution model: JIT compilation, automatic
differentiation, accelerator execution, and JAX sparse compatibility. JAX dtype
behavior depends on local JAX configuration, especially ``jax_enable_x64``.

Writing backend-agnostic code
-----------------------------

A good pattern is to write low-level functions that accept an ``ops`` object.

.. code-block:: python

   from spacecore.backend import BackendOps, NumpyOps, JaxOps

   def gram_matrix(ops: BackendOps, A):
       """Return A^* A."""
       return ops.matmul(ops.conj(ops.transpose(A)), A)

   A_np = NumpyOps().reshape(NumpyOps().arange(6, dtype=float), (3, 2))
   G_np = gram_matrix(NumpyOps(), A_np)

   A_jax = JaxOps().reshape(JaxOps().arange(6), (3, 2))
   G_jax = gram_matrix(JaxOps(), A_jax)

The function only depends on the abstract operation

.. math::

   A \mapsto A^* A.

It does not need to know which backend stores the array.

When to use ops and when to use Context
---------------------------------------

Use ``ops`` directly when writing low-level backend-agnostic helpers or
implementing internals. Use ``Context`` when creating library objects, when you
need dtype normalization, or when you want one explicit runtime policy carried
by spaces and operators.

.. dropdown:: Backend operation categories

   * Array construction: ``asarray``, ``assparse``, ``zeros``, ``ones``,
     ``full``, ``eye``
   * Shape and dtype: ``shape``, ``ndim``, ``size``, ``get_dtype``,
     ``sanitize_dtype``
   * Linear algebra: ``vdot``, ``matmul``, ``sparse_matmul``, ``eigh``,
     ``solve``, ``svd``
   * Transformations: ``reshape``, ``transpose``, ``swapaxes``, ``concatenate``
   * Control flow and trees: ``fori_loop``, ``while_loop``, ``scan``, ``cond``

Sparse support
--------------

``BackendOps`` also carries sparse-related information:

* whether sparse arrays are supported;
* which sparse array types belong to the backend;
* how to convert to sparse form;
* how to perform sparse-dense multiplication.

This lets the same high-level abstractions work with dense and sparse operator
storage.

Practical advice
----------------

Prefer this flow:

1. Choose a backend implementation, such as ``NumpyOps`` or ``JaxOps``.
2. Wrap it in a ``Context``.
3. Build spaces and operators from that context.

Avoid mixing raw backend arrays from different families inside the same object
graph unless you explicitly convert them first.

Custom backends can be registered by subclassing ``BackendOps``:

.. code-block:: python

   import spacecore as sc


   @sc.register_ops
   class MyOps(sc.BackendOps):
       _family = "my_backend"
       _allow_sparse = False

       # Implement the abstract BackendOps surface.

Summary
-------

``BackendOps`` is the portability layer of SpaceCore. It provides one abstract
numerical interface, supports multiple concrete backends, and keeps
mathematical abstractions separate from execution details.
