Conversion and context guide
============================

This tutorial follows ``tutorials/5_Conversion_Policy.ipynb``. It describes
the current conversion behavior in SpaceCore.

The policy model is intentionally small:

* every object is attached to a ``Context``;
* constructors resolve one context by a fixed priority rule;
* conversion always targets the backend, dtype, and checking flag of the
  requested context;
* there are no warning/error/silent conversion policies and no
  dtype-preservation policy.

The important practical rule is: **the target context wins during explicit
conversion**.

What a context controls
-----------------------

SpaceCore is built around the chain:

.. math::

   \texttt{BackendOps} \to \texttt{Context} \to \texttt{Space} \to \texttt{LinOp}.

A ``Context`` stores three runtime choices:

* ``ops``: the backend implementation, such as ``NumpyOps``, ``JaxOps``, or
  ``TorchOps``;
* ``dtype``: the backend-normalized dtype used for arrays created or converted
  through the context;
* ``enable_checks``: whether spaces, operators, and functionals validate inputs
  and outputs.

This means spaces and operators always carry an execution policy, but that
policy is just the context itself. The old global conversion knobs have been
removed.

Context resolution order
------------------------

When a constructor needs a context, SpaceCore resolves it in this order:

1. use an explicit ``ctx=...`` argument if one was provided;
2. otherwise infer a context from inputs that carry a ``ctx`` attribute or from
   registered backend arrays;
3. otherwise use the global default context from ``spacecore.get_context()``.

If the explicit context is a backend name or backend family, missing fields are
filled from backend defaults. For example, ``ctx="jax"`` selects JAX and lets
``JaxOps.sanitize_dtype(None)`` choose the dtype.

Inference is conservative. A common context can be inferred only when all
context-carrying inputs use the same backend family. If compatible inputs have
different dtypes, SpaceCore chooses the most general dtype for that backend. If
their ``enable_checks`` flags differ, the inferred flag is the conjunction:
checks stay enabled only if all source contexts have checks enabled.

Explicit conversion
-------------------

Calling

.. code-block:: python

   obj2 = obj.convert(new_ctx)

normalizes ``new_ctx`` into a full ``Context``. If ``obj.ctx == new_ctx``,
conversion returns ``obj`` unchanged. Otherwise, the object is rebuilt in
``new_ctx`` by its ``_convert(...)`` implementation.

There is no separate conversion policy. A backend change is allowed, and no
warning/error/silent mode is consulted. Use explicit contexts in library code
when backend migration should be controlled tightly.

There is also no dtype-preservation policy. The converted object uses the dtype
of the target context. If you want to preserve a dtype, make that dtype part of
the target context explicitly:

.. code-block:: python

   jax64 = spacecore.Context(spacecore.JaxOps(), dtype="float64", enable_checks=False)
   obj_jax64 = obj.convert(jax64)

Dtype behavior
--------------

Dtype behavior now has two simple cases.

During context inference
~~~~~~~~~~~~~~~~~~~~~~~~

When a context is inferred from several compatible inputs, dtypes are normalized
to the inferred backend and joined:

.. math::

   \{d_1,\dots,d_k\}
   \mapsto
   \begin{cases}
   d_1, & \text{if all } d_i \text{ agree},\\
   \texttt{result\_type}(d_1,\dots,d_k), & \text{otherwise}.
   \end{cases}

During explicit conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

The target context dtype wins. Objects are converted to that dtype. There is no
``keep_native`` mode.

Backend defaults still come from ``BackendOps.sanitize_dtype(None)``. In the
current implementation, NumPy defaults to ``float64``; JAX follows the active
JAX dtype configuration; Torch follows the active PyTorch default dtype rules.

Checking behavior
-----------------

``enable_checks`` is a context field, not a conversion policy.

* Explicit contexts use their own ``enable_checks`` value.
* Inferred contexts enable checks only if all inferred source contexts have
  checks enabled.
* Converted objects use the target context's ``enable_checks`` value.

Checks validate shapes, membership, dtype/backend representation, and operator
input/output contracts. Disabling checks can reduce overhead in tight loops
after data has already been validated.

What was removed
----------------

Older versions exposed speculative global policies such as:

* ``set_resolution_policy(...)`` / ``get_resolution_policy(...)``;
* ``set_dtype_resolution_policy(...)`` / ``get_dtype_resolution_policy(...)``;
* ``ContextPolicy`` and ``DtypePreservePolicy`` values such as ``warning``,
  ``error``, ``silent``, and ``keep_native``.

These are no longer part of the current API. Conversion is deterministic
without them: normalize the target context, rebuild if needed, and cast to the
target context dtype.

Backend-specific dtype defaults
-------------------------------

Each backend implementation defines dtype normalization through
``sanitize_dtype(dtype)``. This method normalizes dtype into backend-native form
and determines the backend default dtype when ``dtype=None``.

* ``NumpyOps.sanitize_dtype(None)`` currently returns ``numpy.float64``.
* ``JaxOps.sanitize_dtype(None)`` depends on JAX configuration: ``float64``
  when ``jax_enable_x64=True``, otherwise ``float32``.
* ``TorchOps.sanitize_dtype(None)`` follows PyTorch default dtype behavior.

If exact dtype matters, pass it explicitly in the ``Context`` you construct.

Summary
-------

When a new object is created, SpaceCore resolves a context by priority:

1. explicit ``ctx``;
2. compatible inferred contexts from inputs;
3. global default context.

When an existing object is converted, the requested target context wins
completely: backend, dtype, and ``enable_checks`` all come from that context.
This is the current conversion policy.
