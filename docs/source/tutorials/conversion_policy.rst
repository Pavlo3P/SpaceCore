Conversion policy guide
=======================

This tutorial follows ``tutorials/5_Conversion_Policy.ipynb``. It explains why
SpaceCore needs conversion policy, how context is chosen, how dtype is handled
during conversion, and which flags regulate conversion behavior.

Motivation
----------

SpaceCore is built around the chain:

.. math::

   \texttt{BackendOps} \to \texttt{Context} \to \texttt{Space} \to \texttt{LinOp}.

Spaces and operators always live under a numerical policy: a backend, a dtype,
and a checking policy.

As soon as objects may come from different sources, SpaceCore must answer:

* Which backend should be used?
* Which dtype should be used?
* Should native dtype be preserved or converted to the target context dtype?
* Should backend changes happen silently or produce warnings/errors?

Without an explicit conversion policy, these situations are ambiguous.

Typical example
---------------

Suppose you create ``ProductSpace((s1, ..., sn))`` without passing ``ctx=...``.
The global default context is NumPy with ``float64``, but one input space
``sk`` has a JAX context.

SpaceCore must decide whether to infer context from the input spaces or use the
global default context and convert the input spaces. It must also decide what to
do with the dtype associated with each input. Conversion policy makes these
decisions deterministic.

Context resolution
------------------

The context resolution procedure is:

.. image:: ../_static/img/context_decision_tree.svg
   :alt: SpaceCore context resolution decision tree
   :align: center

1. If an explicit context is provided via ``ctx``, it is used. If it is given as
   a string, missing context parameters are filled from defaults.
2. If no explicit context is provided and input objects carry ``ctx``
   attributes, SpaceCore attempts to infer a context.
3. A common context can be inferred only if all context-carrying inputs use the
   same backend.
4. The inferred context uses the shared backend and the most general dtype among
   the inputs. Other parameters are set to default values.
5. If no context can be inferred, the global default context set by
   ``spacecore.set_context()`` is used.

Once the context is resolved, it is assigned to the object being created. Any
context-carrying inputs are adapted to the backend of the resolved context.
Their dtype may or may not be preserved, depending on the active dtype policy.

How object conversion happens
-----------------------------

Assume an object ``foo`` needs to be converted to a new context ``new_ctx``.

.. image:: ../_static/img/convert_to_new_ctx.svg
   :alt: SpaceCore conversion to a new context
   :align: center

The backend is determined by ``new_ctx``. Dtype is treated independently by
``dtype_resolution_policy``.

Dtype resolution during conversion
----------------------------------

Dtype resolution is separate from context resolution. Context resolution decides
which context is assigned to the object being created. Once that context is
fixed, input objects carrying their own contexts may need to be converted to it.
During conversion, the backend is determined by the resolved context, but dtype
policy may vary.

If an input object has dtype ``float32`` and is converted to a context on a
different backend, SpaceCore must decide whether to preserve ``float32`` or
cast the object to the dtype of the resolved context.

The resolver supports two dtype policies:

.. dropdown:: ``dtype_resolution_policy`` values

   * ``convert``: convert the object with the dtype that the resolved context
     provides.
   * ``keep_native``: convert the object to an equivalent dtype in the backend
     of the resolved context. This is the default.

Set this policy with ``spacecore.set_dtype_resolution_policy()`` and inspect it
with ``spacecore.get_dtype_resolution_policy()``.

When context is inferred from several objects, dtype is chosen as:

.. math::

   \{d_1,\dots,d_k\}
   \mapsto
   \begin{cases}
   d_1, & \text{if all } d_i \text{ are equal},\\
   \texttt{join}(d_1,\dots,d_k), & \text{otherwise}.
   \end{cases}

Before joining, dtypes are normalized to the inferred backend.

Other inferred context parameters
---------------------------------

When a context is inferred from several source objects, the inferred
``enable_checks`` flag is the conjunction

.. math::

   \texttt{enable\_checks}
   =
   \bigwedge_i \texttt{ctx}_i.\texttt{enable\_checks}.

Checks remain enabled only if all source contexts have checks enabled.

Resolution policy
-----------------

``resolution_policy`` regulates what happens when an object's native context and
target context are backend-incompatible. That means the object is being
converted to a backend other than its own.

.. dropdown:: ``resolution_policy`` values

   * ``warning``: conversion is allowed, but a warning is issued. This is the
     default behavior.
   * ``error``: conversion is rejected. Use this when accidental backend
     migration should be forbidden.
   * ``silent``: conversion proceeds without warning. Use this when automatic
     conversion is expected and the pipeline is trusted.

Set this policy with ``spacecore.set_resolution_policy()`` and inspect it with
``spacecore.get_resolution_policy()``.

Backend-specific default dtypes
-------------------------------

Each backend implementation defines dtype normalization through
``sanitize_dtype(dtype)``. This method normalizes dtype into backend-native form
and determines the backend default dtype when ``dtype=None``.

For ``NumpyOps``:

.. math::

   \texttt{sanitize\_dtype(None)} = \texttt{numpy.float64}.

For ``JaxOps``, the default depends on JAX configuration:

* if ``jax_enable_x64=True``, the default is ``float64``;
* otherwise, the default is ``float32``.

Summary
-------

When a new object is created, SpaceCore resolves context first:

1. use explicit ``ctx`` if provided;
2. otherwise infer from inputs that carry ``ctx``;
3. otherwise use the global default context.

Context inference is possible only when context-carrying inputs agree on the
same backend. The inferred dtype is the most general dtype among inputs, and
``enable_checks`` is inferred by conjunction. Once context is resolved, backend
conversion follows ``resolution_policy`` and dtype handling follows
``dtype_resolution_policy``.
