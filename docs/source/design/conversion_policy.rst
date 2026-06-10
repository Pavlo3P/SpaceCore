Conversion Policy
=================

SpaceCore is built around the chain

.. math::

   \texttt{BackendOps} \to \texttt{Context} \to \texttt{Space} \to \texttt{LinOp}.

Objects that inherit from ``ContextBound`` can be rebuilt under a target
context:

.. code-block:: python

   converted = obj.convert(target_context)

The current policy is intentionally fixed: the target context wins. Conversion
uses the target backend, target dtype, and target ``enable_checks`` value. There
are no global warning/error/silent conversion knobs.

Context Resolution
------------------

When a new context-bound object is created, SpaceCore resolves its context in a
fixed order:

1. use an explicit context provided through ``ctx=...``;
2. otherwise infer a context from inputs that carry ``ctx`` or from registered
   backend arrays;
3. otherwise use the global default context from ``spacecore.get_context()``.

A common context is inferred only when all context-carrying inputs use the same
backend family. If compatible inputs have different dtypes, SpaceCore promotes
them to a common dtype for that backend. The inferred ``enable_checks`` flag is
enabled only if all inferred contexts have checks enabled.

Explicit Conversion
-------------------

Calling ``obj.convert(new_ctx)`` normalizes ``new_ctx`` into a full
``Context``. If the object already uses that context, conversion returns the
object unchanged. Otherwise, the object is rebuilt in the target context by its
implementation-specific conversion method.

Backend changes are explicit because the call site supplies the target context.
If backend migration should be restricted in library code, enforce that policy
before calling ``convert``.

Dtype Behavior
--------------

During context inference, compatible input dtypes are joined with backend
promotion rules. During explicit conversion, arrays are converted to the target
context dtype. To preserve a source dtype across backend conversion, include
that dtype in the target ``Context``.

Summary
-------

* Constructors resolve one context by priority: explicit ``ctx``, compatible
  inferred context, then the global default context.
* Explicit conversion is deterministic: the requested target context controls
  backend, dtype, and checking behavior.
* Runtime validation is controlled by ``Context.enable_checks``.
