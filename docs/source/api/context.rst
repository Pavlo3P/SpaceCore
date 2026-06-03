Context API
===========

Context objects and helpers control backend, dtype, and validation behavior.

.. autosummary::
   :nosignatures:

   spacecore.backend.Context
   spacecore.set_context
   spacecore.get_context
   spacecore.resolve_context_priority
   spacecore.register_ops

Context
-------

.. autoclass:: spacecore.backend.Context
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

Context helpers
---------------

.. autofunction:: spacecore.set_context
.. autofunction:: spacecore.get_context
.. autofunction:: spacecore.resolve_context_priority
.. autofunction:: spacecore.register_ops
