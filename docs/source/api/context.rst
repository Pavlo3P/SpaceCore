Context API
===========

Context objects and helpers control backend, dtype, and validation behavior.

.. autosummary::
   :nosignatures:

   spacecore.backend.Context
   spacecore.set_context
   spacecore.get_context
   spacecore.register_ops
   spacecore.set_resolution_policy
   spacecore.set_dtype_resolution_policy

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
.. autofunction:: spacecore.register_ops
.. autofunction:: spacecore.set_resolution_policy
.. autofunction:: spacecore.get_resolution_policy
.. autofunction:: spacecore.set_dtype_resolution_policy
.. autofunction:: spacecore.get_dtype_resolution_policy
