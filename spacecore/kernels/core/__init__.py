"""Core apply / evaluation kernels.

The check-free cores of LinOp ``apply``/``rapply``/``vapply``/``rvapply`` and
Functional ``value``/``grad``/``vvalue``/``vgrad`` live here as plain functions
(``algebra``, ``dense``, ``diagonal``, ``sparse``, ``functional``). Each operator
class binds its cores as class attributes (``_apply_core = dense_apply_core``),
so a call costs one bound-method call.
"""
