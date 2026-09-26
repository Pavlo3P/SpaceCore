"""Check-free operator cores for SpaceCore.

:mod:`spacecore.kernels.core` holds the check-free cores of an operator's apply
(or a functional's evaluation): the body that runs once the public method has
validated its boundary. Operators bind them as class attributes.
"""
