Vision
======

SpaceCore is a **toolbox for first-order optimization**: spaces, geometry, operators,
functionals and their proxes. It gives the pieces; the algorithms — gradient methods,
PDHG, ADMM — are assembled from them in a separate library. Each method is written once
there, and it does not change when the metric, the space or the element representation
does.

The goal is **prototyping**: an algorithm written with SpaceCore should be easier and
more intuitive to write than without it, read like the mathematics, and run **as fast
as the same algorithm written directly on arrays**. An abstraction that fails any of
the three does not ship.

A tool earns its place in the box only if that library needs it. **Status:**
direction set 2026-09-28; proximal milestone implemented on ``0.5.0a``.

The idea
--------

An algorithm touches geometry in two places. The gradient is the Riesz representer of
the derivative, and the prox penalises distance in the same metric:

.. math::

   \nabla_X f(x) = R_X^{-1} Df(x),
   \qquad
   \operatorname{prox}^{X}_{\tau f}(x)
   = \operatorname*{arg\,min}_{y}\; f(y) + \tfrac{1}{2\tau}\lVert y - x \rVert_X^2 .

SpaceCore supplies these operations where the functional and space support
them; unsupported operations raise ``CapabilityError``. Change :math:`R_X`
and a supported downstream algorithm changes without its code changing.

Rules
-----

1. Elements are reached only through the space: ``zeros``, ``add``, ``scale``,
   ``inner``, ``riesz``. Nothing an algorithm needs may require ``flatten``.
2. Geometry lives on the space. ``A.H``, ``grad`` and ``prox`` are metric by default;
   nobody passes a metric around.
3. Missing capability → ``CapabilityError`` (ADR-005). Never densify to fake it.
4. Proxes are closed-form rules, not a general solver.
5. Everything an iteration calls runs at ``check_level="none"`` and compiles. "As fast
   as raw arrays" is a benchmark, not a promise: a new abstraction lands with a ``bench/``
   case comparing it to the hand-written array version.
6. No optimization loops in SpaceCore. Linear-algebra solvers (``cg``, ``lsqr``,
   ``power_iteration``, Lanczos) stay: algorithms call them for sub-steps and step sizes.

Have
----

Metric adjoints and lazy operator algebra; metric gradients (``Functional.grad``);
``power_iteration`` for operator norms; ``prox_l1`` / ``prox_l2sq`` /
``project_nonneg`` for diagonal metrics (ADR-019); ``Mapping``'s ``register`` rule
table.

Optional ``Functional.prox`` and Hilbert ``conjugate_prox``; positive scaling
and constant shifts; ``Set`` / ``NonnegativeOrthant`` / ``IndicatorFunctional``.
Native spaces need no ``geometry`` and no coordinates: functionals (including
``least_squares``) accept them, elements never densify to fake an operation, and
Lanczos/exponential action preserve native
elements. The :doc:`proximal_contract` acceptance case exercises unchanged
proximal-gradient and PDHG loops across metrics and representations.

Next
----

Smallest first; each item lands when the optimization library needs it.

1. Exercise the proximal contract in the separate optimization library.
2. Add further prox and projection rules when a downstream problem needs them.
3. Reduce measured whole-loop overhead and establish performance regression
   tracking. The initial benchmark records overhead; universal parity is unproven.

Deferred
--------

- **Bregman / mirror geometry.** The Riesz pair is already its Hilbert case,
  :math:`\nabla h = R_X` for :math:`h = \tfrac12\lVert\cdot\rVert_X^2`, so it slots in
  as a nonlinear ``riesz``. Needs a way to hold the dual iterate (entropy underflows at
  the boundary).
- **Second-order tools.** The same idea one derivative up: the Hessian as a metric
  ``LinOp`` on the space, :math:`R_X^{-1} D^2 f(x)`, applied matrix-free, so Newton and
  trust-region steps reuse ``cg`` / Lanczos, which are already here. Quasi-Newton
  updates are the other candidate. Lands when the optimization library needs it.
- **Foreign representations** (tensor networks, PETSc): realistic adapters and
  explicit approximation contracts beyond the exact native-element examples.

Not doing: optimization algorithms, a coordinate array library, a universal prox
solver, kernel dispatch without a jit-era benchmark (removed in ``f28ed29``).
``minimize_scipy`` / ``minimize_optax`` stay as thin adapters, not the start of an
optimizer layer.

Related: :doc:`geometry`, ADR-005, ADR-019, ADR-020,
``docs/dev/optimization-operation-research.md``.
