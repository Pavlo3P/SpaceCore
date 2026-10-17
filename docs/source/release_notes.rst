Release notes
=============

Version 0.5.0
-------------

Functionals are real-valued: the codomain is always ``RealField`` and the
``cod`` and ``real_linear`` arguments are gone. With checks on (``cheap`` and
above), complex outputs, coefficients and offsets raise. Complex domains remain
supported through the real inner-product pairing. This is a breaking change.

SpaceCore 0.5.0 adds the first-order proximal surface, makes every public
operation a checked wrapper over a check-free core, and removes layers that did
not earn their keep, while keeping the Backend → Space → Mapping (``LinOp`` /
``Functional``) → Element division. The full list is in ``CHANGELOG.md``.

Migrating
~~~~~~~~~

* A custom space, operator or functional implements cores (``_add_core``,
  ``_flatten_core``, ``_star_core``, ``_jordan_core``, ``_apply_core``,
  ``_grad_core``, ...). A public method overridden without a decorator still
  works: it becomes the class's core.
* ``cg``, ``lsqr``, ``power_iteration``, ``lanczos_smallest`` and
  ``expm_multiply`` return one ``LinalgResult``; ``eigenvector`` and ``result``
  are ``x``, ``krylov_dim`` is ``num_iters``, ``residual_estimate`` is
  ``residual_norm``.
* Scaling a space over the reals by a non-real number raises
  ``SpaceValidationError``.
* Inside ``jax.jit`` / ``torch.compile`` checks are off by default;
  ``set_traced_check_level("cheap")`` keeps shape and dtype checks.

Added
~~~~~

* Optional ``Functional.prox`` rules, Hilbert-space ``conjugate_prox``, and the
  ``Set`` / ``NonnegativeOrthant`` / ``IndicatorFunctional`` projection surface
  (:doc:`design/proximal_contract`).
* ``Element``, a formal context-bound member of a space; ``Mapping``, the common
  parent of ``LinOp`` and ``Functional``; ``RealField`` as the codomain
  of functionals.
* A field declaration for extensions: a class lists its constructor ``_fields``
  once and gets pytree flattening, ``convert`` and (for maps) equality from it
  (:doc:`design/extending`).
* ``set_traced_check_level`` / ``get_traced_check_level``: the check-level cap
  while a backend traces, ``"none"`` by default.
* Hashable spaces, operators and functionals (consistent with ``==``) and a
  notebook ``_repr_html_``.

Changed
~~~~~~~

* ``LinOp``, ``Functional`` and every space capability own their checked public
  methods; subclasses implement the check-free cores (``_apply_core``,
  ``_grad_core``, ``_riesz_core``, ``_flatten_core``, ``_star_core``,
  ``_jordan_core``, ...). Composites call cores, so each checks once.
* Spaces over the reals reject non-real multipliers in ``scale``, ``axpy`` and
  ``scale_batch`` (breaking).
* A one-column or one-row ``BlockMatrixLinOp`` has a plain endpoint rather than a
  one-leaf ``TreeSpace``, and the containers of its rows give the endpoint trees.
* Coordinate functionals that compared by identity now compare by their fields.
* The pytree ``aux`` of field-declared classes is a tuple of ``(field, value)``
  pairs.

Removed
~~~~~~~

* ``StackedLinOp`` and ``SumToSingleLinOp`` (breaking): use
  ``BlockMatrixLinOp.stack(parts)`` and ``BlockMatrixLinOp.from_operators(parts)``.
* ``spacecore.opfamily`` and the ``Functional * LinOp`` rule (breaking): it had no
  consumer; ``F * A`` now raises ``TypeError``.
* The ``spacecore.kernels`` package: the ADR-016 dispatch specs (mostly slower on
  eager NumPy, never used under ``jax.jit``) and the core-function layer, whose
  cores are now methods of their classes.
* ``spacecore.optimize`` (breaking): the optimizer adapters live in the separate
  ``spacecore-optimize`` package.
* The legacy ``BlockDiagonalLinOp(dom, cod, parts, ctx)`` form, the
  ``enable_checks`` shim, ``ShiftedFunctional`` and other deprecated surfaces
  (breaking).
* Notebooks, tutorial notebooks, examples and the benchmark suite left the
  package repository; the rendered tutorials stay in these docs.

Version 0.4.3
-------------

Released 2026-09-12. SpaceCore 0.4.3 moves validation policy off the context and
onto the bound object, makes pytree registration backend-neutral, and extends the
functional algebra with products, constants, and the operator-family type that
``Functional * LinOp`` needs.

Removed
~~~~~~~

* ``jax_pytree_class`` is removed from the public API (breaking). Pytree
  registration is now expressed through the backend-neutral ``PyTreeNode``
  mixin, so a JAX-named decorator no longer sits in a backend-agnostic surface.
* ``SpectralLpNormFunctional`` is removed (breaking). ``spectralize`` lifts any
  coordinate functional to the spectrum of a Jordan-algebra space, which
  subsumes the per-formula spectral class.

Added
~~~~~

* Backend-neutral pytree registration. ``PyTreeNode`` carries the capability,
  and ``TorchOps.install_pytree_protocol`` makes Torch a transform-capable
  backend alongside JAX.
* ``SpectralFunctional``, ``spectralize``, and ``eigenvalue_space`` lift any
  coordinate functional onto a spectrum; ``RealifiedFunctional`` and ``realify``
  view a complex-domain functional over its real coordinates, so optimizers that
  assume a real vector space can consume it without hand-written Wirtinger
  bookkeeping.
* ``ProductFunctional`` / ``make_functional_product`` and ``ConstantFunctional``
  / ``make_constant_functional`` complete the functional algebra with a
  pointwise product carrying the product-rule Riesz gradient, and with the
  embedding of a scalar as a functional.
* ``spacecore.opfamily``: ``OperatorFamily`` and ``FunctionalScaledOperator``
  model ``F * A``, the functional-weighted map ``m(x) = F(x) A x``. That map is
  not linear, so it is deliberately not a ``LinOp``; ``m.at(x)`` freezes a member
  and ``m.linearize_at(x)`` gives the derivative for Newton-type steps.
* ``Space.scalar_field`` / ``declared_scalar_field`` / ``check_scalar`` declare
  the field a space is closed under rather than inferring it from the dtype, and
  ``checked_method`` gains ``out_scalar`` / ``out_batched_scalar`` so a
  functional's scalar codomain is checked by the same decorator that guards
  every ``LinOp`` output.
* ``OpsRegistry`` extracts the backend registry out of ``Contextual``, and
  ``BackendOps.complex_dtype`` completes the pair with ``real_dtype``.

Changed
~~~~~~~

* ``check_level`` is a property of the bound object, not of ``Context``
  (breaking). ``Context(ops, dtype=..., check_level=...)`` and the deprecated
  ``enable_checks=`` argument are gone -- a ``Context`` is now exactly
  ``(ops, dtype)``. Pass ``check_level=`` to a space, operator, or functional, or
  move the ambient default with ``set_check_level`` / ``use_check_level``. See
  :doc:`design/checking_policy`.
* ``spacecore._contextual`` is now the public ``spacecore.contextual``
  (breaking).
* Ambient context and check level are scoped with ``contextvars``, so an
  override is visible to the current thread and async task rather than to the
  whole process.
* ``Functional.__mul__`` / ``__rmul__`` dispatch on the operand type, and
  functional outputs are checked as scalars at ``standard`` and above.
* ``available_ops()`` is memoized and returns a tuple.

Fixed
~~~~~

* ``ComposedFunctional`` now has a gradient: ``F.compose(A).grad(x)`` previously
  raised.
* A present-but-broken optional backend no longer aborts ``import spacecore``,
  and one broken backend warns once rather than once per discovery call.
* ``scalar_eq`` no longer swallows every exception; it narrows to ``TypeError``
  so a genuinely broken ``__eq__`` propagates.

Version 0.4.2
-------------

Released 2026-07-01. SpaceCore 0.4.2 adds the Jordan spectral primitives, a lazy
functional algebra, and a compiled convergence-aware optimizer driver.

Added
~~~~~

* Jordan spectral primitives on every Jordan-algebra space: ``trace``,
  ``determinant``, and ``unit``, with ``TraceFunctional`` over them.
* ``Functional.value_and_grad`` evaluates a functional and its Riesz gradient in
  one pass.
* Structure-preserving tree spectra, so a ``TreeSpace`` spectral decomposition
  keeps the tree shape.
* ``minimize_optax``, a compiled convergence-aware driver for the Optax
  optimizers.

Version 0.4.1
-------------

Released 2026-06-28. SpaceCore 0.4.1 lands the optimized-kernel **dispatch**
layer (off by default), explicit lazy-operator **fusion**, a materialized-form
**cache** for the batched folds, and a unified **benchmark submodule** — all
without changing any default-path result.

Added
~~~~~

* ADR-016 optimized-kernel **dispatch** (off by default). ``KernelSpec`` gains
  ``dispatch_key`` / ``priority`` / a shape-only ``KernelCost``; the registry
  indexes dispatch-eligible (``rtol == atol == 0``) specs by key, and a single
  ``spacecore.kernels.dispatch`` entry point routes the first applicable,
  affordable spec or runs the inline ``generic`` fallback. ``dispatch_mode``
  (``off`` / ``on`` / ``verify``) is settable process-globally and per-scope;
  the materializing fast path is gated by a memory budget from
  ``BackendOps.free_memory_bytes()``. With dispatch off, every wired call site is
  result-identical to its prior inline path.
* Eight dispatch-eligible kernels (exact, ``rtol = atol = 0``): composed
  zero-annihilation and identity-elision, and the uniform flat-dense
  block-diagonal / stacked / sum-to-single batched-``matmul`` folds across the
  apply / adjoint / batched directions. Each carries a correctness reference and
  a ``python -m bench`` probe; all route only under
  ``dispatch_mode("on"/"verify")``.
* ADR-021 lazy-operator-algebra ``LinOp.fuse(*, materialize=False)``: collapses
  each maximal dense subtree into one ``DenseLinOp`` (composition becomes a matrix
  product, with scalar / sum / adjoint / block folds), mathematically equal up to
  rounding and adjoint-consistent on any geometry. A matrix-free operand is never
  densified unless ``materialize=True``.
* ADR-022 materialized-form cache for the uniform batched folds: the
  input-independent stacked block-matrix array is built once and reused, memoized
  per accessor on ``CachedStackParts`` (excluded from operator identity and the
  pytree; matrix-free never cached; the dispatcher stays stateless).
* ``NumpyOps.free_memory_bytes()`` via ``psutil`` (now a required core
  dependency), so the dispatcher's memory gate can size materializing fast paths
  on the CPU backend.
* Unified ``bench`` benchmark submodule (``python -m bench``): factory probes
  over ``space`` / ``linop`` / ``functional`` measuring per-call overhead against
  a hand-optimal pure-array-library bare, a dispatch/cache regime axis, uniform-
  and-ragged block probes, all backends benchmarked in float64, JAX run
  jitted-vs-jitted (compilation excluded and reported separately), and a
  filter-reactive interactive HTML dashboard. Tooling only — ``spacecore`` never
  imports ``bench``.

Fixed
~~~~~

* Corrected stale ``correctness_ref`` node ids on two shipped ``KernelSpec``
  objects (``composed-chain-apply``, ``block-diagonal-dense-apply``) that named
  non-existent module-level test nodes instead of the real ``Class::method`` ids.

Version 0.4.0
-------------

Released 2026-06-24. SpaceCore 0.4.0 stabilizes the typed linear-algebra core as
a validated algebra of structured mathematical objects: a public check-policy,
the ADR-015 Stage 1 dtype/scalar-field contract, the ``TreeSpace`` direct-product
abstraction with block-structured operators, an everyday functional and proximal
toolbox, external optimizer adapters, reusable test generators, and a backend
conformance matrix with deviation catalog.

Added
~~~~~

* ADR-018 external optimizer adapters in the new ``spacecore.optimize``
  subpackage. ``minimize_scipy`` and ``line_search_scipy`` drive
  ``scipy.optimize``; ``minimize_optax`` runs the canonical optax update loop
  with pytree pass-through. Each adapter evaluates ``F.value`` and converts the
  metric (Riesz) gradient ``F.grad`` to the coordinate gradient external
  optimizers expect with ``X.riesz(F.grad(x))`` — the identity on a Euclidean
  space and mandatory on a weighted one. The external optimizer owns iteration,
  line search, and convergence. ``minimize_optax`` requires a JAX-backed domain
  and the optional ``optax`` extra. See the optimizer adapter docs (now in ``spacecore-optimize``).
* ADR-019 everyday functional and proximal toolbox, in the
  ``spacecore.functional.tools`` subpackage. ``least_squares(A, b, *,
  weights=None, scale=0.5)`` returns a ``LinOpQuadraticForm`` for the (optionally
  weighted) least-squares objective. New battery functionals
  ``SquaredL2NormFunctional``, ``LpNormFunctional``, ``L1NormFunctional``,
  ``NegativeEntropyFunctional``, ``KLDivergenceFunctional``, and
  ``HuberFunctional`` are coordinate objectives whose gradients are the metric
  (Riesz) gradient under the domain geometry. ``SpectralLpNormFunctional`` (with
  the ``NuclearNormFunctional`` wrapper) is the spectral analogue -- the
  Schatten-``p`` norm of a Jordan spectrum (e.g. ``HermitianSpace`` eigenvalues),
  with the spectral function gradient reconstructed through ``from_spectrum``. The
  closed-form ``generalized_shrinkage`` proximal primitive (with ``prox_l1``,
  ``prox_l2sq``, and ``project_nonneg``) folds the metric into the threshold and
  raises on a non-diagonal metric. See :doc:`api/functionals`.
* Public ``check_level`` policy with literal values ``"none"``, ``"cheap"``,
  ``"standard"``, and ``"strict"``, plus the ``CHECK_LEVELS`` ordering and the
  ``_checks_at_least`` dispatch used by spaces, LinOps, functionals, and
  solver preconditions. See :doc:`design/checking_policy`.
* ``Space.field`` exposing a ``Literal["real", "complex"]`` mathematical
  contract derived from the context dtype, distinct from
  ``Context.dtype`` (representation default).
* ``TreeSpace`` finite direct-product coordinate space organized by an
  ``optree`` definition, and ``TreeElement`` optional ordered-leaf binding.
* ``BlockDiagonalLinOp`` and block-matrix ``LinOp`` support on ``TreeSpace``
  domains and codomains, including metric-adjoint behavior.
* Reusable test generators for spaces, LinOps, functionals, and linalg
  references under ``tests/generators/``; see the contributor guide at
  ``docs/dev/contributing/linop_generators.md``.
* ``spacecore.kernels`` subpackage with the optimized-kernel
  registration policy. Two demonstration kernels ship in ``0.4.0``:
  ``composed-chain-apply`` (skips per-link ``@checked_method`` on a flat
  LinOp chain) and ``block-diagonal-dense-apply`` (tight ``ops.matmul``
  loop over dense block leaves). Each kernel has a correctness reference
  in ``tests/kernels/``. No dispatch or fusion is wired in ``0.4.0``: the
  ADR-016 structural-dispatch mechanism is implemented but ships off by default
  and dormant, with no production routing. (The kernel policy page was removed with the spec layer in 0.5.0a.)

Breaking changes
~~~~~~~~~~~~~~~~

* ``ProductSpace`` was removed. ``TreeSpace`` is now the single structured
  finite direct-product abstraction.
* Tuple-style products now use ``TreeSpace.from_leaf_spaces((X1, X2, ...))``.
  Nested tuple, list, dictionary, named-tuple, and registered optree structures
  use ``TreeSpace(template, leaf_spaces)`` or ``TreeSpace.from_template``.
* ``ProductLinOp`` was renamed to ``TreeLinOp``. ``BlockDiagonalLinOp``,
  ``StackedLinOp``, and ``SumToSingleLinOp`` now validate ``TreeSpace`` domains
  and codomains directly.
* Product structure adapters and product-specific checks were removed. Tree
  structure and leaf validation are owned by ``TreeSpace``.
* Conversion (``ctx.asarray`` and construction helpers) refuses silent
  complex-to-real narrowing. Pass an explicit dtype to convert across fields.

Deprecations
~~~~~~~~~~~~

* ``enable_checks`` is deprecated in favor of ``check_level``:

  * ``enable_checks=True`` maps to ``check_level="standard"``;
  * ``enable_checks=False`` maps to ``check_level="none"``;
  * passing both keywords raises ``TypeError``.

  See :doc:`design/checking_policy` "Migration from ``enable_checks``" for
  the full mapping and examples. ``enable_checks`` continues to work in
  ``0.4.0`` but will be removed in a future release.

Stage 1 dtype and scalar-field contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``Context.dtype`` is the array representation default; ``Space.field`` is
  the mathematical real-or-complex contract derived from it.
* Capability guards (Jordan-algebra eligibility, Lanczos/exponential
  diagnostics, Hermitian guards) now consult ``Space.field`` instead of
  inspecting the precision-bearing context dtype.
* No constructor-level ``field`` override is available in ``0.4.0``;
  ADR-015 did not authorize one. Stage 2 (operand-dtype-preserving
  ``ctx.asarray``, opt-in exact dtype membership, operand-dtype solver
  workspaces) is deferred to ``0.5.0``.

Older releases
--------------

0.3.1 and earlier: :doc:`release_notes_archive`.

.. toctree::
   :hidden:

   release_notes_archive
