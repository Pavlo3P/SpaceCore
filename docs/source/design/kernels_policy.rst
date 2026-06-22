Kernels and the optimization policy
===================================

SpaceCore separates the abstract algebra of spaces and linear operators
from the optimized kernels that implement specific fast paths. The
:mod:`spacecore.kernels` subpackage is where those kernels live, and
this page is the contract every contributor must satisfy when adding
one.

The policy is intentionally strict in ``0.4.0``. *Runtime* dispatch and
fusion of the benchmarked numerical kernels — inspecting operands at call
time and automatically choosing an optimized variant when ``applicable``
returns ``True`` — are gated on the ``0.6.0`` design decision. Until that
decision lands, the benchmarked kernels in this subpackage are alternative
entry points: they are tested, but no SpaceCore code path silently routes
through them.

Two kernel layers
-----------------

The subpackage hosts two complementary layers.

**Core kernels** are the check-free cores of an operator's apply (or a
functional's evaluation) — the body that runs once the public method has
validated its boundary. For LinOps these are the ``apply`` / ``rapply`` /
``vapply`` / ``rvapply`` cores; for :class:`~spacecore.functional.Functional`
objects they are the ``value`` / ``grad`` / ``vvalue`` / ``vgrad`` cores. They
live as concrete functions in the kernels subpackage: the composite LinOp algebra
in :mod:`spacecore.kernels.core.algebra`, the concrete LinOp leaves in
:mod:`spacecore.kernels.core.dense`, :mod:`spacecore.kernels.core.diagonal`, and
:mod:`spacecore.kernels.core.sparse` (each of which also owns its operator's private
computation-mode enum), and the functionals in
:mod:`spacecore.kernels.core.functional`. The binding
*rules* live in :mod:`spacecore.kernels.core`. An operator opts in by
*declaring* ``@core_kernels("dense")`` (etc.), and the decorator installs the
registered functions as the class's ``_apply_core`` / ``_rapply_core`` /
``_vapply_core`` / ``_rvapply_core`` methods. This keeps the fast-path logic out
of every operator class body — they say *which* kernel they use, not *how* it
works — while binding statically at class-definition time, so it is **not**
runtime dispatch and adds nothing per call. The base :class:`spacecore.linop.LinOp`
cores remain the generic fallback for operators that register no kernel.
``ComposedLinOp`` additionally caches a flattened ``_apply_chain`` at
construction, so a deep ``A @ B @ C @ ...`` fuses into a single loop. The leaf
kernels import the metric-adjoint helpers lazily, on the non-Euclidean Riesz path
only, so the kernels subpackage keeps no module-level dependency on
:mod:`spacecore.linop` (no import cycle). These cores *are* on the default apply
path; their correctness is pinned by the operator conformance suite together with
``tests/kernels/test_core_kernel_dispatch.py``.

**Benchmarked numerical kernels** are the heavier, opt-in fast paths described
by :class:`~spacecore.kernels.KernelSpec` and governed by the contract below.
They are not auto-selected; a clearly-scoped call site chooses them explicitly.

The rest of this page governs the *benchmarked numerical kernel* layer.

What "kernel" means here
------------------------

A kernel is a pair of callables that produce the same numerical result
on the same inputs:

* a *generic* implementation that mirrors the un-optimized SpaceCore
  path; and
* an *optimized* implementation that produces the same result faster
  (or with lower memory) on a documented subset of inputs.

A kernel is registered through a :class:`~spacecore.kernels.KernelSpec`
that ties the two implementations to a correctness reference and an
applicability predicate.

The contract
------------

Every kernel must satisfy *all* of:

1. **Correctness reference.** Each ``KernelSpec`` names a pytest node id
   under ``tests/kernels/test_kernels_match_generic.py`` that asserts
   the optimized implementation matches the generic one on every
   applicable generated case. The reference exists *before* the
   optimization lands.
2. **Generic implementation.** The reference path the optimized
   implementation must match. May reuse the existing SpaceCore code
   path; the kernel module *re-exposes* it so the test can call it
   directly without relying on a higher-level wrapper.
3. **Applicability predicate.** Returns ``True`` only when the
   optimized implementation is safe. Used by future dispatch logic;
   today it documents the safe envelope.
4. **Stable name.** Kebab-case, unique across the registry. Adding a
   kernel with a duplicate name raises at import time.

The :class:`KernelSpec` ``__post_init__`` raises
:class:`~spacecore.kernels.MissingReferenceError` if the correctness
reference is missing. The enforcement lives in
:mod:`tests.kernels.test_kernel_policy`, which iterates every registered
spec and asserts that its correctness reference resolves.

What is intentionally out of scope in ``0.4.0``
-----------------------------------------------

* **Runtime dispatch.** No code path inspects operands at call time and
  selects a benchmarked :class:`KernelSpec` ``optimized`` variant when
  ``applicable`` returns ``True``. That requires the ``0.6.0`` design
  decision and a clear interaction with the check-policy layer. (The
  *core apply kernels* above are bound statically per operator class, not
  dispatched at call time, so they are out of this gate.)
* **Numerical fusion.** No benchmarked kernel inspects adjacent operators
  to combine them into a precomputed product (``A @ B`` collapsed to one
  dense matrix, for example). Reserved for the same future window. The
  static ``ComposedLinOp`` chain flattening above is structural fusion of
  the apply *loop*, not numerical fusion of the operators.
* **Block-, Kronecker-, or tensor-product specialized kernels** beyond
  the diagonal demonstration kernel. The task list ties those to
  "correctness references exist", which is now true; an implementation
  can land in a follow-up without revisiting the policy.

The ``0.4.0`` shipped kernels
-----------------------------

``composed-chain-apply``
~~~~~~~~~~~~~~~~~~~~~~~~

Apply a sequence of operators ``(A, B, C, ...)`` to ``x`` in order
without rebuilding the nested :class:`spacecore.linop.ComposedLinOp`
tree. The optimized variant skips the per-link
``@checked_method`` wrapper that the generic
``ComposedLinOp(A, ComposedLinOp(B, C)).apply(x)`` path pays on every
intermediate.

* Generic: ``spacecore.kernels.specs.composed.composed_chain_apply_generic``.
* Optimized: ``spacecore.kernels.specs.composed.composed_chain_apply_optimized``.
* Correctness: ``tests/kernels/test_kernels_match_generic.py
  ::test_composed_chain_apply_matches_generic``.

``block-diagonal-dense-apply``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply a flat tuple of block matrices to a flat tuple of input leaves
via ``ops.matmul`` directly. The optimized variant hoists the bound
method lookup out of the loop and writes a tight zip-loop, instead of
going through the :class:`spacecore.linop.tree.BlockDiagonalLinOp`
``@checked_method`` wrapper per leaf.

* Generic: ``spacecore.kernels.specs.block_diagonal.block_diagonal_dense_apply_generic``.
* Optimized: ``spacecore.kernels.specs.block_diagonal.block_diagonal_dense_apply_optimized``.
* Correctness: ``tests/kernels/test_kernels_match_generic.py
  ::test_block_diagonal_dense_apply_matches_generic``.

Adding a new kernel
-------------------

1. Implement ``generic`` and ``optimized`` in a new module under
   ``spacecore/kernels/specs/``. The generic implementation must match the
   un-optimized path that the rest of SpaceCore would use.
2. Register a :class:`~spacecore.kernels.KernelSpec` at module scope.
   Name the correctness reference you are about to add.
3. Add the correctness test function under
   ``tests/kernels/test_kernels_match_generic.py``. Use the existing
   parametrized templates as guides.
4. Run ``pytest tests/kernels/`` — the policy-enforcement test refuses
   to pass until the correctness test exists.

Cross-references
----------------

* :doc:`backend_conformance` — the matrix the kernels
  ultimately ride on. ``ops.matmul`` semantics are pinned there.
* :doc:`checking_policy` — kernels deliberately skip the
  ``@checked_method`` wrapper. The reason it's safe inside a kernel is
  documented in each kernel's ``notes``.
* :doc:`/release_notes` — ``0.4.0`` shipped the policy and the two
  demonstration kernels. Dispatch and fusion are tracked separately.
