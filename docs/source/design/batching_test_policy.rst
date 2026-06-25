Batching test policy
====================

This page records what ``0.4.0`` tests about batching and what it intentionally
does not. The mathematical batching model is :doc:`batching`; this page is
about test scope.

``0.4.0`` does not redesign batching. ADR-006 is the contract; this release
stabilizes it through coverage rather than reshaping it. Any redesign is
out of scope and is tracked separately in ``docs/dev/current.md``.

In scope
--------

The following behavior is exercised by the generated test suite and by
backend conformance tests:

* ``LinOp.vapply`` and ``LinOp.rvapply`` over a leading-axis batch with shape
  ``(B,) + space.shape``, including the degenerate ``B = 1`` case.
* ``Functional.vvalue`` and ``Functional.vgrad`` where implemented, against
  per-row reference computations.
* ``StackedSpace`` as a first-class space, including arithmetic, inner
  products, capabilities, and ``check_level``-aware leaf checks.
* ``TreeSpace`` block operators on batched leaves where every leaf carries the
  same leading-axis batch shape.
* Native JAX ``vmap`` parity with the backend-agnostic loop fallback for
  operations declared as batchable in the backend conformance matrix
  (:doc:`backend_conformance`).
* Trailing-shape and dtype validation under each ``check_level``, including
  the bounded probes used by ``"strict"``.

Out of scope for ``0.4.0``
--------------------------

The following cases are intentionally not tested as part of the release gate.
A defect found here is documented in ``docs/dev/current.md`` and tracked for
a future release rather than fixed in ``0.4.0``:

* Nested batch axes (more than one leading batch dimension).
* Non-leading batch axes.
* Mixed leaf batch sizes within a single ``TreeSpace`` element.
* Cross-backend batched conformance beyond NumPy and JAX. Torch and CuPy
  batched paths are exercised opportunistically through generators but are
  not subject to the strict tolerance pins applied to NumPy and JAX.
* Batched iterative solvers (``cg``, ``lsqr``, ``lanczos_smallest``,
  ``power_iteration``, ``expm_multiply``). These entry points are unbatched
  in ``0.4.0`` and must raise a clear shape error on batched input; choosing
  between explicit batched APIs and a documented user-level loop is deferred
  to a follow-up.

Where the tests live
--------------------

* ``tests/linops/test_batched_apply.py`` and
  ``tests/linops/test_batched_lifting.py`` — leading-axis ``vapply`` /
  ``rvapply`` and structured-operator batched paths.
* ``tests/linops/test_tree_linop_batching.py`` — batched ``TreeSpace`` block
  operators.
* ``tests/spaces/test_vectorized_checks.py`` — trailing-shape and dtype
  validation under each ``check_level``.
* ``tests/spaces/test_stacked_space.py`` — ``StackedSpace`` arithmetic,
  inner products, and capability dispatch.
* ``tests/backend/test_backend_vmap.py`` — native ``vmap`` vs fallback-loop
  parity (Phase J9; see :doc:`backend_conformance`).

Generator families that produce batched cases live under ``tests/generators/``
and include ``dense_coordinate_space_cases``, ``tree_space_generated_cases``,
and the LinOp generators in ``tests/generators/linops.py``. Each generator
documents whether it produces batched, unbatched, or both variants.

Reading this policy
-------------------

This page is not a redesign proposal. It is a statement of which behavior is
under contract for ``0.4.0`` and which is allowed to drift. When the
batching design itself evolves, the new contract belongs in ADR-006 (or its
successor) and is reflected in :doc:`batching`. This page only updates when
the test scope changes.
