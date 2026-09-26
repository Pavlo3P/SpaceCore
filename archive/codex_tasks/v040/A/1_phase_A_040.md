Implement SpaceCore 0.4.0 Phase A: release setup and inventory.

Scope:

* Create or update a `0.4.0` tracking milestone/issue plan if GitHub tooling is available.
* Add a concise inventory document, preferably `docs/dev/0.4.0-phase-a-inventory.md`.
* Do not implement Phase B–M changes yet.

Required content:

1. List all planned `0.4.0` issues from the task list and group them by phase.
2. Review ADR-006 batching and record concrete test implications.
3. Review ADR-014 check policy and confirm implementation tasks.
4. Review ADR-015 dtype/field proposal and separate Stage 1 from optional Stage 2.
5. Inventory reusable test helpers.
6. Inventory missing backend-conformance cases.
7. Inventory duplicated `ProductSpace` code paths that may migrate into `TreeSpace`.
8. Inventory all public `LinOp` families and map each to future test requirements.

Small repo-graph mark:

* Use `graph.json` as an input guide for the inventory. Use it to locate relevant files, classes, functions, and communities, but verify every conclusion against the actual repository files.

Acceptance criteria:

* `docs/dev/0.4.0-phase-a-inventory.md` exists.
* Each A1–A8 item has a clear status and evidence.
* No behavioral API changes are made.
* `ruff check .` and existing tests still pass, or failures are documented as pre-existing.
