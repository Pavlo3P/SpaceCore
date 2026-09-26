Follow-up task after Phase D: remove ProductSpace and make TreeSpace the only structured product abstraction.

Goal:
Complete the Phase D migration by removing ProductSpace support entirely. After this task, SpaceCore should expose and support only `TreeSpace` for structured finite direct-product spaces.

This is intentionally breaking. Do not keep ProductSpace as a wrapper, alias, deprecated class, compatibility shim, or hidden internal abstraction.

Hard constraints:

* Use only the repository `.venv`.
* Do not use system Python.
* Do not install packages globally.
* Run commands through `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`, etc.
* Before editing TreeSpace docstrings, read `docs/dev/docstring_style.md` and apply its standards.

Required work:

1. Remove ProductSpace public support.

   * Remove `ProductSpace` from public exports.
   * Remove ProductSpace constructors, aliases, wrappers, and compatibility shims.
   * Remove ProductSpace-specific docs, tutorials, examples, and migration notes unless they are rewritten to refer to `TreeSpace`.
   * Remove ProductSpace-specific tests or rewrite them as TreeSpace tests.

2. Remove ProductSpace implementation.

   * Delete or empty obsolete ProductSpace implementation modules if they are no longer needed.
   * If any code still imports ProductSpace, replace it with TreeSpace.
   * Do not leave a hidden ProductSpace implementation behind.

3. Migrate internal users to TreeSpace.

   * Search the full repo for `ProductSpace`, `product space`, `product-space`, and product-specific import paths.
   * Replace all internal structured-product logic with `TreeSpace`.
   * Pay special attention to:

     * space capability logic;
     * checks;
     * batching;
     * LinOp domain/codomain validation;
     * block/product LinOp modules;
     * functionals;
     * docs;
     * tests;
     * examples.

4. Decide what to do with product LinOp modules.

   * If `linop/product/*` is still mathematically valid over TreeSpace, rename or migrate it to TreeSpace terminology.
   * If it is not ready, remove public exposure and record a narrow follow-up.
   * Do not leave public APIs that mention ProductSpace.

5. Update TreeSpace docs and docstrings.

   * Read `docs/dev/docstring_style.md` first.
   * Audit every new or modified TreeSpace docstring against that standard.
   * Ensure docstrings clearly distinguish:

     * finite direct product;
     * tree-structured representation;
     * not a tensor product;
     * leafwise operations;
     * check-level behavior;
     * field/dtype behavior inherited from leaves.

6. Update documentation.

   * Replace ProductSpace documentation with TreeSpace documentation.
   * Update examples to use TreeSpace only.
   * Add a breaking-change note:

     * ProductSpace was removed.
     * TreeSpace is now the single structured product abstraction.
     * Tuple-style products should be represented as a TreeSpace with tuple structure.

7. Update tests.

   * Remove tests asserting ProductSpace compatibility.
   * Add TreeSpace-only tests covering the same behavior:

     * tuple tree;
     * nested tuple/list/dict tree if supported;
     * leaf validation;
     * leaf path error messages;
     * zero/add/scale;
     * batch operations if supported;
     * inner/norm;
     * conversion;
     * JAX pytree behavior where available;
     * LinOp/domain/codomain interaction.

8. Update imports and packaging.

   * Ensure `from spacecore import ProductSpace` fails because the symbol no longer exists.
   * Ensure `from spacecore import TreeSpace` works.
   * Ensure generated docs do not reference removed ProductSpace APIs except in changelog or migration notes.

9. Run validation using `.venv` only.
   Required commands:

   * `.venv/bin/python -m pytest tests/ -x -q`
   * `.venv/bin/python -m ruff check .`
   * If docs build tooling is configured, run it through `.venv` as well.

10. Record any deferrals.

* If any ProductSpace-dependent feature cannot be migrated safely, remove it from public API and record a narrow follow-up in `docs/dev/current.md`.
* The follow-up must name the exact module, missing TreeSpace capability, and required tests.

Acceptance criteria:

* No public `ProductSpace` symbol remains.
* No internal implementation depends on ProductSpace.
* No tests rely on ProductSpace.
* TreeSpace is the only structured finite direct-product abstraction.
* TreeSpace docstrings comply with `docs/dev/docstring_style.md`.
* Docs and examples use TreeSpace terminology only.
* `.venv/bin/python -m pytest tests/ -x -q` passes.
* `.venv/bin/python -m ruff check .` passes.
* Any remaining TreeSpace/product-operator gap is recorded as a narrow follow-up, not hidden behind ProductSpace compatibility.
