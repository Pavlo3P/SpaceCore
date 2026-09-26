Implement SpaceCore 0.4.0 Phase D: TreeSpace and structured elements.

Goal:
Introduce a general tree-structured finite product space abstraction and migrate ProductSpace toward it without breaking existing ProductSpace users.

Mathematically, `TreeSpace` represents a finite product

[
X = \prod_{\ell \in L} X_\ell
]

where the leaves (X_\ell) are ordinary SpaceCore spaces and the tree structure records how the corresponding element is organized in Python.

This task implements Phase D only. Do not implement Phase E block LinOps, Phase F generators, or unrelated cleanup except where strictly needed to make TreeSpace coherent.

Inputs:

* `task_list.md`
* `GRAPH_REPORT.md`
* `graph.json`
* Existing ProductSpace implementation
* Existing tree/pytree helpers, if present

Use `GRAPH_REPORT.md` first to navigate the repo and save context. Use `graph.json` only when exact file/function locations are needed. Treat inferred graph edges as hints and verify all relationships against actual source files before editing.

Important graph-navigation hints:

* Start with `space/concrete/_product.py`.
* Inspect existing ProductSpace methods:

  * constructor and capability selection;
  * structure validation;
  * zero/ones/add/scale;
  * flatten/unflatten;
  * flatten_batch/unflatten_batch;
  * tree_flatten/tree_unflatten;
  * inner/norm behavior;
  * conversion behavior.
* Inspect `_tree.py` or equivalent tree helper module.
* Inspect backend pytree hooks, especially JAX registration.
* Inspect checks and batching paths only as needed for TreeSpace validation.

Required implementation:

1. Dependency decision.

   * Decide whether `optree` is a required dependency or optional dependency.
   * Prefer a required dependency if it substantially simplifies and stabilizes the implementation.
   * If optional, provide clear fallback errors when tree functionality needs `optree`.
   * Record the decision in docs.

2. Add `TreeSpace`.

   * Define public constructor:

     `TreeSpace(treedef, leaf_spaces, *, ctx=None, check_level=None, ...)`

   * The precise constructor spelling may be adjusted to fit existing SpaceCore conventions, but it must expose:

     * tree structure;
     * ordered leaf spaces;
     * context;
     * validation policy.

   * `TreeSpace` must preserve leaf order deterministically.

3. Add `TreeElement`.

   * Define `TreeElement(space, leaves)` or an equivalent internal/public representation.
   * It must bind an element to its TreeSpace.
   * It must be possible to recover:

     * the tree space;
     * ordered leaves;
     * reconstructed Python tree value.

4. Implement flatten/unflatten.

   * Use `optree` or the chosen tree library for structure handling.
   * Flatten structured Python values into leaves.
   * Rebuild structured Python values from leaves.
   * Validate arity and structure mismatch with clear errors.

5. Implement tree path handling.

   * Store or compute leaf paths.
   * Use paths in validation/debug error messages.
   * Error messages should identify the failing leaf path when a leaf membership check fails.

6. Implement core vector-space operations.

   * `TreeSpace.zero()` returns a tree element with each leaf equal to the corresponding leaf-space zero.
   * `TreeSpace.add(x, y)` adds leaves componentwise.
   * `TreeSpace.scale(a, x)` scales every leaf.
   * Include batch variants only if they are already required by the current Space API; otherwise explicitly document what is not yet supported.

7. Implement membership/check behavior.

   * `TreeSpace.check(x)` validates:

     * tree structure;
     * leaf count;
     * each leaf belongs to the corresponding leaf space;
     * context compatibility;
     * field/dtype behavior according to the current Phase B/C policy.
   * Respect the existing check-level model.
   * Do not silently coerce invalid leaves during checks.

8. Implement conversion.

   * `TreeSpace.convert(new_ctx)` or the repo’s equivalent conversion method converts all leaf spaces.
   * Element conversion should convert each leaf and rebuild the same structure.
   * Conversion must preserve tree structure and leaf paths.

9. Implement inner product and norm.

   * If all leaves are inner-product spaces, `TreeSpace.inner(x, y)` is the sum of leaf inner products.
   * `TreeSpace.norm(x)` must agree with the induced product norm.
   * If some leaves lack inner-product capability, the resulting TreeSpace must not falsely advertise inner-product capability.

10. Capability logic.

* Capabilities of `TreeSpace` should be the intersection of leaf capabilities when mathematically valid.
* Avoid claiming Jordan/spectral capability unless this is already well-defined leafwise and consistent with current ProductSpace behavior.
* Preserve existing ProductSpace capability semantics.

11. Migrate ProductSpace.

* ProductSpace should become a specialization or thin wrapper around TreeSpace where possible.
* Preserve the existing public ProductSpace API.
* Existing ProductSpace constructors should still work.
* Existing ProductSpace element structure should still work unless a documented migration note is unavoidable.

12. Remove duplicated ProductSpace logic.

* Move duplicated leafwise arithmetic, validation, conversion, inner, norm, and flatten/unflatten behavior into TreeSpace.
* Keep ProductSpace-specific code only for public API compatibility, tuple/named structure conventions, and capability/class selection if still needed.

13. Preserve JAX pytree behavior.

* Existing JAX pytree registration for spaces/operators must not regress.
* Add or update pytree tests if TreeSpace or ProductSpace is registered as a pytree node.

14. Documentation.

* Add TreeSpace documentation.
* Explain the mathematical meaning as a finite direct product, not a tensor product.
* Show examples:

  * tuple tree;
  * dict or named tree if supported;
  * nested tree;
  * ProductSpace compatibility.
* Explain when to use TreeSpace versus ProductSpace.

15. Tests.
    Add tests for:

* constructor validation;
* structure mismatch;
* leaf-count mismatch;
* leaf path reporting;
* zero;
* add;
* scale;
* check;
* convert;
* inner;
* norm;
* ProductSpace compatibility;
* ProductSpace API preservation;
* JAX pytree behavior where available;
* nested tree structures;
* check-level behavior if Phase B is already implemented.

Non-goals:

* Do not implement block-diagonal or block-matrix LinOps.
* Do not redesign batching.
* Do not change public ProductSpace semantics unless absolutely necessary.
* Do not use TreeSpace to represent tensor products.
* Do not introduce broad solver or backend changes.

Acceptance criteria:

* `TreeSpace` exists.
* `TreeElement` exists or an equivalent structured element representation exists.
* Tree flatten/unflatten works and is tested.
* Leaf paths are available for validation/debugging.
* TreeSpace zero/add/scale/check/convert/inner/norm work.
* ProductSpace is migrated onto TreeSpace or a narrow follow-up is recorded for incomplete migration.
* Existing ProductSpace public tests pass.
* Duplicated ProductSpace leafwise logic is removed where practical.
* TreeSpace docs exist and define it as a finite direct product.
* Tests cover structure, arithmetic, conversion, inner, norm, and ProductSpace compatibility.
* `pytest tests/ -x -q` passes.
* `ruff check .` passes.
* Any deferral is recorded in `docs/dev/current.md` with a narrow reason and follow-up.
