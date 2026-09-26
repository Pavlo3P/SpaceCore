# Task: Clean up ProductSpace / product LinOp documentation after structured-element support

Structured `ProductSpace` elements now work for both spaces and product LinOps. The implementation no longer always uses or returns plain tuples: tuple is only the default element representation. Documentation and docstrings should be updated to reflect this.

## Problem

Some docstrings and comments still say product operations return `tuple` or expect tuple-only inputs. This is now misleading because `ProductSpace` may represent elements as registered pytrees/dataclasses through its structure adapter.

Known affected area:

* `spacecore/linop/product/*`

  * `BlockDiagonalLinOp`
  * `StackedLinOp`
  * `SumToSingleLinOp`
  * any other product LinOp docstring saying “tuple,” “tuple of outputs,” or “input tuple” where the correct concept is “product element”

Also inspect:

* `spacecore/space/_product.py`
* `spacecore/space/_structure.py`
* public docs / README snippets mentioning `ProductSpace` elements

## Required wording change

Use representation-neutral language:

* Prefer **“product element”** over **“tuple”**.
* Prefer **“components”** over **“tuple entries.”**
* Say explicitly that tuple elements remain the default representation.
* Say structured pytrees/dataclasses are supported only through `ProductSpace.from_template(...)` or an explicit `PytreeStructure`.
* For dataclasses, mention they must be registered pytrees where required, especially for JAX.

## Do not change behavior

This is a documentation-only task.

Do not modify implementation logic unless a docstring reveals an actual inconsistency.

## Specific updates

1. Update product LinOp class docstrings:

   * domain/codomain may be `ProductSpace`;
   * input/output representation follows the corresponding `ProductSpace` structure;
   * tuple is not guaranteed unless the space uses `TupleStructure`.

2. Update method docstrings for:

   * `apply`
   * `rapply`
   * batch variants
   * adjoint-related methods if they mention tuple output

3. Update `ProductSpace` docs:

   * explain structure adapter briefly;
   * explain tuple default;
   * explain `from_template`;
   * explain flat representation invariant:
     `flatten` always returns the same flat vector for equal components, independent of product element representation.

4. Add or update one short example:

   * tuple default product element;
   * registered dataclass/pytree product element via `from_template`;
   * product LinOp returning the same representation as its codomain.

## Tests / checks

Run:

```bash
pytest tests/spaces/test_product_structure.py
pytest tests/linops/test_product_structure.py
```

Also run any docs/public API checks already present.

## Acceptance criteria

* No product LinOp docstring claims tuple-only input/output unless it specifically refers to the default tuple structure.
* Documentation consistently distinguishes:

  * component sequence/internal iteration;
  * product element/user-facing representation;
  * flat vector representation for solvers.
* Tuple path remains documented as the default and unchanged.
* Structured dataclass/pytree support is documented without implying bare dataclasses work automatically in JAX.
* Tests still pass.
