# Task: ProductSpace accepts tree-structured elements, not just tuples

`ProductSpace` currently hard-wires tuple elements: roughly 15 methods do `zip(self.spaces, x)` and return `tuple(...)`. Decouple element *representation* from component *iteration* via a structure adapter, so users can use tuple elements by default, or opt into a registered pytree/dataclass element type while `ProductSpace` still operates componentwise.

Tuple remains the default and must keep byte-for-byte unchanged behavior. The flat-vector representation (`flatten`/`unflatten`) must stay structure-agnostic so solvers, metric code, and `vapply` machinery remain untouched.

Do phases in order. Run the full test suite after each phase.

## Core invariants

The implementation is correct only if all of the following hold:

1. **Tuple default unchanged.** `ProductSpace(spaces)` behaves exactly as before for tuple elements.
2. **ProductSpace owns arity.** The arity is `len(self.spaces)`. Do not let `TupleStructure` or `PytreeStructure` store an independent arity that can drift out of sync. Pass arity from `ProductSpace` into structure methods when validation is needed.
3. **Round-trip law.** For every valid product element `x`, structurally:
   ```python
   structure.from_components(structure.to_components(x, arity=arity), arity=arity) == x
   ```
4. **Flat representation is structure-agnostic.** A tuple element and a pytree/dataclass element with the same ordered components must produce identical `flatten` output.
5. **Batch means structure of batched components.** A batched product element is a fixed structure whose leaves/components carry the leading batch axis. It is explicitly **not** a batch/list/array of structure instances.
6. **Tree logic is representation-level, not numeric-backend-level.** Use one SpaceCore-level tree API for flattening/unflattening structures. Do not route tree operations through numeric backend ops.

---

## Phase 1 — Structure adapter and tree helpers

### Do

Create a small backend-independent tree helper module, for example:

```python
# spacecore/_tree.py or spacecore/utils/_tree.py

def tree_flatten(x): ...
def tree_unflatten(treedef, leaves): ...
```

This should be a single SpaceCore-level API used by all contexts/backends. It may internally use `jax.tree_util` if available, because JAX treedef machinery is structure-level and works with NumPy/JAX/Torch arrays as leaves. Do **not** put tree flatten/unflatten behind numeric `ops`; tree structure is independent of array backend.

Create `spacecore/space/_structure.py`:

```python
from abc import ABC, abstractmethod


class ProductStructure(ABC):
    """Maps a product element <-> ordered component sequence."""

    @abstractmethod
    def to_components(self, x, *, arity: int) -> tuple:
        """Return ordered components; validate structure and arity."""

    @abstractmethod
    def from_components(self, parts: tuple, *, arity: int):
        """Rebuild an element from ordered components; validate arity."""


class TupleStructure(ProductStructure):
    """Default: elements are plain tuples."""

    def to_components(self, x, *, arity: int) -> tuple:
        # validate isinstance(x, tuple)
        # validate len(x) == arity
        # raise clear TypeError/ValueError otherwise
        ...

    def from_components(self, parts: tuple, *, arity: int):
        # validate len(parts) == arity
        return tuple(parts)

    def __eq__(self, other):
        return isinstance(other, TupleStructure)

    def __hash__(self):
        return hash(TupleStructure)


class PytreeStructure(ProductStructure):
    """Elements are registered pytrees with a fixed treedef/leaf order.

    A bare Python dataclass is not necessarily a pytree. For JAX, tests and
    examples must use a registered pytree/dataclass, e.g.
    `jax.tree_util.register_dataclass` or `@jax.tree_util.register_pytree_node_class`.
    """

    def __init__(self, template_element):
        # leaves, treedef = tree_flatten(template_element)
        # store only treedef / structural metadata, not array leaves
        # optionally store leaf_count for diagnostics
        ...

    @classmethod
    def from_treedef(cls, treedef):
        # optional, useful for pytree-unflattening ProductSpace itself
        ...

    def to_components(self, x, *, arity: int) -> tuple:
        # leaves, treedef = tree_flatten(x)
        # validate treedef == self._treedef
        # validate len(leaves) == arity
        # return tuple(leaves)
        # mismatch -> clear error explaining expected vs received structure/arity
        ...

    def from_components(self, parts: tuple, *, arity: int):
        # validate len(parts) == arity
        # return tree_unflatten(self._treedef, list(parts))
        ...

    def __eq__(self, other):
        # compare structure, not object identity
        # two PytreeStructure objects built from the same pytree structure must be equal
        # compare treedef/leaf-count only; never compare array leaf values
        ...

    def __hash__(self):
        # hash structure-only metadata if possible; otherwise document why not hashable
        ...
```

### Important details

- `ProductSpace`, not the structure, is the source of arity: `arity = len(self.spaces)`.
- `TupleStructure` is stateless apart from type identity; it validates against the arity passed by `ProductSpace`.
- `PytreeStructure` stores only the treedef and structure-only metadata. Do not store template array leaves.
- `PytreeStructure.__eq__` compares treedef/structure, not object identity and not array leaf values.
- A dataclass example must be explicitly registered as a pytree. Otherwise JAX treats the whole dataclass as one opaque leaf and the implementation silently gets the wrong arity.

### Acceptance

- `TupleStructure` round-trips tuples and validates arity using the arity passed by `ProductSpace`.
- `PytreeStructure` round-trips a registered dataclass/pytree element to/from ordered leaves.
- `from_components(to_components(x)) == x` structurally for tuple and registered pytree examples.
- Wrong arity, wrong tuple type, unregistered dataclass, or wrong pytree structure raises a clear error.
- Two `PytreeStructure` instances built from the same registered dataclass structure compare equal even if constructed from different numeric values.

---

## Phase 2 — Route ProductSpace through the adapter

### Do

1. Update construction:

   ```python
   ProductSpace.__init__(spaces, ctx=None, structure=None)
   ```

   Default:

   ```python
   structure = TupleStructure()
   ```

   Store:

   ```python
   self._structure = structure
   ```

   Use:

   ```python
   self._arity = len(self.spaces)  # or a private property returning len(self.spaces)
   ```

   `self._arity` must be derived from `self.spaces`; do not duplicate independently maintained arity state.

2. Replace every `zip(self.spaces, x)` and element-returning `tuple(...)` site with the adapter.

   Pattern for methods returning product elements:

   ```python
   parts = self._structure.to_components(x, arity=len(self.spaces))
   out = tuple(s.op(p, ...) for s, p in zip(self.spaces, parts))
   return self._structure.from_components(out, arity=len(self.spaces))
   ```

   Pattern for methods using two product elements:

   ```python
   x_parts = self._structure.to_components(x, arity=len(self.spaces))
   y_parts = self._structure.to_components(y, arity=len(self.spaces))
   out = tuple(s.op(px, py, ...) for s, px, py in zip(self.spaces, x_parts, y_parts))
   return self._structure.from_components(out, arity=len(self.spaces))
   ```

   Pattern for scalar-returning methods such as `inner`:

   ```python
   x_parts = self._structure.to_components(x, arity=len(self.spaces))
   y_parts = self._structure.to_components(y, arity=len(self.spaces))
   return sum(s.inner(px, py) for s, px, py in zip(self.spaces, x_parts, y_parts))
   ```

3. Convert all componentwise sites, including at least:

   - `add`
   - `sub`, if present
   - `scale`
   - `neg`, if present
   - `zeros`
   - `ones`
   - `inner`
   - `norm`, if it unpacks components directly
   - `flatten`
   - `unflatten`
   - `_check_member`
   - `riesz`
   - `riesz_inverse`
   - `convert`
   - all `*_batch` variants
   - any other method doing `zip(self.spaces, x)`, indexing into product elements, or returning `tuple(...)` componentwise

4. Treat `flatten`/`unflatten` as the structure boundary:

   - `flatten(x)` unpacks via `structure.to_components(...)`, flattens each component, then concatenates into the same flat vector as today.
   - `unflatten(flat)` splits the flat vector exactly as today, unflattens each component, then repacks using `structure.from_components(...)`.
   - The flat vector must be identical for tuple and dataclass/pytree elements with equal ordered components.

5. Preserve batch semantics:

   Existing batch methods must continue to interpret batches as leading-axis component arrays.

   For a pytree/dataclass structure, this means:

   ```python
   MyState(a=batch_a, b=batch_b)
   ```

   not:

   ```python
   [MyState(a=a0, b=b0), MyState(a=a1, b=b1), ...]
   ```

   Unpack the structure once; each leaf/component is already batched. Repack outputs into the same structure with batched leaves/components.

6. `convert(new_ctx)` carries the same structure object or an equivalent structure to the new `ProductSpace`.

7. `__eq__` includes structure equality:

   - tuple-structured products compare equal only to tuple-structured products with equal spaces/context;
   - pytree-structured products compare equal when the spaces/context and treedef structure are equal;
   - tuple and pytree products are not equal even if arity and component spaces match.

8. Pytree registration of `ProductSpace` itself must carry `structure` in aux/static data, not as a traced child. Structure metadata must be structure-only; no array leaves from the template may be stored in aux data.

### Acceptance

- With default `TupleStructure`, all existing `ProductSpace` tests pass unchanged.
- With `PytreeStructure`, `add`/`scale`/`inner`/`flatten`/`unflatten` operate on registered dataclass elements.
- Element-returning operations return the same representation as the product structure: tuple for tuple products, dataclass/pytree for pytree products.
- `inner` returns a scalar and `flatten` returns the same flat vector independent of element representation.
- Batch methods accept and return a structure of batched components, not a batch of structures.

---

## Phase 3 — Ergonomics and public API

### Do

1. Add a factory:

   ```python
   ProductSpace.from_template(spaces, template_element, ctx=None)
   ```

   It builds:

   ```python
   structure = PytreeStructure(template_element)
   return ProductSpace(spaces, ctx=ctx, structure=structure)
   ```

2. Keep the common path unchanged:

   ```python
   ProductSpace(spaces)
   ```

   This must mean tuple elements.

3. Export public names if intended:

   - `ProductStructure`
   - `TupleStructure`
   - `PytreeStructure`

   Update `__all__` and the public API test accordingly.

4. Add a short docstring/example showing registered dataclass usage. For JAX, use either `jax.tree_util.register_dataclass` or `@jax.tree_util.register_pytree_node_class`; do not show a bare unregistered dataclass as the working example.

### Acceptance

- `ProductSpace.from_template((X, Y), template)` yields a product whose valid elements have the same registered pytree/dataclass structure as `template`.
- `add` of two dataclass elements returns a dataclass element.
- `zeros()` and `ones()` return dataclass elements when the product was built from a dataclass template.
- `ProductSpace(spaces)` remains tuple-only by default.

---

## Phase 4 — Tests

Create `tests/spaces/test_product_structure.py`.

### Required tests

1. **Tuple default regression**

   - `ProductSpace((X, Y))` accepts tuple elements.
   - Existing tuple behavior is unchanged.
   - Returned elements are plain tuples.

2. **Registered dataclass/pytree element**

   Use a registered pytree/dataclass, not a bare dataclass. Test both NumPy and JAX where supported.

   Verify:

   - `add`
   - `scale`
   - `inner` returns scalar
   - `zeros` returns dataclass/pytree
   - `ones` returns dataclass/pytree
   - `flatten(x)` equals `flatten(tuple_components)` for equal ordered components
   - `unflatten(flat)` returns dataclass/pytree
   - `unflatten(flatten(x)) == x` structurally and numerically

3. **Round-trip law**

   Directly test:

   ```python
   parts = structure.to_components(x, arity=len(spaces))
   y = structure.from_components(parts, arity=len(spaces))
   assert y == x
   ```

   Use tuple and registered pytree structures.

4. **Mismatch errors**

   Clear errors for:

   - wrong tuple arity;
   - non-tuple passed to tuple product;
   - registered pytree with wrong structure or field order;
   - unregistered dataclass treated as opaque leaf causing arity mismatch;
   - wrong number of components passed to `from_components`.

5. **Structure equality**

   - Two tuple structures compare equal.
   - Two `PytreeStructure` instances built from the same registered dataclass structure compare equal.
   - Two `PytreeStructure` instances with different field structure/treedef compare unequal.
   - `TupleStructure() != PytreeStructure(template)`.
   - `ProductSpace.__eq__` distinguishes different structures.

6. **Convert preserves structure**

   - `convert(new_ctx)` preserves tuple structure for tuple products.
   - `convert(new_ctx)` preserves pytree/dataclass structure for template-built products.

7. **Batch semantics**

   Test a dataclass/pytree of batched components:

   ```python
   MyState(a=batch_a, b=batch_b)
   ```

   Verify `*_batch` methods return the same dataclass/pytree structure with batched leaves.

   Explicitly reject or do not support batch-of-structures such as:

   ```python
   [MyState(a=a0, b=b0), MyState(a=a1, b=b1)]
   ```

8. **Solver smoke test**

   A `ProductSpace` with registered dataclass elements feeds a simple solver such as `cg` that uses `flatten`/`unflatten`/`inner`, and solves correctly. This proves solvers remain structure-agnostic.

9. **ProductSpace pytree registration**

   If `ProductSpace` itself is registered as a pytree, verify that `tree_flatten`/`tree_unflatten` preserves the structure via aux/static data and does not trace or store numeric template leaves.

### Acceptance

- All new tests pass for NumPy.
- JAX tests pass where JAX is available.
- Existing tuple-path tests pass unchanged.
- The full suite passes after every phase.

---

## Constraints

- Tuple default identical to current behavior; no downstream change required to opt in.
- Flat representation structure-agnostic; solvers remain untouched.
- JIT/pytree-safe: structure lives in aux/static metadata, not as a traced child.
- No Python loops added beyond the existing componentwise iteration.
- Tree flatten/unflatten are handled by one SpaceCore-level tree API, not numeric backend `ops`.
- Batch means structure-of-batched-components, not batch-of-structures.
- Do not store template array leaves inside `PytreeStructure`; store only treedef/structure metadata.

---

## Report

For each phase, report:

1. Tests passing/failing.
2. The converted componentwise call sites, including all direct `zip(self.spaces, x)` and tuple-returning paths.
3. Confirmation that tuple default behavior is unchanged.
4. Confirmation that `flatten` output is identical across tuple and pytree/dataclass structures for equal ordered components.
5. Confirmation that `from_components(to_components(x)) == x` holds structurally.
6. Confirmation that batch methods use structure-of-batched-components semantics.
7. Confirmation that structure equality compares treedef/structure, not object identity or array leaf values.
