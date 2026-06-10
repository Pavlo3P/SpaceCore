# SpaceCore docstring style

SpaceCore uses NumPy/numpydoc docstrings for public APIs.

Canonical exemplars:

- `spacecore/linop/_base.py::LinOp` for class-style documentation.
- `spacecore/linalg/_cg.py::cg` and `CGResult` for solver/result documentation.

## Required section order

Use these sections in this order. Omit sections that do not apply; do not
reorder them.

1. One-line summary.
2. Extended summary.
3. `Parameters`.
4. `Returns` / `Yields`, or `Attributes` for classes.
5. `Raises`.
6. `See Also`.
7. `Notes`.
8. `References`.
9. `Examples`.

## Mathematical requirements

- Always state spaces for maps, for example `A : X -> Y`.
- Always state the element array representation and shape conventions.
- Always state backend/context assumptions.
- Operator docstrings must distinguish metric adjoint from coordinate conjugate transpose.
- Functional docstrings must state gradient semantics, including whether `grad(x)` is Riesz-represented in the domain geometry.
- Solver docstrings must state square/Hermitian/positive-definite assumptions with respect to the relevant space inner product.
- Do not document future or planned APIs as current APIs.

## Tiny-example rules

Every public class, constructor, and linalg function should have an `Examples`
section unless it is genuinely impossible to demonstrate in 10 lines or fewer;
record exceptions in the 0.3.1 docstring audit.

Examples must be deterministic and backend-stable:

- Use `sc.Context(sc.NumpyOps(), dtype=np.float64)`.
- Do not use random values in doctests.
- Use small integer-valued floats so printed output is stable.
- Demonstrate the contract, not just the call.
- LinOp examples should show both `apply` and `rapply`.
- Weighted-geometry examples should show that `inner` differs from Euclidean dot.
- Solver examples should show reading result fields such as `result.converged` and `result.num_iters`.
- Non-NumPy backend examples must be `.. code-block:: python`, not doctests, because optional dependencies and array reprs differ.

## Class template

```python
r"""
Represent <mathematical object> as a SpaceCore <Space/LinOp>.

<For operators: state A : X -> Y, element representation, what rapply computes,
and geometry restrictions.>

Parameters
----------
...

Attributes
----------
...

Raises
------
...

See Also
--------
...

Notes
-----
<Adjoint/geometry/backend contract.>

Examples
--------
>>> import numpy as np
>>> import spacecore as sc
>>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
>>> X = sc.DenseCoordinateSpace((2,), ctx)
>>> A = sc.DenseLinOp(ctx.asarray([[1.0, 0.0], [0.0, 2.0]]), X, X, ctx)
>>> A.apply(ctx.asarray([3.0, 4.0]))
array([3., 8.])
>>> A.rapply(ctx.asarray([1.0, 1.0]))
array([1., 2.])
"""
```

## Solver template

```python
r"""
Solve :math:`A x = b` on a SpaceCore space.

The operator is a map :math:`A : X \to X` and must be Hermitian positive
definite with respect to ``A.domain.inner``.

Parameters
----------
A : LinOp
    Square Hermitian positive-definite operator.
b : array-like
    Right-hand side in ``A.domain``.

Returns
-------
CGResult
    Named tuple containing the solution and convergence diagnostics.

Notes
-----
Residual norms are measured with ``A.domain.norm``. The convergence criterion is
``residual_norm <= tol`` or the documented relative criterion.

Examples
--------
>>> import numpy as np
>>> import spacecore as sc
>>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
>>> X = sc.DenseCoordinateSpace((2,), ctx)
>>> A = sc.DenseLinOp(ctx.asarray([[2.0, 0.0], [0.0, 3.0]]), X, X, ctx)
>>> result = sc.cg(A, ctx.asarray([4.0, 9.0]), tol=1e-12, maxiter=10)
>>> result.converged
True
>>> result.num_iters <= 2
True
"""
```
