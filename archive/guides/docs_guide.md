# Docstring standards for SpaceCore

A short, prescriptive standard. Apply consistently across the codebase; reject PRs that violate it. The goal is docstrings users can read and that Sphinx can render cleanly.

---

## 1. Format

**Use NumPy style throughout.** Already used by `numpydoc` in your Sphinx setup. Don't mix with Google style or reStructuredText field lists.

```python
def cg(A, b, *, tol=1e-8, maxiter=None):
    """One-line summary in imperative mood ending with a period.

    Longer description if needed. Wrap at 88 columns. Use blank lines between
    paragraphs. Refer to other functions with :func:`other`, classes with
    :class:`Other`, attributes with :attr:`self.attr`.

    Parameters
    ----------
    A : LinOp
        Symmetric positive-definite operator.
    b : array-like
        Right-hand side, an element of ``A.codomain``.
    tol : float, optional
        Relative residual tolerance. Default is 1e-8.
    maxiter : int, optional
        Maximum number of iterations. Default is ``prod(A.domain.shape)``.

    Returns
    -------
    CGResult
        Named tuple with fields ``x``, ``residual_norm``, ``num_iters``,
        ``converged``.

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If ``A`` is not square.

    Notes
    -----
    Standard Hestenes-Stiefel formulation. The implementation is JIT-compatible
    on the JAX backend.

    Examples
    --------
    >>> import spacecore as sc
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseVectorSpace((100,), ctx)
    >>> A = sc.DenseLinOp(make_spd_matrix(100), X, X, ctx)
    >>> b = ctx.asarray(np.random.randn(100))
    >>> result = sc.cg(A, b, tol=1e-10)
    >>> result.converged
    True
    """
```

## 2. Required sections, by element type

### Functions and methods

| Section | When required |
|---|---|
| Summary line | Always |
| Extended description | When summary insufficient |
| Parameters | Always for public functions; types in annotation, semantics in docstring |
| Returns | Always for public functions; describe what's returned, not just the type |
| Raises | When the function raises documented exceptions |
| Notes | Algorithm details, references, caveats |
| Examples | Public functions, always |
| See Also | Related functions worth knowing about |

### Classes

| Section | When required |
|---|---|
| Summary line | Always |
| Description | Always — explain what kind of object this is, when to use it |
| Parameters | Constructor parameters, always |
| Attributes | Public attributes, always |
| Notes | Mathematical content, design rationale |
| Examples | Public classes, always |

### Properties

A one-line docstring is enough unless there's nontrivial behavior.

```python
@property
def domain(self) -> Space:
    """Domain space of this linear operator."""
    return self._dom
```

### Private methods (underscore-prefixed)

A one-line docstring describing what it does. No Parameters/Returns required. **Always have at least the one-liner** — `def _foo(self): pass` with no docstring is unacceptable even for private methods.

## 3. Summary lines

- Imperative mood: `"Solve Ax = b via conjugate gradients."` not `"Solves Ax = b..."` or `"This function solves..."`.
- One sentence, under 79 characters if possible.
- End with a period.
- No leading whitespace inside the triple-quotes:

```python
def f():
    """Good summary line."""
```

not:

```python
def f():
    """
    Bad — leading blank line.
    """
```

## 4. Math expressions

**Inline math** via reStructuredText math role, double backticks for code:

```
The operator ``A`` represents a linear map :math:`A \colon X \to Y`.
```

**Display math** via the `.. math::` directive in the docstring, with a blank line before and after:

```python
def cg(A, b):
    r"""Solve a linear system via conjugate gradients.

    Find :math:`x \in X` such that

    .. math::

        A x = b

    where :math:`A` is Hermitian positive-definite.
    """
```

**Use raw strings** (`r"""..."""`) for any docstring containing LaTeX. Otherwise backslashes get interpreted as escape sequences and break the math.

**Operator and vector notation conventions:**

- Operators are uppercase italic: $A$, $B$, $K$.
- Vectors are lowercase italic: $x$, $y$, $v$, $b$.
- Spaces are calligraphic or uppercase italic: $\mathcal{H}$, $X$, $Y$.
- Inner products are angle brackets: $\langle x, y \rangle$, written `\langle x, y \rangle`.
- Adjoints use $^*$ (mathematician convention) or $^\dagger$ (physicist convention). **Pick one and use it consistently throughout the library.** SpaceCore uses $^*$ for adjoint: $A^*$.
- Conjugate transpose is $A^*$ for operators on Hilbert spaces. Reserve $A^T$ for the literal transpose without conjugation.
- Norms are double bars: $\|x\|$, written `\| x \|` (escape with backslashes).
- Use $i$ for imaginary unit, not $j$.

**Don't use Unicode math symbols in docstrings.** `→`, `⟨`, `⟩`, `∈`, `‖` may not render correctly in all consoles and IDEs. Always use LaTeX in `:math:` and ASCII (`->`, `<`, `>`, `in`, `||`) in plain prose.

**Don't write math in ASCII for "clarity."** `A.H @ A + lam * I` belongs in code blocks. The mathematical statement belongs in `.. math::` blocks. Pick one mode per paragraph.

## 5. Parameter descriptions

Three rules:

1. **Type annotations carry the type.** Don't repeat `int` in the docstring if the annotation already says `int`.
2. **Docstring carries the semantics.** What does the parameter *mean*? What are valid values? What's the default?
3. **Format**: `name : short type description`.

```python
def lanczos_smallest(A, initial_vector, *, max_iter=100, tol=1e-6):
    """
    Parameters
    ----------
    A : LinOp
        Hermitian operator whose smallest eigenpair is sought.
    initial_vector : array-like
        Starting vector in ``A.domain``. Need not be normalized.
    max_iter : int, optional
        Maximum Krylov dimension. Default is 100.
    tol : float, optional
        Breakdown tolerance for the off-diagonal Lanczos coefficient.
        Default is 1e-6.
    """
```

For complex constraints, use `optional` and explain in the description:

```python
preconditioner : LinOp or None, optional
    Optional left preconditioner. If supplied, must satisfy ``preconditioner.domain
    == preconditioner.codomain == A.domain``. Default is no preconditioning.
```

## 6. Return descriptions

Always document what's returned. Three patterns:

**Simple return:**
```python
Returns
-------
LinOp
    Adjoint view of this operator. Identity satisfies ``op.H.H is op``.
```

**NamedTuple result:**
```python
Returns
-------
CGResult
    Named tuple with fields:

    - ``x``: solution vector in ``A.domain``
    - ``residual_norm``: final residual norm
    - ``num_iters``: number of iterations executed
    - ``converged``: whether ``residual_norm < tol * ||b||``
```

**Multiple returns:**
```python
Returns
-------
unit : array-like
    Normalized vector.
norm : float
    Original norm of ``x``.
```

## 7. Raises section

Document the exceptions a user can reasonably hit due to misuse. Don't document `MemoryError` or `KeyboardInterrupt`.

```python
Raises
------
TypeError
    If ``A`` is not a :class:`LinOp`.
ValueError
    If ``A.domain != A.codomain``.
ContextConflictError
    If ``A`` and ``b`` are bound to incompatible contexts.
```

If the function doesn't raise anything noteworthy, omit the section.

## 8. Examples

Every public function and class **must have** at least one example. Two principles:

**Doctestable when possible.** Examples in `>>> ...` format that can run as doctests. Use them where the inputs and outputs are deterministic. Skip when the example involves random data or floating-point output that won't match exactly.

```python
>>> import spacecore as sc
>>> import numpy as np
>>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
>>> X = sc.DenseVectorSpace((3,), ctx)
>>> A = sc.IdentityLinOp(X, ctx)
>>> x = ctx.asarray([1.0, 2.0, 3.0])
>>> np.allclose(A.apply(x), x)
True
```

**Realistic when not doctestable.** Use a fenced code block within the docstring for examples that need to show realistic usage:

```python
"""
Examples
--------
Build a Tikhonov-regularized normal-equation operator and solve it:

.. code-block:: python

    K = sc.MatrixFreeLinOp(blur_apply, blur_adjoint, X, X, ctx)
    grad = make_gradient_operator(X, ctx)
    system = K.H @ K + lam * grad.H @ grad
    result = sc.cg(system, K.H.apply(b))
"""
```

**Examples should be:**

- Self-contained: include the imports and context setup.
- Minimal: under 10 lines of actual code where possible.
- Meaningful: solve a real problem, not a toy.
- Runnable: copy-paste should work.

Don't show APIs as `func(arg1, arg2, ...)` — show with concrete values.

## 9. Cross-references

Use Sphinx roles to link other parts of the library:

- `:class:\`LinOp\`` → `LinOp` link
- `:func:\`cg\`` → `cg()` link
- `:attr:\`self.ctx\`` → `self.ctx` link
- `:meth:\`Space.inner\`` → `Space.inner()` link
- `:mod:\`spacecore.linalg\`` → `spacecore.linalg` link

**Always use cross-references** when mentioning a documented API. Never:

```
"See the LinOp class for details."
```

Always:

```
"See :class:`LinOp` for details."
```

## 10. Notes section

Use for material that's important but not essential for first reading:

- Algorithm details and references
- Convergence properties
- JIT or backend caveats
- Mathematical conventions (e.g., conjugation convention)
- Complexity

Keep brief. If it grows past 200 words, consider putting it in a Sphinx tutorial instead.

**Cite references in NumPy style:**

```
Notes
-----
The implementation follows the LSQR algorithm of Paige and Saunders [1]_.

References
----------
.. [1] Paige, C. C. and M. A. Saunders, "LSQR: An algorithm for sparse linear
   equations and sparse least squares," ACM Trans. Math. Soft., 8 (1982),
   43-71.
```

Place `References` *after* `Notes`, before `Examples`.

## 11. Conventions specific to SpaceCore

These reflect the library's particular abstractions and should appear in many docstrings.

### Always specify the inner-product convention

When mentioning an inner product in math, state which space's inner product:

```
Returns :math:`\langle x, y \rangle_X`, the inner product on the domain space.
```

Not just `\langle x, y \rangle` — different spaces have different inner products.

### Always specify conjugation in adjoints

For complex operators, the adjoint convention must be explicit:

```
The adjoint :math:`A^*` satisfies :math:`\langle A x, y \rangle_Y =
\langle x, A^* y \rangle_X` for :math:`x \in X` and :math:`y \in Y`.

For complex operators this is the *conjugate* transpose, not just the transpose.
```

### Explain what `convert` does in `_convert` docstrings

`_convert` is the hook subclasses override. Always document what's converted:

```python
def _convert(self, new_ctx):
    """Convert to ``new_ctx``.

    Rebuilds the stored matrix by calling ``new_ctx.asarray(self.A)``, and
    converts the domain and codomain spaces. The matrix dtype follows the
    dtype resolution policy of ``new_ctx``.
    """
```

### State JIT compatibility for solver functions

```
Notes
-----
This function is JIT-compatible on the JAX backend. ``max_iter`` is a static
argument; changing it will trigger retracing.
```

### Document complex-input behavior

If a function works on real and complex inputs, say so explicitly:

```
Works on both real-valued and complex-valued ``A``. For complex ``A``, the
adjoint uses conjugate transpose.
```

If a function does *not* support complex, say so even more loudly:

```
Notes
-----
This function only supports real-valued operators. For complex operators, use
:func:`other_thing` instead.
```

## 12. What to avoid

- **Don't write "TODO" in user-facing docstrings.** Either fix it or move the TODO to an issue tracker.
- **Don't write what the code does line by line.** Write what the function does as a whole.
- **Don't restate the type signature in prose.** "`x` is an integer parameter that takes an integer value" — kill the redundancy.
- **Don't use first person ("I", "we") in user-facing docstrings.** Use imperative ("Returns...", "Computes...").
- **Don't use second person ("you") in technical sections.** OK in tutorials and READMEs; jarring in API docs.
- **Don't include implementation details users don't need.** "Internally we use a while loop with..." — only document if it affects behavior (e.g., "uses ``ops.while_loop`` so this function is JIT-compatible").
- **Don't promise behavior the code doesn't deliver.** Test the examples.
- **Don't use vague qualifiers.** "Usually converges" — under what conditions? "Reasonably fast" — vs. what?

## 13. Examples of the standard applied

### Example: a simple method

```python
@property
def H(self) -> "LinOp":
    """Hermitian-adjoint view of this operator.

    Returns an adjoint view where ``apply`` becomes ``rapply`` and vice versa.
    The result satisfies ``op.H.H is op``.

    Returns
    -------
    LinOp
        Adjoint view satisfying :math:`\langle A x, y \rangle_Y = \langle x,
        A^* y \rangle_X` for all admissible :math:`x, y`.
    """
```

### Example: a solver function

```python
def cg(
    A: LinOp,
    b: Any,
    *,
    tol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    check_every: int = 64,
) -> CGResult:
    r"""Solve :math:`A x = b` via the conjugate gradient method.

    Requires :math:`A` to be Hermitian positive-definite. For non-Hermitian or
    indefinite operators, see :func:`lsqr` or :func:`minres` (when available).

    The iteration terminates when :math:`\| r_k \| < \max(\text{atol},
    \text{tol} \cdot \| b \|)` or ``maxiter`` iterations have elapsed,
    whichever comes first.

    Parameters
    ----------
    A : LinOp
        Hermitian positive-definite operator.
    b : array-like
        Right-hand side, an element of ``A.codomain``.
    tol : float, optional
        Relative tolerance for the residual. Default is 1e-8.
    atol : float, optional
        Absolute tolerance. Default is 0 (relative only).
    maxiter : int or None, optional
        Maximum iterations. Default is ``prod(A.domain.shape)``.
    check_every : int, optional
        Refresh convergence diagnostics every this many iterations. Default
        is 64.

    Returns
    -------
    CGResult
        Named tuple with fields ``x``, ``residual_norm``, ``num_iters``,
        ``converged``.

    Raises
    ------
    TypeError
        If ``A`` is not a :class:`LinOp`.
    ValueError
        If ``A`` is not square.

    Notes
    -----
    Implements the standard Hestenes-Stiefel CG iteration. JIT-compatible on
    the JAX backend; ``maxiter`` and ``check_every`` are static arguments.

    See Also
    --------
    lsqr : least-squares solver for non-square or non-symmetric systems
    lanczos_smallest : smallest eigenpair of a Hermitian operator

    References
    ----------
    .. [1] Hestenes, M. R. and Stiefel, E., "Methods of Conjugate Gradients
       for Solving Linear Systems," J. Res. Natl. Bur. Stand., 49 (1952),
       409-436.

    Examples
    --------
    Solve a positive-definite system:

    >>> import spacecore as sc
    >>> import numpy as np
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseVectorSpace((3,), ctx)
    >>> M = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
    >>> A = sc.DenseLinOp(M, X, X, ctx)
    >>> b = ctx.asarray([1., 2., 3.])
    >>> result = sc.cg(A, b, tol=1e-10)
    >>> result.converged
    True
    >>> np.allclose(A.apply(result.x), b)
    True
    """
```

### Example: a class

```python
class DiagonalLinOp(LinOp):
    r"""Linear operator acting as elementwise multiplication by a vector.

    Represents the linear map :math:`D \colon X \to X` defined by

    .. math::

        (D x)_i = d_i x_i

    where :math:`d` is the stored diagonal. For complex :math:`d`, the
    adjoint :math:`D^* x_i = \overline{d_i} x_i` uses the conjugate diagonal.

    Parameters
    ----------
    diagonal : array-like
        Diagonal entries. Shape must match ``space.shape``.
    space : VectorSpace or None, optional
        Vector space on which the operator acts. If ``None``, inferred from
        ``diagonal.shape``.
    ctx : Context, str, or None, optional
        Context for backend, dtype, and validation. Inferred from inputs if
        ``None``.

    Attributes
    ----------
    diagonal : array-like
        Stored diagonal as supplied. Has shape ``space.shape``.
    domain : VectorSpace
        Domain space, same as ``space``.
    codomain : VectorSpace
        Codomain space, equal to ``domain``.

    Notes
    -----
    Storage is :math:`O(n)` for the diagonal plus :math:`O(n)` for the cached
    conjugated diagonal in the complex case.

    Examples
    --------
    >>> import spacecore as sc
    >>> import numpy as np
    >>> ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    >>> X = sc.DenseVectorSpace((3,), ctx)
    >>> D = sc.DiagonalLinOp(ctx.asarray([2., 3., 4.]), X, ctx)
    >>> x = ctx.asarray([1., 1., 1.])
    >>> D.apply(x)
    array([2., 3., 4.])
    """
```

## 14. Enforcement

Use the following tools to keep docstrings honest:

- **`numpydoc validation`** — `numpydoc.validate` in CI catches missing sections, wrong section order, etc. Add to `pyproject.toml`:

  ```toml
  [tool.numpydoc_validation]
  checks = [
      "all",           # report all checks
      "EX01",          # but allow missing Examples (relax later)
      "SA01",          # and missing See Also
  ]
  exclude = [
      '\._',           # don't validate private methods
  ]
  ```

- **`ruff D` rules** — pydocstyle checks. Enable `D101` (missing docstring in public class), `D102` (public method), `D103` (public function), `D200`, `D205`, `D400`, `D401`.

- **`sphinx-build -W -n`** — treats warnings as errors and reports nitpicky cross-reference errors. Run in CI.

- **`doctest`** — run examples as tests:

  ```bash
  pytest --doctest-modules spacecore/
  ```

  Catches the "documented example doesn't actually work" failure mode.

## 15. Migration approach

You have ~3000 lines of source. Don't rewrite all docstrings at once. Instead:

1. **Today**: enforce on new code via CI. Add the tooling now.
2. **One module per week**: rewrite docstrings module-by-module, in roughly this priority: `linalg/` (most user-facing), `linop/` (most-used types), `space/`, `functional/`, then backends.
3. **By v0.3**: all public APIs conformant. Private internals can stay as is.

The most-important docstrings are the ones for `cg`, `lsqr`, `lanczos_smallest`, `power_iteration`, `DenseLinOp`, `MatrixFreeLinOp`, `Context`, `VectorSpace`, `LinOp.H`, `LinOp.compose`. Those ten things together are 80% of what users will look at. Get them right; everything else can come gradually.