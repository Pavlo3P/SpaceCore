1 · Backend and Context
=======================

**Goal:** build elements and an operator on a NumPy context, write power
iteration once against the space's operations, and run it unchanged on JAX
after converting the operator.

**Before you start:** basic Python and NumPy arrays.
Run the code blocks in order in a fresh session; see :doc:`index` for setup.

A ``Context`` is a ``BackendOps`` (the array library: ``matmul``, ``exp``,
``eigh``, ... from NumPy, JAX, Torch or CuPy) plus a default ``dtype``.
Every space, operator and functional is bound to one context and carries its
own ``check_level``, which sets how much runtime validation its public
methods perform.

1 · A context is a backend plus a dtype
---------------------------------------

``NumpyOps`` is always available; ``JaxOps``, ``TorchOps`` and ``CuPyOps``
need their optional dependency. ``ctx.asarray`` moves Python or NumPy data
onto the context's backend at the context's dtype.

.. code:: python

    import importlib.util
    import numpy as np
    import spacecore as sc

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    ops = ctx.ops

    x = ctx.asarray([1, 2, 3])
    assert isinstance(x, np.ndarray) and x.dtype == np.float64
    print("backend:", ops.family, " dtype:", ctx.dtype)

The integer list became a ``float64`` NumPy array because the context fixes
the dtype of new arrays. The context owns no arrays and holds no checking
policy; it only describes how arrays are made.

2 · Build elements of a space
-----------------------------

A space fixes the shape, the scalar field and the inner product of its
elements. ``DenseVectorSpace`` holds one-dimensional vectors;
``DenseCoordinateSpace`` holds arrays of any fixed shape. With the default
Euclidean geometry,

.. math::

   \langle x,y\rangle=\sum_i x_i y_i,\qquad \|x\|=\sqrt{\langle x,x\rangle}.

The norm is induced by the inner product, so both come from the space, not
from ``numpy``.

.. code:: python

    X = sc.DenseVectorSpace((3,), ctx)
    Mats = sc.DenseCoordinateSpace((2, 3), ctx)

    z = X.zeros()
    y = ctx.asarray([4.0, -1.0, 0.5])
    W = Mats.zeros()
    for element, space in [(z, X), (x, X), (y, X), (W, Mats)]:
        space.check_member(element)
    assert z.shape == (3,) and W.shape == (2, 3)

    w = X.axpy(2.0, x, y)                      # 2 x + y
    np.testing.assert_allclose(w, 2 * np.array([1, 2, 3]) + np.array([4, -1, 0.5]))
    np.testing.assert_allclose(X.inner(x, y), np.dot(x, y))
    np.testing.assert_allclose(X.norm(x), np.linalg.norm(x))

``zeros`` and ``asarray`` produce members; ``check_member`` raises when an
array is not one. ``axpy``, ``inner`` and ``norm`` agree with the NumPy
formulas for the Euclidean geometry.

3 · ``check_level`` decides what is caught
------------------------------------------

Each object takes ``check_level=`` at construction; without it, it uses the
ambient level (``sc.get_check_level()``, ``"standard"`` unless changed with
``sc.set_check_level`` or ``with sc.use_check_level(...)``).

============ ==============================================
level        meaning
============ ==============================================
``none``     skip optional checks; invariants still apply
``cheap``    local structural checks (shape, dtype, backend)
``standard`` the default
``strict``   adds expensive probes (e.g. adjoint dot-tests)
============ ==============================================

Three wrong inputs for the real space :math:`\mathbb R^3`: a wrong shape, a
complex array, and a plain Python list.

.. code:: python

    X_checked = sc.DenseVectorSpace((3,), ctx, check_level="standard")
    X_silent = sc.DenseVectorSpace((3,), ctx, check_level="none")
    wrong = {
        "shape (4,)": np.zeros(4),
        "complex": np.array([1 + 1j, 0, 0]),
        "list": [0.0, 0.0, 0.0],
    }
    for name, bad in wrong.items():
        try:
            X_checked.check_member(bad)
        except sc.SpaceValidationError as exc:
            print(f"standard rejects {name}: {exc}")
        else:
            raise AssertionError(f"standard accepted {name}")
        X_silent.check_member(bad)             # no exception at "none"

    try:
        X_checked.scale(1j, x)
    except sc.SpaceValidationError:
        pass
    else:
        raise AssertionError("a real space accepted a complex multiplier")
    assert X_checked.ctx == X_silent.ctx

Both spaces share one context and differ only in policy. At ``standard``
every wrong input raises ``SpaceValidationError``, including scaling by
``1j``, which would leave a space over the reals. At ``none`` the same
inputs pass; ``none`` skips optional checks, not constructor invariants or
backend errors.

4 · Write an operator: cores, not public methods
------------------------------------------------

A ``LinOp`` subclass implements two check-free cores, ``_apply_core`` and
``_rapply_core``. The inherited public ``apply`` and ``rapply`` check the
input and output against the domain and codomain, then call the cores. For
a symmetric matrix M with the Euclidean inner product the adjoint is M
itself:

.. math::

   \langle Mx,y\rangle=\langle x,M^\top y\rangle=\langle x,My\rangle.

The class below stores M and lists its constructor fields; ``_fields`` is
what lets ``convert`` rebuild it on another backend in section 6.

.. code:: python

    class SymmetricMatVec(sc.LinOp):
        """x -> M x for a real symmetric matrix M."""

        _fields = ("M", "space:domain", "ctx")
        _dynamic = ("M",)

        def __init__(self, M, space, ctx=None, check_level=None):
            super().__init__(space, space, ctx, check_level=check_level)
            self.M = self.ctx.asarray(M)

        def _apply_core(self, x):
            return self.ops.matmul(self.M, x)

        def _rapply_core(self, y):
            return self.ops.matmul(self.M, y)     # symmetric: adjoint is M

    n = 6
    rng = np.random.default_rng(0)
    G = rng.standard_normal((n, n))
    M = (G + G.T) / 2
    V = sc.DenseVectorSpace((n,), ctx)
    A = SymmetricMatVec(M, V)

    u = ctx.asarray(rng.standard_normal(n))
    v = ctx.asarray(rng.standard_normal(n))
    np.testing.assert_allclose(V.inner(A.apply(u), v), V.inner(u, A.H.apply(v)))
    np.testing.assert_allclose(A.apply(u), sc.DenseLinOp(M, V, V, ctx).apply(u))
    try:
        A.apply(np.zeros(n + 1))
    except sc.SpaceValidationError:
        pass
    else:
        raise AssertionError("apply accepted a vector of the wrong length")

The adjoint identity holds, and the result matches SpaceCore's own
``DenseLinOp``. The wrong-length input is rejected by the inherited
``apply``; ``_apply_core`` contains no check.

5 · Power iteration, written once against the space
---------------------------------------------------

Power iteration for a self-adjoint T repeats

.. math::

   x_{k+1}=\frac{Tx_k}{\|Tx_k\|},\qquad
   \lambda_k=\langle x_k,Tx_k\rangle,\qquad
   r_k=Tx_k-\lambda_k x_k.

It converges to the eigenvalue of largest *magnitude*. For self-adjoint T
and unit :math:`x_k`, some eigenvalue lies within :math:`\|r_k\|` of
:math:`\lambda_k`, so the residual norm is the stopping test.

.. code:: python

    eigs = np.linalg.eigvalsh(M)
    print("eigenvalues of M:", eigs)
    assert abs(eigs[0]) > abs(eigs[-1])

Here the most negative eigenvalue has the largest magnitude, so iterating
with A would find :math:`\lambda_{\min}`. Shift by
:math:`c\ge\max_i|\lambda_i|`: every eigenvalue of :math:`A+cI` is
nonnegative, the largest one is :math:`\lambda_{\max}+c`, and the shift is
undone afterwards.

.. code:: python

    def largest_eigenvalue(T, x0, tol, max_iter=2000):
        """Power iteration using only T.apply and T.domain's operations."""
        X = T.domain
        x = X.scale(1.0 / X.norm(x0), x0)
        for k in range(max_iter):
            Tx = T.apply(x)
            lam = X.inner(x, Tx)
            residual = X.norm(X.axpy(-lam, x, Tx))
            if residual <= tol * abs(lam):
                return float(lam), k
            x = X.scale(1.0 / X.norm(Tx), Tx)
        raise RuntimeError("power iteration did not converge")

    c = float(np.max(np.abs(eigs)))
    B = A + c * sc.IdentityLinOp(V)           # SumLinOp, built by the algebra
    np.testing.assert_allclose(V.inner(B.apply(u), v), V.inner(u, B.H.apply(v)))

    x0 = ctx.asarray(rng.standard_normal(n))
    lam_B, iters = largest_eigenvalue(B, x0, tol=1e-8)
    lam_np = lam_B - c
    np.testing.assert_allclose(lam_np, eigs[-1], rtol=1e-10)
    print(f"numpy: lambda_max = {lam_np:.12f} after {iters} iterations")

``largest_eigenvalue`` never names ``numpy``: it uses ``T.apply`` and the
domain's ``scale``, ``inner``, ``norm`` and ``axpy``. The shifted operator
``B`` comes from operator algebra and still satisfies the adjoint
identity. The result matches ``np.linalg.eigvalsh`` to ten digits.

6 · Move the operator with ``.convert(...)``
--------------------------------------------

``convert`` accepts a ``Context``, a backend name or a ``BackendFamily``.
It rebuilds an object from its fields on the new context; arrays such as
``M`` are moved with the new context's ``asarray``. A JAX context is used
here in ``float32`` (JAX's default; 64-bit mode is not enabled), so the
tolerances are those of single precision. Without JAX installed, the block
falls back to a NumPy ``float32`` context and runs the same code.

.. code:: python

    HAVE_JAX = importlib.util.find_spec("jax") is not None
    if HAVE_JAX:
        import jax.numpy as jnp
        ctx32 = sc.Context(sc.JaxOps(), dtype=jnp.float32)
    else:
        ctx32 = sc.Context(sc.NumpyOps(), dtype=np.float32)

    B32 = B.convert(ctx32)
    V32 = B32.domain
    assert V.convert(ctx) is V                 # same context: no rebuild
    assert V32.ops.family == ctx32.ops.family and V32.dtype == ctx32.dtype

    u32 = ctx32.asarray(np.asarray(u))
    np.testing.assert_allclose(np.asarray(B32.apply(u32)), B.apply(u), rtol=1e-5, atol=1e-5)

    lam_B32, iters32 = largest_eigenvalue(B32, ctx32.asarray(np.asarray(x0)), tol=1e-5)
    np.testing.assert_allclose(lam_B32 - c, eigs[-1], rtol=1e-4)
    print(f"{ctx32.ops.family}: lambda_max = {lam_B32 - c:.6f} after {iters32} iterations")

    try:
        B32.apply(u)                           # float64 NumPy input
    except sc.SpaceValidationError as exc:
        print("converted operator rejects the old array:", exc)
    else:
        raise AssertionError("float32 operator accepted a float64 array")

``SymmetricMatVec`` inside the sum was rebuilt from its ``_fields`` with no
conversion code of its own. The converted operator agrees with the original
to single precision, and the unchanged ``largest_eigenvalue`` reproduces
:math:`\lambda_{\max}` to four digits. A float64 NumPy array is not a member
of the float32 space, so the converted operator rejects it at ``standard``.

7 · Set defaults once for the whole program
--------------------------------------------

Every object so far received ``ctx`` and ``check_level`` explicitly. You can
instead set both once, at the top of a script; every space, operator and
functional built without them picks up the defaults.

* ``sc.set_context("jax", dtype="float32")`` — default backend and dtype.
* ``sc.set_check_level("cheap")`` — default validation level.
* ``sc.set_traced_check_level("cheap")`` — the cap while ``jax.jit`` /
  ``torch.compile`` trace (default ``"none"``: no checks in compiled code).

``with sc.use_context(...)`` and ``with sc.use_check_level(...)`` change the
defaults only inside a block, for the current thread or task. An explicit
argument always wins over a default.

.. code:: python

    sc.set_context("numpy", dtype="float32")
    sc.set_check_level("cheap")

    W = sc.DenseCoordinateSpace((3,))                 # no ctx, no check_level
    assert W.dtype == np.float32 and W.check_level == "cheap"

    with sc.use_check_level("strict"):
        assert sc.DenseCoordinateSpace((3,)).check_level == "strict"
    assert sc.DenseCoordinateSpace((3,), check_level="none").check_level == "none"

    sc.set_context("numpy", dtype="float64")          # back to the built-in defaults
    sc.set_check_level(None)
    assert sc.get_check_level() == "standard"

Defaults are read when an object is constructed. Changing them later does not
touch objects that already exist: ``W`` stays ``float32`` and ``"cheap"``.

Check your understanding
------------------------

* Remove the shift (call ``largest_eigenvalue(A, x0, tol=1e-8)``). Which
  eigenvalue does it return, and why?
* Rebuild ``A`` with ``check_level="none"`` and pass ``np.zeros(n + 1)`` to
  ``A.apply``. Which error appears now, and which layer raises it?
* Tighten the float32 tolerance to ``1e-8``. Predict whether the loop still
  stops, given float32 machine epsilon of about ``1.2e-7``.
* Count iterations against the ratio
  :math:`(\lambda_{n-1}+c)/(\lambda_n+c)` of the two largest eigenvalues of
  ``B``; a smaller ratio means fewer iterations.

**Next:** :doc:`2 · Linear algebra <02_linear_algebra>` — build
spaces and operators and solve a linear system with conjugate gradients.
