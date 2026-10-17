2 · Linear algebra: spaces, operators, conjugate gradients
==========================================================

**Goal:** build a matrix-free Laplacian operator, assemble new operators
from it, and solve its linear system with a conjugate-gradient loop you
write yourself over the space operations.

**Before you start:** :doc:`01_backend_and_context`; dot products and matrix multiplication.
Run the code blocks in order in a fresh session; see :doc:`index` for setup.

1 · A space carries geometry
----------------------------

A space fixes the shape of its elements and the inner product between
them. ``DenseVectorSpace((n,), ctx)`` uses the dot product; a
``WeightedInnerProduct(w)`` geometry uses

.. math::

   \langle x, y\rangle_W = \sum_i w_i\, x_i y_i, \qquad (Rx)_i = w_i x_i.

The second formula is the Riesz map R: it converts a vector into the
coefficients that pair with it, so that :math:`\langle x, y\rangle_W = (Rx)^\top y`.

.. code:: python

    import numpy as np
    import spacecore as sc

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    ops = ctx.ops

    w = ctx.asarray([1.0, 4.0, 9.0])
    X3 = sc.DenseVectorSpace((3,), ctx)
    X3w = sc.DenseVectorSpace((3,), ctx, geometry=sc.WeightedInnerProduct(w))
    x = ctx.asarray([3.0, 4.0, 0.0])
    y = ctx.asarray([0.0, 1.0, 2.0])

    np.testing.assert_allclose(X3.inner(x, y), x @ y)
    np.testing.assert_allclose(X3w.inner(x, y), np.sum(w * x * y))
    np.testing.assert_allclose(X3w.norm(ctx.asarray([0.0, 1.0, 0.0])), 2.0)
    np.testing.assert_allclose(X3w.riesz(x), w * x)
    np.testing.assert_allclose(X3w.inner(x, y), X3w.riesz(x) @ y)
    np.testing.assert_allclose(X3w.riesz_inverse(X3w.riesz(x)), x)
    assert X3.is_euclidean and not X3w.is_euclidean

The two spaces hold the same arrays but measure them differently: the
second basis vector has length 2 under the weights. The Riesz map of the
weighted space is multiplication by w, and ``riesz_inverse`` undoes it.

2 · Operators are typed maps with an adjoint
--------------------------------------------

A linear operator :math:`A : X \to Y` is bound to its two spaces.
``A.apply`` is the forward map, and ``A.H`` is the adjoint defined by

.. math::

   \langle A x, y\rangle_Y = \langle x, A^\# y\rangle_X
   \quad\text{for all } x \in X,\ y \in Y.

On Euclidean spaces with real arrays, :math:`A^\#` is the transpose.

.. code:: python

    M = ctx.asarray([[2.0, 1.0, 0.0],
                     [0.0, 3.0, 1.0]])
    X2 = sc.DenseVectorSpace((2,), ctx)
    A = sc.DenseLinOp(M, X3, X2, ctx)
    v = ctx.asarray([1.0, -2.0])

    np.testing.assert_allclose(A.apply(x), M @ x)
    np.testing.assert_allclose(A.H.apply(v), M.T @ v)
    np.testing.assert_allclose(X2.inner(A.apply(x), v), X3.inner(x, A.H.apply(v)))
    try:
        A.apply(v)                      # v lives in X2, not in the domain X3
    except sc.SpaceValidationError as err:
        print("rejected:", err)
    else:
        raise AssertionError("a wrong-shape input was accepted")

``A.H`` maps ``X2`` back to ``X3``, and the identity holds with each
inner product taken in its own space. The public ``apply`` checks its
input against the domain, so a codomain vector is rejected before any
arithmetic runs.

3 · Build a matrix-free Laplacian
---------------------------------

The 1-D Dirichlet Laplacian on n interior grid points is the stencil

.. math::

   (Lx)_i = 2x_i - x_{i-1} - x_{i+1}, \qquad x_0 = x_{n+1} = 0.

L is the tridiagonal matrix with 2 on the diagonal and -1 beside it; it
is symmetric with eigenvalues :math:`2-2\cos(k\pi/(n+1)) > 0`, so it is positive definite.

A SpaceCore operator is written by implementing the check-free cores
``_apply_core`` and ``_rapply_core``; the inherited ``apply``, ``rapply``
and ``A.H`` check inputs and outputs around them. ``_fields`` names the
constructor keyword (``space``, read from the attribute ``domain``) so
that equality, conversion and pytree flattening can rebuild the operator.
For the adjoint use the metric formula :math:`L^\# = R^{-1} L^\top R`;
on a Euclidean space R is the identity and this is the stencil again.

.. code:: python

    class Laplacian1D(sc.LinOp):
        """Dirichlet second difference on one vector space; no matrix stored."""

        _fields = ("space:domain",)

        def __init__(self, space):
            super().__init__(space, space, space.ctx)

        def _stencil(self, x):
            zero = x[:1] * 0
            left = ops.concatenate([zero, x[:-1]])      # x_{i-1}, with x_0 = 0
            right = ops.concatenate([x[1:], zero])      # x_{i+1}, with x_{n+1} = 0
            return 2.0 * x - left - right

        def _apply_core(self, x):
            return self._stencil(x)

        def _rapply_core(self, y):
            R = self.domain                             # L^# = R^{-1} L^T R, L^T = L
            return R.riesz_inverse(self._stencil(R.riesz(y)))

    n = 40
    X = sc.DenseVectorSpace((n,), ctx)
    L = Laplacian1D(X)
    L_np = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

    def matrix_of(op):
        """Column j is op applied to the j-th basis vector."""
        return np.column_stack([op.apply(e) for e in np.eye(op.domain.shape[0])])

    rng = np.random.default_rng(0)
    u = ctx.asarray(rng.standard_normal(n))
    z = ctx.asarray(rng.standard_normal(n))
    np.testing.assert_allclose(matrix_of(L), L_np)
    np.testing.assert_allclose(matrix_of(L.H), L_np.T)
    np.testing.assert_allclose(X.inner(L.apply(u), z), X.inner(u, L.H.apply(z)))
    assert L == Laplacian1D(X)
    lam = np.linalg.eigvalsh(L_np)
    np.testing.assert_allclose(lam, 2 - 2 * np.cos(np.arange(1, n + 1) * np.pi / (n + 1)))
    assert lam.min() > 0

``matrix_of`` applies the stencil to every basis vector and recovers the
tridiagonal matrix built with NumPy; ``L.to_dense()`` does the same through
the batched ``vapply``, which warns on NumPy unless ``_vapply_core`` is also written. The adjoint identity and the
eigenvalue formula both hold, and two Laplacians on the same space
compare equal because their ``_fields`` agree.

4 · Assemble operators from the algebra
---------------------------------------

Sums, scalings and compositions of operators are operators. With
:math:`\sigma > 0`, the reaction-diffusion operator and the squared
Laplacian

.. math::

   S = L + \sigma I, \qquad K = L^\# L

are both symmetric positive definite: :math:`\langle Sx,x\rangle \ge \sigma\|x\|^2`
and :math:`\langle Kx,x\rangle = \|Lx\|^2 > 0` for :math:`x \ne 0`.

.. code:: python

    sigma = 2.0
    S = L + sigma * sc.IdentityLinOp(X)
    K = L.H @ L

    np.testing.assert_allclose(matrix_of(S), L_np + sigma * np.eye(n))
    np.testing.assert_allclose(matrix_of(K), L_np.T @ L_np)
    np.testing.assert_allclose(X.inner(K.apply(u), u), X.norm(L.apply(u)) ** 2)
    np.testing.assert_allclose(X.inner(S.apply(u), z), X.inner(u, S.H.apply(z)))
    print(type(S).__name__, type(K).__name__)

No new class was written: ``S`` and ``K`` are lazy nodes whose ``apply``
calls the Laplacian's cores, and ``S.H`` is assembled from ``L.H`` by the
rule :math:`(L+\sigma I)^\# = L^\# + \sigma I`.

5 · Write conjugate gradients
-----------------------------

For a symmetric positive-definite A, conjugate gradients minimises
:math:`\tfrac12\langle Ax,x\rangle - \langle b,x\rangle` over growing
Krylov subspaces. From :math:`x_0 = 0,\ r_0 = p_0 = b` it repeats

.. math::

   \alpha_k = \frac{\langle r_k, r_k\rangle}{\langle p_k, Ap_k\rangle},\quad
   x_{k+1} = x_k + \alpha_k p_k,\quad
   r_{k+1} = r_k - \alpha_k Ap_k,\quad
   p_{k+1} = r_{k+1} + \frac{\langle r_{k+1}, r_{k+1}\rangle}{\langle r_k, r_k\rangle} p_k,

and stops once :math:`\|r_k\| \le \mathrm{tol}\,\|b\|`. In exact arithmetic it
terminates in at most n steps. Every operation below is a space method or ``A.apply``.

.. code:: python

    def conjugate_gradient(A, b, tol=1e-10, maxiter=1000):
        X = A.domain
        x = X.zeros()
        r = b
        p = r
        rr = X.inner(r, r)
        b_norm = X.norm(b)
        for k in range(maxiter):
            if np.sqrt(rr) <= tol * b_norm:
                return x, k
            Ap = A.apply(p)
            alpha = rr / X.inner(p, Ap)
            x = X.axpy(alpha, p, x)          # x + alpha p
            r = X.axpy(-alpha, Ap, r)        # r - alpha A p
            rr_new = X.inner(r, r)
            p = X.axpy(rr_new / rr, p, r)    # r + beta p
            rr = rr_new
        raise RuntimeError("CG did not reach the tolerance")

    b = ctx.asarray(np.ones(n))
    x_cg, iters = conjugate_gradient(L, b)

    i = np.arange(1, n + 1)
    np.testing.assert_allclose(x_cg, i * (n + 1 - i) / 2, rtol=1e-8)
    np.testing.assert_allclose(x_cg, np.linalg.solve(L_np, b), rtol=1e-8)
    np.testing.assert_allclose(x_cg, sc.cg(L, b, tol=1e-10).x, rtol=1e-8)
    assert X.norm(X.axpy(-1.0, L.apply(x_cg), b)) <= 1e-10 * X.norm(b)
    assert iters <= n
    print("CG iterations on L:", iters)

    _, iters_L = conjugate_gradient(L, z)          # random load: no symmetry to exploit
    x_s, iters_S = conjugate_gradient(S, z)
    np.testing.assert_allclose(x_s, np.linalg.solve(L_np + sigma * np.eye(n), z), rtol=1e-8)
    assert iters_S < iters_L / 2
    print("random load, iterations on L and on S:", iters_L, iters_S)

The unit load gives the discrete parabola :math:`x_i = i(n+1-i)/2`, and
the loop matches it, ``np.linalg.solve`` and ``sc.cg``. The shift in S
raises the smallest eigenvalue from about :math:`(\pi/(n+1))^2 \approx 0.006`
to above :math:`\sigma = 2`, so the condition number drops from about 700
to below 3 and CG on a random load needs less than half the iterations.

6 · Geometry lives in the space, not in the operator
----------------------------------------------------

Put the same stencil on a weighted space. The forward map is unchanged,
but the adjoint must satisfy the identity in the weighted inner product:

.. math::

   L_W^\# = W^{-1} L^\top W \ne L^\top \quad\text{unless } w \text{ is constant}.

The core ``_rapply_core`` already applies :math:`R^{-1}` and R, so nothing
new is written; only the space changes.

.. code:: python

    weights = ctx.asarray(np.linspace(1.0, 3.0, n))
    Xw = sc.DenseVectorSpace((n,), ctx, geometry=sc.WeightedInnerProduct(weights))
    Lw = Laplacian1D(Xw)
    W = np.diag(weights)

    np.testing.assert_allclose(matrix_of(Lw), L_np)
    np.testing.assert_allclose(matrix_of(Lw.H), np.linalg.inv(W) @ L_np.T @ W)
    np.testing.assert_allclose(Xw.inner(Lw.apply(u), z), Xw.inner(u, Lw.H.apply(z)))
    gap = abs(Xw.inner(Lw.apply(u), z) - Xw.inner(u, L_np.T @ z))
    assert gap > 1e-3
    assert Lw != L
    print("transpose misses the weighted identity by", gap)

The weighted adjoint is not the transpose, and the transpose fails the
identity by a visible margin. Since :math:`L_W^\# \ne L_W`, the Laplacian
is not self-adjoint on ``Xw``, so CG's hypothesis fails there; the
geometry, not the stored data, decides which algorithms apply.

Check your understanding
------------------------

* Delete the Riesz maps from ``_rapply_core``. Which assertion fails
  first, and on which space?
* Why does ``conjugate_gradient(S, b)`` converge faster than on L? Compare
  the extreme eigenvalues of both dense matrices.
* Show that :math:`\langle Kx,x\rangle = \|Lx\|^2` would fail on ``Xw`` if
  K were built as ``Laplacian1D(Xw) @ Laplacian1D(Xw)`` instead of ``Lw.H @ Lw``.
* With a constant load CG finishes in about n/2 steps rather than n.
  Which symmetry of the problem halves the Krylov space?

**Next:** :doc:`3 · Functionals <03_functionals>` — scalar
objectives, metric-aware gradients, and gradient descent.
