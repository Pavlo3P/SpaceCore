4 · Tree spaces: structured elements and block operators
========================================================

**Goal:** solve a coupled 2×2 block system whose unknown keeps its
names and its block shapes. Build the block operator, write the energy
it minimises as a functional, and solve by block Gauss–Seidel, updating
``x.u`` and ``x.p`` by name.

**Before you start:** read :doc:`02_linear_algebra` and
:doc:`03_functionals`. Run the blocks in order in a fresh session; see
:doc:`index` for setup.

1 · A space made of named blocks
--------------------------------

The unknown is a pair :math:`x=(u,p)` with :math:`u\in\mathbb R^3` and
:math:`p\in\mathbb R^2`. Its space is the direct product

.. math::

   T=\mathbb R^3\times\mathbb R^2,\qquad
   \langle x,y\rangle_T=\langle u,u'\rangle+\langle p,p'\rangle .

The inner product is the sum of the leaf inner products, so
:math:`\|x\|_T^2=\|u\|^2+\|p\|^2`. A ``TreeSpace`` built from a
``NamedTuple`` template returns elements of that type.

.. code:: python

    from typing import NamedTuple
    import numpy as np
    import spacecore as sc

    class State(NamedTuple):
        u: object
        p: object

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    U = sc.DenseVectorSpace((3,), ctx)
    P = sc.DenseVectorSpace((2,), ctx)
    T = sc.TreeSpace.from_template(State(0, 0), (U, P), ctx=ctx)

    x = State(u=ctx.asarray([1.0, 2.0, 3.0]), p=ctx.asarray([1.0, -1.0]))
    y = State(u=ctx.asarray([0.5, 0.0, -1.0]), p=ctx.asarray([2.0, 1.0]))
    T.check_member(x)
    z = T.zeros()
    assert isinstance(z, State) and z.u.shape == (3,) and z.p.shape == (2,)
    assert isinstance(T.axpy(2.0, x, y), State)
    np.testing.assert_allclose(T.inner(x, y), x.u @ y.u + x.p @ y.p)
    np.testing.assert_allclose(T.norm(x) ** 2, x.u @ x.u + x.p @ x.p)
    np.testing.assert_allclose(T.flatten(x), [1.0, 2.0, 3.0, 1.0, -1.0])
    assert T.unflatten(T.flatten(x)).p.tolist() == [1.0, -1.0]

``zeros``, ``axpy`` and ``unflatten`` all return a ``State``, so a block
is read as ``x.p`` rather than ``x[1]``. ``flatten`` concatenates the
leaves in field order, each in C order; this is the bridge to a dense
reference used in section 5.

2 · A coupled block operator
----------------------------

Take symmetric positive definite blocks :math:`A:\mathbb R^3\to\mathbb R^3`
and :math:`D:\mathbb R^2\to\mathbb R^2`, and a coupling
:math:`B:\mathbb R^3\to\mathbb R^2`. The block operator is

.. math::

   K\begin{pmatrix}u\\p\end{pmatrix}
   =\begin{pmatrix}A & B^\#\\ B & D\end{pmatrix}
    \begin{pmatrix}u\\p\end{pmatrix}
   =\begin{pmatrix}Au+B^\#p\\ Bu+Dp\end{pmatrix}.

Row :math:`i` of the block matrix produces output block :math:`i`.
Unlike a KKT saddle-point matrix, this K has a positive definite lower
block, and the choice below makes K positive definite.

.. code:: python

    A = sc.DenseLinOp(ctx.asarray([[4.0, 1.0, 0.0],
                                   [1.0, 3.0, 1.0],
                                   [0.0, 1.0, 2.0]]), U, U, ctx)
    B = sc.DenseLinOp(ctx.asarray([[1.0, 0.0, 1.0],
                                   [0.0, 1.0, -1.0]]), U, P, ctx)
    D = sc.DenseLinOp(ctx.asarray([[5.0, 0.0],
                                   [0.0, 4.0]]), P, P, ctx)

    K = sc.BlockMatrixLinOp(State(u=State(u=A, p=B.H),
                                  p=State(u=B, p=D)))
    Kdiag = sc.BlockDiagonalLinOp(State(u=A, p=D))
    coupling = K - Kdiag

    assert K.dom == T and K.cod == T and Kdiag.dom == T
    Kx = K.apply(x)
    assert isinstance(Kx, State)
    np.testing.assert_allclose(Kx.u, A.apply(x.u) + B.H.apply(x.p))
    np.testing.assert_allclose(Kx.p, B.apply(x.u) + D.apply(x.p))
    Cx = coupling.apply(x)
    np.testing.assert_allclose(Cx.u, B.H.apply(x.p))
    np.testing.assert_allclose(Cx.p, B.apply(x.u))

The outer ``State`` lists the rows and each inner ``State`` lists one
row's blocks, so both endpoints of K are ``T`` with named fields.
``BlockDiagonalLinOp`` keeps only A and D. Subtracting it from K with
operator algebra leaves the off-diagonal coupling, which mixes the
blocks: ``Cx.u`` depends only on ``x.p`` and ``Cx.p`` only on ``x.u``.

3 · The adjoint in the tree inner product
-----------------------------------------

The adjoint of a block operator is the transposed block matrix of the
block adjoints:

.. math::

   \langle Kx,y\rangle_T=\langle x,K^\#y\rangle_T,\qquad
   K^\#=\begin{pmatrix}A^\# & B^\#\\ (B^\#)^\# & D^\#\end{pmatrix}
       =\begin{pmatrix}A & B^\#\\ B & D\end{pmatrix}=K .

The second equality uses :math:`A^\#=A`, :math:`D^\#=D` and
:math:`(B^\#)^\#=B`, so K is self-adjoint on T.

.. code:: python

    np.testing.assert_allclose(T.inner(K.apply(x), y),
                               T.inner(x, K.H.apply(y)), atol=1e-12)
    np.testing.assert_allclose(T.flatten(K.H.apply(y)),
                               T.flatten(K.apply(y)), atol=1e-12)

    Kmat = K.to_matrix()
    assert Kmat.shape == (5, 5)
    np.testing.assert_allclose(Kmat, Kmat.T)
    eigenvalues = np.linalg.eigvalsh(Kmat)
    assert eigenvalues.min() > 0.5
    print("eigenvalues of K:", np.round(eigenvalues, 3))

``K.H`` is assembled block by block, never from a matrix. ``to_matrix``
is used only for checks: it is the 5×5 matrix of K in the flattened
coordinates, and its smallest eigenvalue, about 0.51, certifies that K
is positive definite.

4 · The energy as a custom tree functional
------------------------------------------

Because K is self-adjoint and positive definite, solving :math:`Kx=b`
is the same as minimising the strictly convex energy

.. math::

   E(x)=\tfrac12\langle x,Kx\rangle_T-\langle b,x\rangle_T,
   \qquad \nabla E(x)=Kx-b .

The gradient formula uses :math:`K^\#=K`; it is the metric gradient in
T, itself a tree element. Its blocks are the two residuals
:math:`Au+B^\#p-f` and :math:`Bu+Dp-g`, where :math:`b=(f,g)`.

Write E with the two check-free cores. The inherited public ``value``
and ``grad`` check that their argument belongs to T before calling them.

.. code:: python

    class BlockEnergy(sc.Functional):
        """E(x) = 1/2 <x, K x>_T - <b, x>_T for a self-adjoint K on a tree space."""

        _fields = ("K", "b", "dom", "ctx")
        _dynamic = ("K", "b")

        def __init__(self, K, b, dom, ctx=None, check_level=None):
            super().__init__(dom, ctx, check_level=check_level)
            if K.dom != self.domain or K.cod != self.domain:
                raise ValueError("K must map the energy's domain to itself")
            self.domain.check_member(b)
            self.K, self.b = K, b

        def _value_core(self, x):
            T = self.domain
            return 0.5 * T.inner(x, self.K.apply(x)) - T.inner(self.b, x)

        def _grad_core(self, x):
            return self.domain.axpy(-1.0, self.b, self.K.apply(x))

    b = State(u=ctx.asarray([1.0, 2.0, 3.0]), p=ctx.asarray([0.0, 1.0]))
    energy = BlockEnergy(K, b, T)

    grad = energy.grad(x)
    assert isinstance(grad, State)
    np.testing.assert_allclose(grad.u, A.apply(x.u) + B.H.apply(x.p) - b.u)
    np.testing.assert_allclose(grad.p, B.apply(x.u) + D.apply(x.p) - b.p)

    h = 1e-6
    slope = (energy.value(T.axpy(h, y, x))
             - energy.value(T.axpy(-h, y, x))) / (2 * h)
    np.testing.assert_allclose(slope, T.inner(grad, y), rtol=1e-7)

    wrong_shape = State(u=ctx.asarray([1.0, 2.0]), p=x.p)
    try:
        energy.value(wrong_shape)
    except sc.SpaceValidationError as error:
        print("rejected:", error)
    else:
        raise AssertionError("a 2-vector u should not be a member of T")

The first two checks match the gradient's named fields to the block
residuals. The central difference along ``y`` matches
:math:`\langle\nabla E(x),y\rangle_T`, which tests sign and scale. The
last check comes from the inherited ``value``: the cores never test
shapes, yet a ``u`` of length 2 is rejected.

5 · Block Gauss–Seidel, by name
-------------------------------

Block Gauss–Seidel minimises E exactly over one block with the other
fixed, then alternates:

.. math::

   u\leftarrow A^{-1}(f-B^\#p),\qquad p\leftarrow D^{-1}(g-Bu).

Each update sets one block of :math:`\nabla E` to zero, so E never
increases. For a symmetric positive definite K the iteration converges
to the solution of :math:`Kx=b`. Each leaf solve is a CG solve on the
small SPD block; stop when :math:`\|b-Kx\|_T\le10^{-10}\|b\|_T`.

.. code:: python

    x = T.zeros()
    values = [float(energy.value(x))]
    for sweep in range(200):
        x = x._replace(u=sc.cg(A, U.axpy(-1.0, B.H.apply(x.p), b.u), tol=1e-14).x)
        x = x._replace(p=sc.cg(D, P.axpy(-1.0, B.apply(x.u), b.p), tol=1e-14).x)
        values.append(float(energy.value(x)))
        residual = T.axpy(-1.0, K.apply(x), b)
        if T.norm(residual) <= 1e-10 * T.norm(b):
            break
    else:
        raise RuntimeError("block Gauss-Seidel did not converge")

    reference = sc.cg(K, b, tol=1e-12, maxiter=50)
    dense = np.linalg.solve(Kmat, T.flatten(b))

    assert isinstance(x, State) and bool(reference.converged)
    np.testing.assert_allclose(T.flatten(x), T.flatten(reference.x), atol=1e-9)
    np.testing.assert_allclose(T.flatten(x), dense, atol=1e-9)
    np.testing.assert_allclose(x.p, T.unflatten(dense).p, atol=1e-9)
    assert np.max(np.diff(values)) <= 1e-12
    assert np.linalg.norm(T.flatten(energy.grad(x))) <= 1e-9
    print("sweeps:", sweep + 1)
    print("u:", x.u)
    print("p:", x.p)

``_replace`` writes one named field and keeps the other, so the loop
reads like the two update formulas. The residual and its norm come from
T, not from a flattened vector. The result agrees with ``sc.cg`` run on
the whole tree and with ``np.linalg.solve`` on the flattened matrix. The
energy sequence is nonincreasing, as exact block minimisation requires,
and the gradient vanishes at the solution. The loop takes 36 sweeps;
the rate is set by the coupling strength relative to A and D.

Check your understanding
------------------------

* Why does ``coupling = K - Kdiag`` act on T without flattening? Both
  operands map T to T, so operator subtraction is defined blockwise.
* Why must ``_grad_core`` return a ``State`` rather than a flat array?
  The gradient is an element of T, and ``axpy`` in the loop and in the
  finite-difference test combines it with other elements of T.
* Scale the coupling by replacing B with 3B. Recompute the smallest
  eigenvalue of ``K.to_matrix()``. If it is no longer positive, neither
  Gauss–Seidel convergence nor ``sc.cg`` is justified.
* Replace the two leaf CG solves with one step of the gradient
  ``energy.grad(x)`` on each field. Expect many more sweeps; keep the
  residual stopping test and the dense comparison.

**Next:** :doc:`5 · Weighted Tikhonov <05_weighted_tikhonov>` — a
worked inverse problem where the *metric adjoint* does the heavy
lifting.
