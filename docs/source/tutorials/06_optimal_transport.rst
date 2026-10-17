6 · Optimal transport: build the dual from functionals
======================================================

**Goal:** construct the entropic OT dual as a linear functional minus an
exponential functional pulled back through an adjoint. Then maximise it
with gradient ascent, using named potentials ``u`` and ``v``.

**Before you start:** read :doc:`03_functionals` and :doc:`04_tree_spaces`.
Run the blocks in order in a fresh session; see :doc:`index` for setup.

1 · Plans, marginals, and potentials live in spaces
---------------------------------------------------

A plan P sends mass from n source bins to m target bins: ``P[i, j]`` is
the amount sent from i to j. Its row sums must equal the source marginal
and its column sums must equal the target marginal. These vectors may
have different sizes, so their product space is a ``TreeSpace``.

Use the same named structure for the dual potentials and the prescribed
marginals. The field ``u`` refers to the source block, and ``v`` to the
target block. With Euclidean geometry we identify dual covectors with
their Riesz representers in this product space.

.. code:: python

    from typing import NamedTuple
    import numpy as np
    import spacecore as sc

    class UV(NamedTuple):
        u: object
        v: object

    ctx = sc.Context(sc.NumpyOps(), dtype=np.float64)
    ops = ctx.ops
    n, m = 3, 4
    U = sc.DenseVectorSpace((n,), ctx)
    V = sc.DenseVectorSpace((m,), ctx)
    dual_space = sc.TreeSpace.from_template(UV(0, 0), (U, V), ctx=ctx)
    plan_space = sc.DenseCoordinateSpace((n, m), ctx)

    marginals = UV(ctx.asarray([0.2, 0.5, 0.3]),
                   ctx.asarray([0.1, 0.2, 0.3, 0.4]))
    z = dual_space.zeros()
    dual_space.check_member(marginals)
    dual_space.check_member(z)
    assert isinstance(z, UV)
    assert z.u.shape == (n,) and z.v.shape == (m,)
    np.testing.assert_allclose(ops.sum(marginals.u), ops.sum(marginals.v))
    assert np.all(marginals.u > 0) and np.all(marginals.v > 0)

``z.u`` and ``z.v`` are the two potential vectors. ``marginals.u`` and
``marginals.v`` are the prescribed masses. Both are members of
``dual_space``; they have different mathematical roles, not different
storage layouts. The space's inner product pairs corresponding blocks
and adds their dot products.

2 · The marginal operator and its adjoint
-----------------------------------------

Let :math:`A:P\mapsto(P\mathbf1,P^\top\mathbf1)`. It maps the plan
space into the dual-variable space. Its adjoint goes the other way:

.. math::

   A^\#(u,v)_{ij}=u_i+v_j.

To derive this, expand the pairing:

.. math::

   \langle AP,(u,v)\rangle
   =\sum_i u_i\sum_jP_{ij}+\sum_jv_j\sum_iP_{ij}
   =\sum_{ij}P_{ij}(u_i+v_j).

Thus the adjoint of summation is broadcasting. In SpaceCore, ``A.H``
is :math:`A^\#`.

.. code:: python

    def marginal_sums(P):
        return UV(ops.sum(P, axis=1), ops.sum(P, axis=0))

    def potential_sums(z):
        return z.u[:, None] + z.v[None, :]

    A = sc.MatrixFreeLinOp(
        marginal_sums, potential_sums, plan_space, dual_space, ctx,
        euclidean_adjoint=False,
    )

    rng = np.random.default_rng(0)
    P_test = ctx.asarray(rng.standard_normal((n, m)))
    direction = UV(ctx.asarray(rng.standard_normal(n)),
                   ctx.asarray(rng.standard_normal(m)))
    np.testing.assert_allclose(
        dual_space.inner(A.apply(P_test), direction),
        plan_space.inner(P_test, A.H.apply(direction)),
        atol=1e-12,
    )

``euclidean_adjoint=False`` explicitly says the supplied callable already
is the adjoint for the declared geometries. Here both geometries are
Euclidean. We test the identity before using this pair in optimisation.
No matrix representing A is stored.

3 · Derive the linear-minus-exponential dual
--------------------------------------------

Choose a cost matrix C and a positive entropy parameter epsilon. The
primal problem is

.. math::

   \min_{P\geq0,\ AP=r}
   \sum_{ij}C_{ij}P_{ij}
   +\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1),

where r denotes ``marginals``. The entropy term smooths the transport
plan; this is not the unregularised transport linear program.

Use the Lagrangian with the sign convention
:math:`L(P,z)=\text{primal objective}+\langle r-AP,z\rangle`.
Differentiating each positive entry of P gives

.. math::

   C_{ij}+\varepsilon\log P_{ij}-(A^\#z)_{ij}=0,
   \qquad P(z)=\exp((A^\#z-C)/\varepsilon).

Substitution eliminates P and leaves a concave dual objective to
**maximise**:

.. math::

   D(z)=\underbrace{\langle r,z\rangle}_{\ell(z)}
       -\underbrace{\varepsilon\sum_{ij}
          \exp(((A^\#z)_{ij}-C_{ij})/\varepsilon)}_{E(A^\#z)}.

Build the exponential functional on the matrix space first:
:math:`E(X)=\varepsilon\sum_{ij}\exp((X_{ij}-C_{ij})/\varepsilon)`.
Then pull it back along :math:`X=A^\#z`. A pullback simply means
composition: evaluate the operator, then the functional. When epsilon
is one this is precisely ``sum(exp(X - C))``.

.. code:: python

    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, m)
    C = ctx.asarray((xs[:, None] - ys[None, :]) ** 2)
    eps = 0.15

    class SumExpCost(sc.Functional):
        """E(X) = eps * sum(exp((X - C) / eps)), on a real matrix space."""

        _fields = ("C", "eps", "dom", "ctx")
        _dynamic = ("C",)

        def __init__(self, C, eps, dom, ctx=None, check_level=None):
            super().__init__(dom, ctx, check_level=check_level)
            if not np.isfinite(eps) or eps <= 0:
                raise ValueError("eps must be finite and positive")
            self.C = self.ctx.asarray(C)
            self.domain.check_member(self.C)
            self.eps = eps

        def _value_core(self, X):
            return self.eps * self.ops.sum(self.ops.exp((X - self.C) / self.eps))

        def _grad_core(self, X):
            coordinate_gradient = self.ops.exp((X - self.C) / self.eps)
            return self.domain.riesz_inverse(coordinate_gradient)

    linear = sc.InnerProductFunctional(marginals, dual_space)
    exponential = SumExpCost(C, eps, plan_space)
    pulled_back = exponential @ A.H
    dual = linear - pulled_back

``linear`` and ``pulled_back`` now have the same domain, so subtracting
them produces a functional on ``dual_space``. Only ``SumExpCost`` needs
custom code. Composition, subtraction, and their gradient rules come
from SpaceCore's functional algebra.

4 · Read the gradient as a marginal mismatch
--------------------------------------------

In the Euclidean plan geometry, :math:`\nabla E(X)=\exp((X-C)/\varepsilon)`.
The chain rule for the pullback applies the adjoint of :math:`A^\#`,
which is A. Consequently,

.. math::

   \nabla D(z)=r-A\exp((A^\#z-C)/\varepsilon)=r-AP(z).

The gradient has named fields too. ``grad.u`` is missing source mass;
``grad.v`` is missing target mass. Increasing a potential where mass is
missing raises the corresponding entries of the next plan.

.. code:: python

    def transport_plan(z):
        return ops.exp((A.H.apply(z) - C) / eps)

    P = transport_plan(z)
    expected = dual_space.add(marginals, dual_space.scale(-1.0, A.apply(P)))
    grad = dual.grad(z)
    assert isinstance(grad, UV)
    np.testing.assert_allclose(grad.u, expected.u)
    np.testing.assert_allclose(grad.v, expected.v)
    np.testing.assert_allclose(
        dual.value(z),
        dual_space.inner(marginals, z) - eps * ops.sum(P),
    )

    h = 1e-6
    slope = (dual.value(dual_space.axpy(h, direction, z))
             - dual.value(dual_space.axpy(-h, direction, z))) / (2 * h)
    np.testing.assert_allclose(slope, dual_space.inner(grad, direction),
                               rtol=1e-6, atol=1e-8)

The first check tests the composed gradient against the marginal formula;
the second checks the sign and scale of the derivative by finite differences.
Unlike a row-normalised parameterisation, this full dual does not impose
either marginal exactly until convergence. Positivity is automatic.

5 · Maximise the dual with gradient ascent
------------------------------------------

Because D is concave, take an ascent step ``z + step * dual.grad(z)``.
Equivalently, this is gradient descent on ``-dual``. Exponential curvature
has no global constant bound, so choose the step by backtracking: halve
it until the objective increases by at least a small fraction of the
linear prediction :math:`\text{step}\,\|\nabla D\|^2`.

.. code:: python

    z = dual_space.zeros()
    values = []
    for iteration in range(5000):
        value, grad = dual.value_and_grad(z)
        values.append(float(value))
        error = max(float(ops.max(ops.abs(grad.u))),
                    float(ops.max(ops.abs(grad.v))))
        if error < 1e-8:
            break
        step = 1.0
        slope = float(dual_space.inner(grad, grad))
        for _ in range(40):
            candidate = dual_space.axpy(step, grad, z)
            # NumPy trial values may overflow; reject them and shorten the step.
            with np.errstate(over="ignore"):
                trial = float(dual.value(candidate))
            if np.isfinite(trial) and trial >= float(value) + 1e-4 * step * slope:
                break
            step *= 0.5
        else:
            raise RuntimeError("Backtracking failed to find an ascent step")
        z = candidate
    else:
        raise RuntimeError("Marginal tolerance not reached")

    P = transport_plan(z)
    actual = A.apply(P)
    assert np.all(P > 0)
    np.testing.assert_allclose(actual.u, marginals.u, atol=1e-8, rtol=0)
    np.testing.assert_allclose(actual.v, marginals.v, atol=1e-8, rtol=0)
    assert np.min(np.diff(values)) >= -1e-12
    print("gradient steps:", iteration)
    print("source potential u:", z.u)
    print("target potential v:", z.v)
    print("maximum marginal error:", error)

Both marginal errors should be below ``1e-8``, and the dual objective
should be nondecreasing up to rounding. A small transport cost alone
would not certify feasibility. This loop uses the composed functional's
gradient; there are no alternating row and column scaling updates.

Potentials are not unique: adding a constant to u and subtracting it
from v leaves their pairwise sums unchanged. Equal total marginal mass
also leaves the linear term unchanged. Thus D and the recovered plan
are unchanged. Compare plans and marginal errors, not raw potentials.

Check your understanding
------------------------

* Why do both terms live on ``dual_space``? The linear term starts there;
  composition pulls the matrix-space exponential term back there.
* Why does A appear in the gradient although ``A.H`` appears in the
  objective? The chain rule applies the adjoint of the inner map.
* Check the potential freedom by evaluating ``dual.value(UV(z.u + 2,
  z.v - 2))``. It should equal ``dual.value(z)`` up to roundoff.
* Decrease epsilon and restart from zero. Expect a more concentrated plan
  and potentially more difficult step selection; retain the feasibility
  and derivative checks.

**Next:** :doc:`07_manifold_descent` constructs gradient steps respecting
another kind of constraint: membership in a curved manifold.
