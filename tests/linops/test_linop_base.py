"""Tests for :class:`spacecore.LinOp` — the abstract linear-map base.

Checklist item 1:

* ``domain`` / ``codomain`` properties match the constructor args (after
  context normalization).
* ``apply`` / ``rapply`` are abstract — must be overridden.
* ``__call__(x)`` is an alias for ``apply(x)``.
* ``adjoint_apply(y) == rapply(y)`` on every concrete subclass.
* ``H`` returns an adjoint view; ``A.H.H is A`` (idempotent double adjoint).
* Operator algebra dunders: ``__add__``, ``__radd__``, ``__neg__``, ``__sub__``,
  ``__rsub__``, ``__mul__``, ``__rmul__``, ``__matmul__``.
* ``to_dense`` / ``to_matrix`` round-trip via ``apply``.
* ``assert_domain`` / ``assert_codomain`` raise on mismatched inputs.
* ``__eq__`` returns ``NotImplemented`` for non-LinOp (so ``op == None``
  returns ``False`` without raising).
* ``is_hermitian()`` default returns ``None`` (unknown) — subclasses
  override.

The default ``vapply`` / ``rvapply`` fallback (via ``ops.vmap``) is exercised
implicitly through every concrete-class test; the dedicated batched suite
is in :mod:`tests.linops.test_batched_apply`.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacecore as sc


# ===========================================================================
# Domain / codomain / __call__
# ===========================================================================
class TestDomainCodomain:
    def test_domain_and_codomain_match_constructor(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), X, Y, numpy_ctx
        )
        assert op.domain == X
        assert op.codomain == Y
        # Aliases.
        assert op.dom == op.domain
        assert op.cod == op.codomain

    def test_call_alias_equals_apply(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        x = numpy_ctx.asarray([1.0, 2.0])
        np.testing.assert_allclose(op(x), op.apply(x))


# ===========================================================================
# adjoint_apply == rapply
# ===========================================================================
class TestAdjointAlias:
    def test_adjoint_apply_equals_rapply(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        op = sc.DenseLinOp(A, X, Y, numpy_ctx)
        y = numpy_ctx.asarray([1.0, -1.0, 2.0])
        np.testing.assert_allclose(op.adjoint_apply(y), op.rapply(y))


# ===========================================================================
# H — adjoint view, idempotent double application
# ===========================================================================
class TestAdjointView:
    def test_H_returns_linop(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        assert isinstance(op.H, sc.LinOp)

    def test_H_H_is_original(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, numpy_ctx,
        )
        # The double-adjoint view returns the original operator instance.
        assert op.H.H is op

    def test_adjoint_method_equals_H(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        assert op.adjoint() is op.H


# ===========================================================================
# Operator algebra dunders
# ===========================================================================
class TestAlgebraDunders:
    def _id(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        return sc.IdentityLinOp(X, numpy_ctx)

    def test_add_two_linops(self, numpy_ctx):
        a = self._id(numpy_ctx)
        b = self._id(numpy_ctx)
        out = a + b
        assert isinstance(out, sc.LinOp)

    def test_add_returns_notimplemented_for_non_linop(self, numpy_ctx):
        a = self._id(numpy_ctx)
        assert a.__add__(42) is NotImplemented

    def test_radd_zero_returns_self(self, numpy_ctx):
        a = self._id(numpy_ctx)
        # Lets ``sum([a, b])`` work (starts from 0).
        assert (0 + a) is a

    def test_neg_yields_scaled_negative_one(self, numpy_ctx):
        a = self._id(numpy_ctx)
        out = -a
        assert isinstance(out, sc.ScaledLinOp)
        assert out.scalar == -1

    def test_sub_uses_make_sum(self, numpy_ctx):
        a = self._id(numpy_ctx)
        b = self._id(numpy_ctx)
        out = a - b
        assert isinstance(out, sc.LinOp)

    def test_rsub_zero_yields_negation(self, numpy_ctx):
        a = self._id(numpy_ctx)
        out = 0 - a
        assert isinstance(out, sc.ScaledLinOp)
        assert out.scalar == -1

    def test_mul_by_scalar_returns_scaled(self, numpy_ctx):
        a = self._id(numpy_ctx)
        out = a * 3.0
        assert isinstance(out, sc.ScaledLinOp)

    def test_rmul_by_scalar_returns_scaled(self, numpy_ctx):
        a = self._id(numpy_ctx)
        out = 3.0 * a
        assert isinstance(out, sc.ScaledLinOp)

    def test_mul_returns_notimplemented_for_non_scalar(self, numpy_ctx):
        a = self._id(numpy_ctx)
        assert a.__mul__([1.0, 2.0]) is NotImplemented

    def test_matmul_composes(self, numpy_ctx):
        a = self._id(numpy_ctx)
        b = self._id(numpy_ctx)
        out = a @ b
        # IdentityLinOp @ IdentityLinOp simplifies back to one of them.
        assert isinstance(out, sc.LinOp)

    def test_matmul_returns_notimplemented_for_non_linop(self, numpy_ctx):
        a = self._id(numpy_ctx)
        assert a.__matmul__([1.0, 2.0]) is NotImplemented


# ===========================================================================
# to_dense / to_matrix round-trip via apply
# ===========================================================================
class TestToDenseToMatrix:
    def test_to_dense_matches_action_on_basis(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        op = sc.DenseLinOp(A, X, Y, numpy_ctx)
        dense = op.to_dense()
        # Shape == cod + dom.
        assert dense.shape == (3, 2)
        np.testing.assert_allclose(dense, A)

    def test_to_matrix_shape_is_codomain_size_x_domain_size(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2, 2), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        A = numpy_ctx.asarray(np.arange(12.0).reshape(3, 2, 2))
        op = sc.DenseLinOp(A, X, Y, numpy_ctx)
        matrix = op.to_matrix()
        # (cod.size, dom.size) = (3, 4)
        assert matrix.shape == (3, 4)

    def test_to_sparse_default_raises(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        with pytest.raises(NotImplementedError):
            op.to_sparse()

    def test_default_to_matrix_uses_batched_vapply_path(self, numpy_ctx):
        """The default ``to_matrix`` materializes via a single batched
        ``vapply`` call (folded from test_to_dense.py)."""
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        dense = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        class BatchedMaterializationLinOp(sc.LinOp):
            def __init__(self):
                super().__init__(dom, cod, numpy_ctx)
                self.apply_calls = 0
                self.vapply_calls = 0

            def apply(self, x):
                self.apply_calls += 1
                return dense @ x

            def rapply(self, y):
                return dense.T @ y

            def vapply(self, xs):
                self.vapply_calls += 1
                return xs @ dense.T

            def tree_flatten(self):
                return (), (self.domain, self.codomain, self.ctx)

            @classmethod
            def tree_unflatten(cls, aux, children):
                return cls()

        op = BatchedMaterializationLinOp()
        matrix = op.to_matrix()

        assert op.vapply_calls == 1
        assert op.apply_calls == 0
        assert matrix.shape == (3, 2)
        np.testing.assert_allclose(matrix, dense)
        np.testing.assert_allclose(op.to_dense(), dense)

    def test_default_to_matrix_falls_back_when_batch_helpers_unavailable(
        self, numpy_ctx
    ):
        """When the domain cannot batch-(un)flatten, ``to_matrix`` falls back
        to a per-column Python loop (folded from test_to_dense.py)."""

        class NoBatchVectorSpace(sc.DenseCoordinateSpace):
            def flatten_batch(self, xs):
                raise NotImplementedError

            def unflatten_batch(self, vs):
                raise NotImplementedError

            def _convert(self, new_ctx):
                if new_ctx == self.ctx:
                    return self
                return NoBatchVectorSpace(self.shape, new_ctx)

        dom = NoBatchVectorSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        dense = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        class FallbackMaterializationLinOp(sc.LinOp):
            def __init__(self):
                super().__init__(dom, cod, numpy_ctx)
                self.apply_calls = 0
                self.vapply_calls = 0

            def apply(self, x):
                self.apply_calls += 1
                return dense @ x

            def rapply(self, y):
                return dense.T @ y

            def vapply(self, xs):
                self.vapply_calls += 1
                return super().vapply(xs)

            def tree_flatten(self):
                return (), (self.domain, self.codomain, self.ctx)

            @classmethod
            def tree_unflatten(cls, aux, children):
                return cls()

        op = FallbackMaterializationLinOp()
        with pytest.warns(RuntimeWarning, match="falling back to a Python loop"):
            matrix = op.to_matrix()

        assert op.vapply_calls == 0
        assert op.apply_calls == 2
        assert matrix.shape == (3, 2)
        np.testing.assert_allclose(matrix, dense)


# ===========================================================================
# .A native numerical representation (folded from test_to_dense.py)
# ===========================================================================
class TestNativeRepresentation:
    def test_default_A_raises_not_implemented(self, numpy_ctx):
        """The base ``.A`` property has no native representation by default."""
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        dense = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        op = sc.MatrixFreeLinOp(
            lambda x: dense @ x, lambda y: numpy_ctx.ops.transpose(dense) @ y, X, Y, numpy_ctx,
        )
        with pytest.raises(NotImplementedError, match="native numerical representation"):
            _ = op.A

    def test_custom_linop_can_define_A_representation(self, numpy_ctx):
        """A subclass may override ``.A`` to expose its own native data."""
        dom = sc.DenseCoordinateSpace((2,), numpy_ctx)
        cod = sc.DenseCoordinateSpace((3,), numpy_ctx)
        dense = numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        class CustomLinOp(sc.LinOp):
            @property
            def A(self):
                return {"backend": "custom", "data": dense}

            def apply(self, x):
                return dense @ x

            def rapply(self, y):
                return numpy_ctx.ops.transpose(dense) @ y

            def tree_flatten(self):
                return (), (self.domain, self.codomain, self.ctx)

            @classmethod
            def tree_unflatten(cls, aux, children):
                domain, codomain, ctx = aux
                return cls(domain, codomain, ctx)

        op = CustomLinOp(dom, cod, numpy_ctx)
        assert op.A["backend"] == "custom"
        assert op.A["data"] is dense


# ===========================================================================
# assert_domain / assert_codomain
# ===========================================================================
class TestAssertions:
    def test_assert_domain_accepts_valid(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        op.assert_domain(numpy_ctx.asarray([1.0, 2.0]))  # no raise

    def test_assert_domain_rejects_wrong_shape(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        with pytest.raises(sc.SpaceValidationError, match="Expected shape"):
            op.assert_domain(numpy_ctx.asarray([1.0, 2.0, 3.0]))

    def test_assert_codomain_rejects_wrong_shape(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        Y = sc.DenseCoordinateSpace((3,), numpy_ctx)
        op = sc.DenseLinOp(
            numpy_ctx.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), X, Y, numpy_ctx,
        )
        with pytest.raises(sc.SpaceValidationError, match="Expected shape"):
            op.assert_codomain(numpy_ctx.asarray([1.0, 2.0]))


# ===========================================================================
# __eq__: returns NotImplemented for non-LinOp (so == doesn't raise)
# ===========================================================================
class TestEquality:
    def test_eq_against_none_returns_false(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        # ``op == None`` must not raise.
        assert (op == None) is False  # noqa: E711

    def test_eq_against_arbitrary_object_returns_false(self, numpy_ctx):
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        assert (op == "linop") is False
        assert (op == 42) is False

    def test_op_in_list_works(self, numpy_ctx):
        """``op in [op, other]`` relies on ``__eq__`` not raising."""
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        op = sc.IdentityLinOp(X, numpy_ctx)
        assert op in [op]


# ===========================================================================
# is_hermitian default
# ===========================================================================
class TestIsHermitianDefault:
    def test_base_default_is_none(self, numpy_ctx):
        """``LinOp.is_hermitian`` default returns ``None`` (unknown)."""
        X = sc.DenseCoordinateSpace((2,), numpy_ctx)
        # MatrixFreeLinOp inherits the base default.
        op = sc.MatrixFreeLinOp(lambda x: x, lambda y: y, X, X, numpy_ctx)
        assert op.is_hermitian() is None
