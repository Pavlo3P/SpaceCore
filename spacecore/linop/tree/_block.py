from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import optree

from ._base import TreeLinOp
from .._algebra import _same_space_for_algebra
from .._base import LinOp
from ..._checks import checked_method
from ..._contextual._bound import _same_math_context
from ...backend import Context, jax_pytree_class
from ...kernels import dispatch, should_consult_dispatch
from ...space import TreeSpace

# ADR-016 approves the block-diagonal apply as a dispatch call site. The
# per-part loop is the ``generic`` fallback; the ``block-diagonal-uniform-dense-
# batched`` spec routes here when dispatch is on and the blocks are uniform
# flat-dense. The block operators (not just their bound cores) are passed so a
# spec can inspect block structure; ``parts[i]._apply_core`` is exactly the
# bound core in ``self._apply_parts[i]``, so the generic stays byte-identical.
# The ``should_consult_dispatch`` guard keeps the default path untouched.
_BLOCK_DIAGONAL_APPLY_KEY = "linop.block_diagonal.apply"
_BLOCK_DIAGONAL_RAPPLY_KEY = "linop.block_diagonal.rapply"


def _block_diagonal_apply(parts: Any, x_parts: Any) -> tuple[Any, ...]:
    """Apply each block core to its own component (generic block-diagonal apply)."""
    return tuple(p._apply_core(xi) for p, xi in zip(parts, x_parts))


def _block_diagonal_rapply(parts: Any, y_parts: Any) -> tuple[Any, ...]:
    """Apply each block adjoint core to its own component (generic block-diagonal rapply)."""
    return tuple(p._rapply_core(yi) for p, yi in zip(parts, y_parts))


def _validate_blocks(blocks: Sequence[Any], owner: str) -> tuple[LinOp, ...]:
    """Validate a nonempty block collection and its shared execution policy."""
    validated = tuple(blocks)
    if not validated:
        raise ValueError(f"{owner} requires at least one block.")
    for index, block in enumerate(validated):
        if not isinstance(block, LinOp):
            raise TypeError(
                f"{owner} requires every block to be a LinOp; "
                f"block {index} is {type(block).__name__}."
            )

    first = validated[0]
    for index, block in enumerate(validated[1:], start=1):
        if not _same_math_context(first.ctx, block.ctx):
            raise ValueError(
                f"All {owner} blocks must have the same mathematical context; "
                f"block 0 has {first.ctx!r}, block {index} has {block.ctx!r}."
            )
        if first.check_level != block.check_level:
            raise ValueError(
                f"All {owner} blocks must have the same check policy; "
                f"block 0 uses {first.check_level!r}, "
                f"block {index} uses {block.check_level!r}."
            )
    return validated


def _sum_values(space: Any, values: Sequence[Any], *, batched: bool) -> Any:
    """Sum a nonempty sequence through the owning space's vector operation."""
    iterator = iter(values)
    result = next(iterator)
    add = space.add_batch if batched else space.add
    for value in iterator:
        result = add(result, value)
    return result


@jax_pytree_class
class BlockDiagonalLinOp(TreeLinOp[TreeSpace, TreeSpace]):
    r"""
    Represent independent blocks over a finite direct-product tree.

    ``BlockDiagonalLinOp(blocks)`` infers matching domain and codomain
    :class:`TreeSpace` objects from the block domains and codomains. The Python
    tree structure of ``blocks`` is also the element structure on both sides.
    This is a direct-product operator, not a tensor or Kronecker product.

    Parameters
    ----------
    blocks : tree of LinOp or TreeSpace
        Nonempty block tree (one-argument form). Each leaf ``A_i`` maps the
        corresponding domain leaf ``X_i`` to codomain leaf ``Y_i``. In the legacy
        four-argument form this is instead the domain :class:`TreeSpace`.
    cod : TreeSpace or None, optional
        Codomain tree for the legacy ``(dom, cod, parts, ctx)`` form; inferred
        from the blocks otherwise.
    parts : sequence of LinOp or None, optional
        Block operators for the legacy form; inferred from ``blocks`` otherwise.
    ctx : Context, str, or None, optional
        Backend context specification. Default is resolved from the blocks.

    Notes
    -----
    The legacy ``BlockDiagonalLinOp(dom, cod, blocks, ctx)`` form remains
    accepted so callers can provide distinct custom domain and codomain tree
    structures. New code should use the inferred one-argument form.
    """

    def __init__(
        self,
        blocks: Any,
        cod: TreeSpace | None = None,
        parts: Sequence[LinOp] | None = None,
        ctx: Context | str | None = None,
    ) -> None:
        if isinstance(blocks, TreeSpace):
            dom = blocks
            if not isinstance(cod, TreeSpace):
                raise TypeError("Legacy BlockDiagonalLinOp construction requires a TreeSpace cod.")
            if parts is None:
                raise TypeError("Legacy BlockDiagonalLinOp construction requires component blocks.")
            block_parts = _validate_blocks(parts, type(self).__name__)
        else:
            if cod is not None or parts is not None:
                raise TypeError(
                    "BlockDiagonalLinOp(blocks) accepts only a block tree; "
                    "use the legacy (dom, cod, blocks, ctx) form for explicit layouts."
                )
            leaves, treedef = optree.tree_flatten(blocks)
            block_parts = _validate_blocks(leaves, type(self).__name__)
            dom = TreeSpace(treedef, tuple(block.domain for block in block_parts), ctx=ctx)
            cod = TreeSpace(treedef, tuple(block.codomain for block in block_parts), ctx=ctx)

        super().__init__(dom, cod, block_parts, ctx)

    def _check_layout(self) -> None:
        """Check that each block maps the corresponding pair of tree leaves."""
        if not isinstance(self.dom, TreeSpace) or not isinstance(self.cod, TreeSpace):
            raise TypeError("BlockDiagonalLinOp expects dom and cod to be TreeSpace.")
        if len(self.parts) != self.dom.arity or len(self.parts) != self.cod.arity:
            raise ValueError("Number of blocks must match domain and codomain tree arity.")

        for index, block in enumerate(self.parts):
            if not _same_space_for_algebra(block.domain, self.dom.leaf_spaces[index]):
                raise TypeError(f"Block {index} has an incompatible domain leaf.")
            if not _same_space_for_algebra(block.codomain, self.cod.leaf_spaces[index]):
                raise TypeError(f"Block {index} has an incompatible codomain leaf.")

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Apply each block to the matching direct-product component."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        x_parts = self.dom._components(x)
        if should_consult_dispatch(self.ctx):
            y_parts = dispatch(
                _BLOCK_DIAGONAL_APPLY_KEY,
                self.parts,
                x_parts,
                generic=_block_diagonal_apply,
                ctx=self.ctx,
            )
        else:
            y_parts = _block_diagonal_apply(self.parts, x_parts)
        return self.cod._from_components(y_parts)

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Apply each block's metric adjoint to the matching component."""
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        y_parts = self.cod._components(y)
        if should_consult_dispatch(self.ctx):
            x_parts = dispatch(
                _BLOCK_DIAGONAL_RAPPLY_KEY,
                self.parts,
                y_parts,
                generic=_block_diagonal_rapply,
                ctx=self.ctx,
            )
        else:
            x_parts = _block_diagonal_rapply(self.parts, y_parts)
        return self.dom._from_components(x_parts)

    @checked_method(
        in_space="domain", out_space="codomain", in_batched=True, out_batched=True
    )
    def vapply(self, x: Any) -> Any:
        """Apply each block over a tree of leading-axis batches."""
        x_parts = self.dom._components(x)
        y_parts = tuple(op.vapply(xi) for op, xi in zip(self.parts, x_parts))
        return self.cod._from_components(y_parts)

    @checked_method(
        in_space="codomain", out_space="domain", in_batched=True, out_batched=True
    )
    def rvapply(self, y: Any) -> Any:
        """Apply each metric adjoint over a tree of leading-axis batches."""
        y_parts = self.cod._components(y)
        x_parts = tuple(op.rvapply(yi) for op, yi in zip(self.parts, y_parts))
        return self.dom._from_components(x_parts)

    @property
    def H(self) -> BlockDiagonalLinOp:
        """Return a block-diagonal adjoint with every block replaced by ``A_i.H``."""
        view = getattr(self, "_adjoint_view", None)
        if view is None:
            view = BlockDiagonalLinOp(
                self.codomain,
                self.domain,
                tuple(block.H for block in self.parts),
                self.ctx,
            )
            self._adjoint_view = view
            view._adjoint_view = self
        return view

    @classmethod
    def from_operators(cls, parts: Sequence[LinOp]) -> BlockDiagonalLinOp:
        """Build a tuple-structured block-diagonal operator."""
        return cls(tuple(parts))

    def _convert(self, new_ctx: Context) -> BlockDiagonalLinOp:
        """Convert spaces and blocks while retaining explicit tree layouts."""
        return BlockDiagonalLinOp(
            self.dom.convert(new_ctx),
            self.cod.convert(new_ctx),
            tuple(op.convert(new_ctx) for op in self.parts),
            new_ctx,
        )


@jax_pytree_class
class BlockMatrixLinOp(TreeLinOp[TreeSpace, TreeSpace]):
    r"""
    Represent a rectangular matrix of blocks over direct products.

    For blocks ``A_ij : X_j -> Y_i``, the operator maps
    ``X_0 x ... x X_n`` to ``Y_0 x ... x Y_m`` and computes
    ``y_i = sum_j A_ij x_j``. These are direct-product blocks, not tensor or
    Kronecker products.

    Parameters
    ----------
    block_rows : sequence of sequences of LinOp
        Nonempty rectangular block matrix. Blocks in one row must have
        compatible codomains, and blocks in one column must have compatible
        domains.
    """

    def __init__(self, block_rows: Sequence[Sequence[LinOp]]) -> None:
        if not isinstance(block_rows, Sequence) or isinstance(block_rows, (str, bytes)):
            raise TypeError("BlockMatrixLinOp block_rows must be a sequence of rows.")
        rows = tuple(block_rows)
        if not rows:
            raise ValueError("BlockMatrixLinOp requires at least one block row.")

        normalized_rows: list[tuple[Any, ...]] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError(f"BlockMatrixLinOp row {row_index} must be a sequence.")
            normalized_rows.append(tuple(row))
        if not normalized_rows[0]:
            raise ValueError("BlockMatrixLinOp rows must contain at least one block.")

        column_count = len(normalized_rows[0])
        for row_index, row in enumerate(normalized_rows[1:], start=1):
            if len(row) != column_count:
                raise ValueError(
                    "BlockMatrixLinOp requires a rectangular block structure; "
                    f"row 0 has {column_count} blocks but row {row_index} has {len(row)}."
                )

        flat_blocks = _validate_blocks(
            tuple(block for row in normalized_rows for block in row), type(self).__name__
        )
        normalized_rows = [
            flat_blocks[index * column_count : (index + 1) * column_count]
            for index in range(len(normalized_rows))
        ]

        for row_index, row in enumerate(normalized_rows):
            expected = row[0].codomain
            for column_index, block in enumerate(row[1:], start=1):
                if not _same_space_for_algebra(block.codomain, expected):
                    raise ValueError(
                        f"Block row {row_index} has incompatible codomains at columns "
                        f"0 and {column_index}."
                    )
        for column_index in range(column_count):
            expected = normalized_rows[0][column_index].domain
            for row_index in range(1, len(normalized_rows)):
                block = normalized_rows[row_index][column_index]
                if not _same_space_for_algebra(block.domain, expected):
                    raise ValueError(
                        f"Block column {column_index} has incompatible domains at rows "
                        f"0 and {row_index}."
                    )

        ctx = flat_blocks[0].ctx
        dom = TreeSpace.from_leaf_spaces(
            tuple(normalized_rows[0][column].domain for column in range(column_count)), ctx
        )
        cod = TreeSpace.from_leaf_spaces(tuple(row[0].codomain for row in normalized_rows), ctx)
        self._row_count = len(normalized_rows)
        self._column_count = column_count
        super().__init__(dom, cod, flat_blocks, ctx)
        self.block_rows = tuple(
            self.parts[index * column_count : (index + 1) * column_count]
            for index in range(self._row_count)
        )

    def _check_layout(self) -> None:
        """Check the row and column incidence against inferred tree leaves."""
        if len(self.parts) != self._row_count * self._column_count:
            raise ValueError("BlockMatrixLinOp block count does not match its rectangular shape.")
        if self.dom.arity != self._column_count or self.cod.arity != self._row_count:
            raise ValueError("BlockMatrixLinOp inferred TreeSpace arity mismatch.")
        for row in range(self._row_count):
            for column in range(self._column_count):
                block = self.parts[row * self._column_count + column]
                if not _same_space_for_algebra(block.domain, self.dom.leaf_spaces[column]):
                    raise ValueError(f"Block ({row}, {column}) has an incompatible domain.")
                if not _same_space_for_algebra(block.codomain, self.cod.leaf_spaces[row]):
                    raise ValueError(f"Block ({row}, {column}) has an incompatible codomain.")

    @checked_method(in_space="domain", out_space="codomain")
    def apply(self, x: Any) -> Any:
        """Apply the block matrix and sum each output row."""
        return self._apply_unchecked(x)

    def _apply_unchecked(self, x: Any) -> Any:
        x_parts = self.dom._components(x)
        y_parts = []
        for row_index, codomain in enumerate(self.cod.leaf_spaces):
            values = tuple(
                self._apply_parts[row_index * self._column_count + column](x_parts[column])
                for column in range(self._column_count)
            )
            y_parts.append(_sum_values(codomain, values, batched=False))
        return self.cod._from_components(tuple(y_parts))

    @checked_method(in_space="codomain", out_space="domain")
    def rapply(self, y: Any) -> Any:
        """Apply the metric-adjoint blocks and sum each transposed column."""
        return self._rapply_unchecked(y)

    def _rapply_unchecked(self, y: Any) -> Any:
        y_parts = self.cod._components(y)
        x_parts = []
        for column, domain in enumerate(self.dom.leaf_spaces):
            values = tuple(
                self._rapply_parts[row * self._column_count + column](y_parts[row])
                for row in range(self._row_count)
            )
            x_parts.append(_sum_values(domain, values, batched=False))
        return self.dom._from_components(tuple(x_parts))

    @checked_method(
        in_space="domain", out_space="codomain", in_batched=True, out_batched=True
    )
    def vapply(self, x: Any) -> Any:
        """Apply the block matrix over a tuple of leading-axis batches."""
        x_parts = self.dom._components(x)
        y_parts = []
        for row, codomain in zip(self.block_rows, self.cod.leaf_spaces):
            values = tuple(block.vapply(x_part) for block, x_part in zip(row, x_parts))
            y_parts.append(_sum_values(codomain, values, batched=True))
        return self.cod._from_components(tuple(y_parts))

    @checked_method(
        in_space="codomain", out_space="domain", in_batched=True, out_batched=True
    )
    def rvapply(self, y: Any) -> Any:
        """Apply the block metric adjoint over leading-axis batches."""
        y_parts = self.cod._components(y)
        x_parts = []
        for column, domain in enumerate(self.dom.leaf_spaces):
            values = tuple(
                self.block_rows[row][column].rvapply(y_parts[row])
                for row in range(self._row_count)
            )
            x_parts.append(_sum_values(domain, values, batched=True))
        return self.dom._from_components(tuple(x_parts))

    @property
    def H(self) -> BlockMatrixLinOp:
        """Transpose the block layout and replace every block by its adjoint."""
        view = getattr(self, "_adjoint_view", None)
        if view is None:
            rows = tuple(
                tuple(self.block_rows[row][column].H for row in range(self._row_count))
                for column in range(self._column_count)
            )
            view = BlockMatrixLinOp(rows)
            self._adjoint_view = view
            view._adjoint_view = self
        return view

    def tree_flatten(self):
        """Flatten row-major blocks for JAX pytree registration."""
        return self.parts, (self._row_count, self._column_count)

    @classmethod
    def tree_unflatten(cls, aux: Any, children: Sequence[LinOp]) -> BlockMatrixLinOp:
        """Rebuild a block matrix from row-major pytree children."""
        row_count, column_count = aux
        rows = tuple(
            tuple(children[row * column_count : (row + 1) * column_count])
            for row in range(row_count)
        )
        return cls(rows)

    @classmethod
    def from_operators(cls, parts: Sequence[LinOp]) -> BlockMatrixLinOp:
        """Build a one-row block matrix from a sequence of operators."""
        return cls((tuple(parts),))

    def _convert(self, new_ctx: Context) -> BlockMatrixLinOp:
        """Convert every block to ``new_ctx`` and preserve the matrix layout."""
        return BlockMatrixLinOp(
            tuple(tuple(block.convert(new_ctx) for block in row) for row in self.block_rows)
        )
