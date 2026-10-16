from __future__ import annotations

from functools import reduce

from collections.abc import Sequence
from typing import Any

import optree

from ._base import TreeLinOp
from .._algebra import _same_space_for_algebra
from .._base import LinOp
from ..._checks import checked_method
from ...contextual import Context
from ..._checks import CheckLevel
from ...space import TreeSpace



def _refold(row_def: Any, column_def: Any, blocks: Sequence[LinOp]) -> Any:
    """Refold row-major blocks into the row and column containers."""
    width = column_def.num_leaves
    return optree.tree_unflatten(
        row_def,
        [
            optree.tree_unflatten(column_def, list(blocks[row * width : (row + 1) * width]))
            for row in range(row_def.num_leaves)
        ],
    )


def _validate_blocks(blocks: Sequence[Any], owner: str) -> tuple[LinOp, ...]:
    """Validate a nonempty block collection and its shared execution policy."""
    # Non-empty, all LinOps, one context and one check level.
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
        if not first.ctx.same_math(block.ctx):
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
    # Left fold with the space's (batched) addition.
    return reduce(space.add_batch if batched else space.add, values)


class BlockDiagonalLinOp(TreeLinOp[TreeSpace, TreeSpace]):
    r"""
    Independent blocks acting on the leaves of a direct-product tree.

    ``BlockDiagonalLinOp(blocks)`` infers both tree spaces from the blocks; the tree of
    ``blocks`` is the element structure. This is a direct product, not a Kronecker product.

    Parameters
    ----------
    blocks : tree of LinOp
        Block tree, each leaf mapping ``X_i`` to ``Y_i``.
    ctx : Context, str, or None, optional
        Backend context; resolved from the blocks when omitted.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def __init__(
        self,
        blocks: Any,
        ctx: Context | str | None = None,
        check_level: CheckLevel | None = None,
    ) -> None:
        # Flatten the blocks and build both tree spaces from them.
        leaves, treedef = optree.tree_flatten(blocks)
        block_parts = _validate_blocks(leaves, type(self).__name__)
        dom = TreeSpace(treedef, tuple(block.domain for block in block_parts), ctx=ctx)
        cod = TreeSpace(treedef, tuple(block.codomain for block in block_parts), ctx=ctx)
        super().__init__(dom, cod, block_parts, ctx, check_level=check_level)

    def _with_blocks(self, parts: Sequence[LinOp], ctx: Any = None) -> BlockDiagonalLinOp:
        # Rebuild in this operator's tree layout from new leaf blocks.
        return BlockDiagonalLinOp(optree.tree_unflatten(self.dom.treedef, list(parts)), ctx or self.ctx)

    def tree_flatten(self):
        return self.parts, (self.dom.treedef, self.ctx)

    @classmethod
    def tree_unflatten(cls, aux, children):
        treedef, ctx = aux
        return cls(optree.tree_unflatten(treedef, list(children)), ctx)

    def _convert(self, new_ctx: Context) -> BlockDiagonalLinOp:
        return self._with_blocks([op.convert(new_ctx) for op in self.parts], new_ctx)

    def _check_layout(self) -> None:
        """Check that each block maps the corresponding pair of tree leaves."""
        # One block per leaf, each matching its domain and codomain leaves.
        if not isinstance(self.dom, TreeSpace) or not isinstance(self.cod, TreeSpace):
            raise TypeError("BlockDiagonalLinOp expects dom and cod to be TreeSpace.")
        if len(self.parts) != self.dom.arity or len(self.parts) != self.cod.arity:
            raise ValueError("Number of blocks must match domain and codomain tree arity.")

        for index, block in enumerate(self.parts):
            if not _same_space_for_algebra(block.domain, self.dom.leaf_spaces[index]):
                raise TypeError(f"Block {index} has an incompatible domain leaf.")
            if not _same_space_for_algebra(block.codomain, self.cod.leaf_spaces[index]):
                raise TypeError(f"Block {index} has an incompatible codomain leaf.")

    def _apply_core(self, x: Any) -> Any:
        x_parts = self.dom._components(x)
        y_parts = tuple(p._apply_core(xi) for p, xi in zip(self.parts, x_parts))
        return self.cod._from_components(y_parts)

    def _rapply_core(self, y: Any) -> Any:
        y_parts = self.cod._components(y)
        x_parts = tuple(p._rapply_core(yi) for p, yi in zip(self.parts, y_parts))
        return self.dom._from_components(x_parts)

    @checked_method(
        in_space="domain", out_space="codomain", in_batched=True, out_batched=True
    )
    def vapply(self, x: Any) -> Any:
        """Apply each block over a tree of leading-axis batches."""
        return self._vapply_core(x)

    def _vapply_core(self, x: Any) -> Any:
        x_parts = self.dom._components(x)
        y_parts = tuple(p._vapply_core(xi) for p, xi in zip(self.parts, x_parts))
        return self.cod._from_components(y_parts)

    @checked_method(
        in_space="codomain", out_space="domain", in_batched=True, out_batched=True
    )
    def rvapply(self, y: Any) -> Any:
        """Apply each metric adjoint over a tree of leading-axis batches."""
        return self._rvapply_core(y)

    def _rvapply_core(self, y: Any) -> Any:
        y_parts = self.cod._components(y)
        x_parts = tuple(p._rvapply_core(yi) for p, yi in zip(self.parts, y_parts))
        return self.dom._from_components(x_parts)

    @property
    def H(self) -> BlockDiagonalLinOp:
        """Return a block-diagonal adjoint with every block replaced by ``A_i.H``."""
        # The adjoint is block diagonal in the blocks' adjoints; cache both ways.
        view = getattr(self, "_adjoint_view", None)
        if view is None:
            view = self._with_blocks([block.H for block in self.parts])
            self._adjoint_view = view
            view._adjoint_view = self
        return view

    def fuse(self, *, materialize: bool = False) -> BlockDiagonalLinOp:
        """Fuse each block (ADR-021), preserving the tree layout and context."""
        return self._with_blocks([op.fuse(materialize=materialize) for op in self.parts])

    @classmethod
    def from_operators(cls, parts: Sequence[LinOp]) -> BlockDiagonalLinOp:
        """Build a tuple-structured block-diagonal operator."""
        return cls(tuple(parts))

def _endpoint(container: Any, spaces: tuple[Any, ...], ctx: Any) -> Any:
    """Return the endpoint space for a row or column container of blocks."""
    # A plain one-element list or tuple is a single space; any other container is a tree.
    if type(container) in (list, tuple) and len(spaces) == 1:
        return spaces[0]
    return TreeSpace(_container_def(container), spaces, ctx=ctx)


def _container_def(container: Any) -> Any:
    """Return the treedef of ``container`` with each element as one leaf."""
    return optree.tree_structure(container, is_leaf=lambda node: node is not container)


def _split(space: Any, x: Any) -> tuple[Any, ...]:
    """Return the components of ``x``: the tree leaves, or ``(x,)`` for a single space."""
    return space._components(x) if isinstance(space, TreeSpace) else (x,)


def _join(space: Any, parts: tuple[Any, ...]) -> Any:
    """Inverse of :func:`_split`."""
    return space._from_components(parts) if isinstance(space, TreeSpace) else parts[0]


def _leaf_spaces(space: Any) -> tuple[Any, ...]:
    return space.leaf_spaces if isinstance(space, TreeSpace) else (space,)


class BlockMatrixLinOp(TreeLinOp[Any, Any]):
    r"""
    Block matrix over direct products: ``y_i = sum_j A_ij x_j`` for blocks ``A_ij : X_j -> Y_i``.

    The container of rows is the codomain's tree and the container of each row is the
    domain's tree, so a NamedTuple of rows gives NamedTuple codomain elements. A side
    given as a plain one-element list or tuple is a single space, not a one-leaf tree:
    ``BlockMatrixLinOp([[A1], [A2]])`` stacks ``x -> (A1 x, A2 x)`` and
    ``BlockMatrixLinOp([[A1, A2]])`` sums ``(x1, x2) -> A1 x1 + A2 x2``; each is the
    other's adjoint when the blocks are replaced by theirs.

    Parameters
    ----------
    block_rows : sequence of sequences of LinOp
        Rectangular block matrix; rows share codomains and columns share domains.
    check_level : {"none", "cheap", "standard", "strict"}, optional
        Validation level; the ambient default when omitted.
    """

    def __init__(
        self,
        block_rows: Sequence[Sequence[LinOp]],
        check_level: CheckLevel | None = None,
    ) -> None:
        # Normalize to a rectangular tuple of rows.
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

        # Validate every block, then check rows share codomains and columns share domains.
        flat_blocks = _validate_blocks(
            tuple(block for row in normalized_rows for block in row), type(self).__name__
        )

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

        # Domain leaves come from the columns, codomain leaves from the rows; the
        # containers give the trees.
        ctx = flat_blocks[0].ctx
        dom = _endpoint(rows[0], tuple(block.domain for block in normalized_rows[0]), ctx)
        cod = _endpoint(block_rows, tuple(row[0].codomain for row in normalized_rows), ctx)
        self._row_def = _container_def(block_rows)
        self._column_def = _container_def(rows[0])
        self._row_count = len(normalized_rows)
        self._column_count = column_count
        super().__init__(dom, cod, flat_blocks, ctx, check_level=check_level)
        self.block_rows = tuple(
            self.parts[index * column_count : (index + 1) * column_count]
            for index in range(self._row_count)
        )

    def _check_layout(self) -> None:
        """Check the row and column incidence against inferred endpoint leaves."""
        dom_leaves, cod_leaves = _leaf_spaces(self.dom), _leaf_spaces(self.cod)
        if len(dom_leaves) != self._column_count or len(cod_leaves) != self._row_count:
            raise ValueError("BlockMatrixLinOp inferred endpoint arity mismatch.")
        for row in range(self._row_count):
            for column in range(self._column_count):
                block = self.parts[row * self._column_count + column]
                if not _same_space_for_algebra(block.domain, dom_leaves[column]):
                    raise ValueError(f"Block ({row}, {column}) has an incompatible domain.")
                if not _same_space_for_algebra(block.codomain, cod_leaves[row]):
                    raise ValueError(f"Block ({row}, {column}) has an incompatible codomain.")

    def _apply_core(self, x: Any) -> Any:
        # y_i = sum_j A_ij x_j, row by row.
        x_parts = _split(self.dom, x)
        y_parts = tuple(
            _sum_values(codomain, tuple(
                self._apply_parts[row * self._column_count + column](x_parts[column])
                for column in range(self._column_count)
            ), batched=False)
            for row, codomain in enumerate(_leaf_spaces(self.cod))
        )
        return _join(self.cod, y_parts)

    def _rapply_core(self, y: Any) -> Any:
        # x_j = sum_i A_ij* y_i, column by column.
        y_parts = _split(self.cod, y)
        x_parts = tuple(
            _sum_values(domain, tuple(
                self._rapply_parts[row * self._column_count + column](y_parts[row])
                for row in range(self._row_count)
            ), batched=False)
            for column, domain in enumerate(_leaf_spaces(self.dom))
        )
        return _join(self.dom, x_parts)

    def _vapply_core(self, x: Any) -> Any:
        x_parts = _split(self.dom, x)
        y_parts = tuple(
            _sum_values(codomain, tuple(
                block._vapply_core(x_part) for block, x_part in zip(row, x_parts)
            ), batched=True)
            for row, codomain in zip(self.block_rows, _leaf_spaces(self.cod))
        )
        return _join(self.cod, y_parts)

    def _rvapply_core(self, y: Any) -> Any:
        y_parts = _split(self.cod, y)
        x_parts = tuple(
            _sum_values(domain, tuple(
                self.block_rows[row][column]._rvapply_core(y_parts[row])
                for row in range(self._row_count)
            ), batched=True)
            for column, domain in enumerate(_leaf_spaces(self.dom))
        )
        return _join(self.dom, x_parts)

    def is_hermitian(self) -> bool | None:
        """Return ``True`` when the block layout is square, ``A_ij# == A_ji`` and every diagonal block is self-adjoint."""
        n = self._row_count
        if n != self._column_count or self.dom != self.cod:
            return None
        rows = self.block_rows
        if all(rows[i][i].is_hermitian() is True for i in range(n)) and all(
            rows[i][j].H == rows[j][i] for i in range(n) for j in range(i + 1, n)
        ):
            return True
        return None

    @property
    def H(self) -> BlockMatrixLinOp:
        """Transpose the block layout and replace every block by its adjoint."""
        view = getattr(self, "_adjoint_view", None)
        if view is None:
            # The adjoint is the transposed matrix of block adjoints; cache both ways.
            transposed = [
                self.block_rows[row][column].H
                for column in range(self._column_count)
                for row in range(self._row_count)
            ]
            view = BlockMatrixLinOp(_refold(self._column_def, self._row_def, transposed))
            self._adjoint_view = view
            view._adjoint_view = self
        return view

    def fuse(self, *, materialize: bool = False) -> BlockMatrixLinOp:
        """Fuse each block (ADR-021), preserving the block layout."""
        fused = [op.fuse(materialize=materialize) for op in self.parts]
        return BlockMatrixLinOp(_refold(self._row_def, self._column_def, fused))

    def tree_flatten(self):
        """Flatten row-major blocks for JAX pytree registration."""
        return self.parts, (self._row_def, self._column_def)

    @classmethod
    def tree_unflatten(cls, aux: Any, children: Sequence[LinOp]) -> BlockMatrixLinOp:
        """Rebuild a block matrix from row-major pytree children."""
        row_def, column_def = aux
        return cls(_refold(row_def, column_def, children))

    @classmethod
    def from_operators(cls, parts: Sequence[LinOp]) -> BlockMatrixLinOp:
        """Build a one-row block matrix, ``(x_1, ..., x_k) -> sum_j A_j x_j``."""
        return cls((tuple(parts),))

    @classmethod
    def stack(cls, parts: Sequence[LinOp]) -> BlockMatrixLinOp:
        """Build a one-column block matrix, ``x -> (A_1 x, ..., A_k x)``."""
        if not parts:
            raise ValueError("Parts must be non-empty.")
        return cls(tuple((op,) for op in parts))

    def _convert(self, new_ctx: Context) -> BlockMatrixLinOp:
        """Convert every block to ``new_ctx`` and preserve the block layout."""
        converted = [block.convert(new_ctx) for block in self.parts]
        return BlockMatrixLinOp(_refold(self._row_def, self._column_def, converted))
