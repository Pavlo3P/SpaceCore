"""Reusable deterministic generators for SpaceCore tests."""

from ._arrays import DEFAULT_DENSE_SHAPES, Field, dense_array_case, dense_array_cases
from ._contexts import ContextCase, context_cases
from ._hermitian import hermitian_case, hermitian_cases
from ._metrics import spd_metric_case, spd_metric_cases
from ._params import BatchCase, batch_cases, batch_params, check_level_params, context_params
from ._protocol import GeneratedCase
from ._seed import DEFAULT_SEED, resolve_rng, seeded_rng
from ._trees import TreeKind, tree_space_case, tree_space_cases
from .functionals import FunctionalCase, NUMPY_FUNCTIONAL_DTYPES, functional_cases
from .linops import (
    NUMPY_LINOP_DTYPES,
    LinOpCase,
    backend_linop_cases,
    dense_linop_case,
    diagonal_linop_case,
    linop_cases,
    matrix_free_linop_case,
    sparse_linop_case,
    tree_linop_cases,
)
from .spaces import (
    NUMPY_SPACE_DTYPES,
    MatrixInnerProduct,
    SpaceCase,
    dense_coordinate_space_cases,
    dense_vector_space_cases,
    inner_product_space_cases,
    jordan_space_cases,
    mixed_jordan_tree_case,
    tree_space_generated_cases,
    vector_space_law_cases,
)


__all__ = [
    "DEFAULT_DENSE_SHAPES",
    "DEFAULT_SEED",
    "BatchCase",
    "ContextCase",
    "Field",
    "FunctionalCase",
    "GeneratedCase",
    "MatrixInnerProduct",
    "LinOpCase",
    "NUMPY_LINOP_DTYPES",
    "NUMPY_SPACE_DTYPES",
    "NUMPY_FUNCTIONAL_DTYPES",
    "SpaceCase",
    "TreeKind",
    "batch_cases",
    "batch_params",
    "backend_linop_cases",
    "check_level_params",
    "context_cases",
    "context_params",
    "dense_array_case",
    "dense_array_cases",
    "dense_coordinate_space_cases",
    "dense_vector_space_cases",
    "dense_linop_case",
    "diagonal_linop_case",
    "hermitian_case",
    "hermitian_cases",
    "functional_cases",
    "inner_product_space_cases",
    "jordan_space_cases",
    "linop_cases",
    "matrix_free_linop_case",
    "mixed_jordan_tree_case",
    "resolve_rng",
    "seeded_rng",
    "spd_metric_case",
    "spd_metric_cases",
    "sparse_linop_case",
    "tree_space_case",
    "tree_space_cases",
    "tree_space_generated_cases",
    "tree_linop_cases",
    "vector_space_law_cases",
]
