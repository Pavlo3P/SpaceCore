# SpaceCore

SpaceCore is a lightweight backend-agnostic library for working with vector spaces and linear operators.

It provides a small set of abstractions for:

- backend-aware numerical operations
- contexts carrying backend and dtype information
- structured vector spaces
- structured linear operators
- conversion between compatible contexts

## Installation

Base install:

```bash
pip install spacecore
```

With JAX support:

```bash
pip install "spacecore[jax]"
```

* `spacecore[jax]`: installs optional JAX support.
* GPU users should install the appropriate CUDA-enabled JAX build first, following the official JAX installation guide.

## Main concepts

### `Context`

A `Context` specifies how objects are represented, in particular:

* backend (`NumPy`, `JAX`, etc.)
* dtype
* validation/conversion behavior

### `Space`

A `Space` describes the structure of objects space, for example:

* `VectorSpace` - Euclidean space
* `HermitianSpace` - space of Hermitian (symmetric) matrices 
* `ProductSpace` - Cartesian product of spaces

### `LinOp`

A `LinOp` represents a linear operator between spaces, for example:

* `DenseLinOp` - linear operator represented by dense matrix
* `SparseLinOp` - linear operator represented by sparse matrix
* `BlockDiagonalLinOp` - linear operator from $X_1 \times \dots \times X_k$ to $Y_1 \times \dots \times Y_k$
* `StackedLinOp` - linear operator from $X$ to $Y_1 \times \dots \times Y_k$
* `SumToSingleLinOp` - linear operator from $X_1 \times \dots \times X_k$ to $Y$

## Minimal example

```python
import numpy as np
import spacecore as sc

sc.set_context('numpy', dtype='float64')

X = sc.VectorSpace((3,))
Y = sc.VectorSpace((2,))

A = np.array(
    [[1.0, 2.0, 3.0],
     [0.0, 1.0, 0.0]]
)
linop = sc.DenseLinOp(
    A,
    dom=X,
    cod=Y,
)

x = X.ctx.asarray([1.0, 0.0, -1.0])
y = linop.apply(x)

print(y)
```

## Status

SpaceCore is currently experimental and under active development.
The public API may still evolve.

## Tutorials

See the Sphinx documentation under `docs/source/` for tutorials, design notes,
and API reference.

## Documentation

The documentation website is built with Sphinx from `docs/source`.

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Build the local HTML documentation:

```bash
sphinx-build -b html docs/source docs/build/html
```

## License

Apache License 2.0
