# BrainEvent Sparse Formats: COO, CSR, And CSC

Use this reference when explicit sparse connectivity must be constructed, converted, inspected, or oriented. The skill body keeps only CSR as the representative sparse format.

## Format Routing

| Form | What it stores | Use when | Important boundary |
|---|---|---|---|
| COO triplets | Parallel `row`, `col`, and `data` arrays | Edges naturally arrive as `(row, column, value)` records or a sparse matrix is being assembled | BrainEvent does not expose a standalone COO matrix class; convert triplets with `coo2csr()` |
| `CSR` | Nonzero `data`, column `indices`, and row `indptr` | Row-wise access and normal `BinaryArray @ connectivity` forward propagation | Reorder `data` with the `order` returned by `coo2csr()` |
| `CSC` | Nonzero `data`, row `indices`, and column `indptr` | Column-wise access, transposed products, or efficient column slicing | Construct arrays in column order |

Official tutorial recommendation: use CSR with `BinaryArray` for most spiking-network applications.

## COO Triplets: Construction Input

COO is a flexible build-time layout, not a BrainEvent connectivity object.

```python
import brainevent
import jax.numpy as jnp

shape = (4, 5)
row = jnp.array([0, 1, 2, 3])
col = jnp.array([1, 3, 0, 4])
data = jnp.array([1.5, 2.0, 0.5, 3.0])

indptr, indices, order = brainevent.coo2csr(row, col, shape=shape)
csr = brainevent.CSR(
    (data[order], indices, indptr),
    shape=shape,
)
```

Use when: edges are assembled as coordinate records. Preserve `order`; it maps the original `data` into CSR storage order.

## CSR: Row-Compressed Connectivity

Construct CSR directly only when its arrays are already in row order.

```python
import brainevent
import jax.numpy as jnp

data = jnp.array([1.5, 2.0, 0.5, 3.0])
indices = jnp.array([1, 3, 0, 4])       # column indices
indptr = jnp.array([0, 1, 2, 3, 4])    # one boundary per row, plus the end

csr = brainevent.CSR(
    (data, indices, indptr),
    shape=(4, 5),
)
spikes = brainevent.BinaryArray([1, 0, 1, 0])
postsynaptic_input = spikes @ csr
```

Use when: the main contraction is row-oriented, especially presynaptic `BinaryArray @ CSR` forward propagation.

## CSC: Column-Compressed Connectivity

The same logical matrix can be stored by column.

```python
import brainevent
import jax.numpy as jnp

# Column-ordered values for nonzeros at (2,0), (0,1), (1,3), (3,4).
data = jnp.array([0.5, 1.5, 2.0, 3.0])
indices = jnp.array([2, 0, 1, 3])          # row indices
indptr = jnp.array([0, 1, 2, 2, 3, 4])    # one boundary per column, plus the end

csc = brainevent.CSC(
    (data, indices, indptr),
    shape=(4, 5),
)
spikes = brainevent.BinaryArray([1, 0, 1, 0])
postsynaptic_input = spikes @ csc
```

Use when: the access or contraction is column-oriented, the transpose path is central, or efficient column slicing is required.

## Relevant Sparse API Surface

From the sparse-data API category:

- `DataRepresentation` — shared data-representation base.
- `CSR` — compressed sparse row matrix.
- `CSC` — compressed sparse column matrix.

`coo2csr()` is listed under BrainEvent utility functions rather than the sparse-data class category.

## Sources

- Tutorial 2: Sparse Data Structures - CSR and CSC: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/02_sparse_matrices.html
- Sparse Matrix Data Structures API: https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html
- Utility Functions API: https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html
