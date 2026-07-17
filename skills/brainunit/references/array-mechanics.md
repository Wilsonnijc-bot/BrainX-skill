# Array Mechanics

Use this reference for array metadata, indexing, functional updates, shape and axis transformations, joins and splits, repetition, named-axis transformations, and backend conversion of an existing `Quantity`.

## Contents

- [Metadata and identity](#metadata-and-identity)
- [Indexing and functional updates](#indexing-and-functional-updates)
- [Quantity methods](#quantity-methods)
- [Functional structural API](#functional-structural-api)
- [Named-axis transformations](#named-axis-transformations)
- [Backend conversion](#backend-conversion)
- [Source-backed gotchas](#source-backed-gotchas)

## Metadata And Identity

| Property | Official meaning |
|---|---|
| `q.shape` | Shape of the mantissa array |
| `q.dtype` | Data type of the mantissa |
| `q.backend` | One of `'numpy'`, `'jax'`, `'cupy'`, `'torch'`, `'dask'`, or `'ndonnx'` |
| `q.itemsize` | Bytes in one mantissa element |
| `q.nbytes` | Total bytes consumed by the mantissa array |
| `q.strides` | Byte steps in each dimension, mirroring `numpy.ndarray.strides` |
| `q.flat` | 1-D iterator over mantissa elements with units preserved |
| `q.mT` | Transpose of the final two dimensions; requires at least 2-D |

Use `u.math.ndim(x)`, `u.math.shape(x)`, and `u.math.size(x)` when a functional query is preferable to a property lookup.

## Indexing And Functional Updates

Ordinary indexing preserves the unit. `q.at` provides a functionally pure update interface:

```python
import brainunit as u
import jax.numpy as jnp

x = jnp.arange(5.0) * u.mV
y = x.at[2].add(10 * u.mV)
selected = x.at[2].get()
```

The documented update mappings are:

| Functional form | In-place equivalent |
|---|---|
| `x = x.at[idx].set(y)` | `x[idx] = y` |
| `x = x.at[idx].add(y)` | `x[idx] += y` |
| `x = x.at[idx].multiply(y)` | `x[idx] *= y` |
| `x = x.at[idx].divide(y)` | `x[idx] /= y` |
| `x = x.at[idx].power(y)` | `x[idx] **= y` |
| `x = x.at[idx].min(y)` | `x[idx] = minimum(x[idx], y)` |
| `x = x.at[idx].max(y)` | `x[idx] = maximum(x[idx], y)` |
| `x = x.at[idx].apply(ufunc)` | `ufunc.at(x, idx)` |
| `x.at[idx].get()` | `x[idx]` |

None of these expressions modifies the original object; each returns a modified copy. Inside `jit()`, assignments such as `x = x.at[idx].set(y)` are documented as being applied in place.

## Quantity Methods

| Exact signature | Structural effect |
|---|---|
| `reshape(shape, order='C')` | New shape, same data |
| `flatten()` | 1-D copy |
| `squeeze(axis=None)` | Removes length-one axes |
| `expand_dims(axis)` | Inserts axes |
| `unsqueeze(axis)` | PyTorch-style alias for `expand_dims()` |
| `transpose(*axes)` | Permutes axes |
| `swapaxes(axis1, axis2)` | Interchanges two axes |
| `repeat(repeats, axis=None)` | Repeats elements |
| `tile(reps)` | Repeats the full array pattern |
| `split(indices_or_sections, axis=0)` | Returns multiple sub-arrays |
| `take(indices, axis=None, mode=None, unique_indices=False, indices_are_sorted=False, fill_value=None)` | Selects elements along an axis |
| `sort(axis=-1, stable=True, order=None)` | Sorts along an axis |
| `diagonal(offset=0, axis1=0, axis2=1)` | Selects diagonals and preserves units |
| `trace(offset=0, axis1=0, axis2=1)` | Sums diagonals and preserves units |
| `astype(dtype)` | Casts the mantissa dtype |

## Functional Structural API

Use these `brainunit.math` functions when composing transformations without binding the operation to a method:

| Family | Exact APIs |
|---|---|
| Join | `concatenate(arrays, axis=0, dtype=None, **kwargs)`, `stack(arrays, axis=0, dtype=None, **kwargs)`, `block(arrays, **kwargs)`, `append(arr, values, axis=None, **kwargs)` |
| Stack variants | `row_stack`, `vstack`, `hstack`, `dstack`, `column_stack` |
| Split | `split`, `array_split`, `dsplit`, `hsplit`, `vsplit` |
| Minimum rank | `atleast_1d`, `atleast_2d`, `atleast_3d` |
| Broadcast | `broadcast_arrays(*args, **kwargs)`, `broadcast_to(array, shape, **kwargs)` |
| Shape | `reshape`, `moveaxis`, `transpose`, `swapaxes`, `expand_dims`, `squeeze`, `flatten`, `unflatten` |
| Repetition | `tile(A, reps, **kwargs)`, `repeat(a, repeats, axis=None, total_repeat_length=None, **kwargs)` |
| Selection | `take`, `gather`, `choose`, `compress`, `extract` |

The main API categorizes these as functions that keep units. Join examples on the generated pages use values carrying the same unit.

```python
a = [1, 2] * u.second
b = [3, 4] * u.second
joined = u.math.concatenate([a, b])
rows = u.math.stack([a, b])
parts = u.math.array_split(joined, 2)
```

## Named-Axis Transformations

`einrearrange(x, pattern, **axes_lengths)` covers transpose, reshape, squeeze, unsqueeze, stack, and concatenate. Composition uses C-order enumeration.

```python
channels_last = u.math.einrearrange(x, 'b c h w -> b h w c')
flat = u.math.einrearrange(x, 'b h w c -> (b h w c)')
split_batch = u.math.einrearrange(
    x,
    '(b1 b2) h w c -> b1 b2 h w c',
    b1=2,
)
```

`einrepeat(x, pattern, **axes_lengths)` covers repeat, tile, and broadcast-like patterns:

```python
expanded = u.math.einrepeat(x, 'h w c -> h new_axis w c', new_axis=5)
tiled = u.math.einrepeat(x, 'h w c -> (2 h) (2 w) c')
```

Use `einshape(x, pattern)` to map named axes to lengths. Use `einreduce()` and `einsum()` through `math-function-library.md` when axes are reduced or contracted.

## Backend Conversion

All `Quantity` backend methods retain the wrapper and attached unit:

| Exact signature | Target mantissa |
|---|---|
| `to_jax()` | JAX `Array` |
| `to_numpy()` | `numpy.ndarray` |
| `to_cupy(*, device=None)` | `cupy.ndarray` |
| `to_torch(*, device=None, dtype=None)` | `torch.Tensor` |
| `to_dask(*, chunks='auto')` | `dask.array.Array` |
| `to_ndonnx()` | Symbolic `ndonnx.Array` |

`u.math.as_numpy(x)` is different: for a `Quantity`, it returns the underlying mantissa in the current unit scale as a NumPy array.

## Source-Backed Gotchas

- `q.at` supports NumPy, JAX, CuPy, Torch, and Dask mantissas. The ndonnx backend raises `BackendError`; the API recommends `.to_numpy()` first.
- Repeated-index update semantics differ by backend. JAX, NumPy, and CuPy accumulate documented operations; some Torch and Dask operations use last-write-wins behavior.
- Dask functional updates support slice, scalar integer, 1-D integer, and boolean-mask indices; multidimensional fancy integer indexing raises `NotImplementedError`.
- `mT` transposes only the last two dimensions and requires at least two dimensions.
- `u.math.as_numpy(q)` drops the `Quantity` wrapper; `q.to_numpy()` keeps it.
- In Einstein patterns, axis order inside parentheses changes element order. A literal `1` or `()` creates a length-one axis, and either can be removed in the inverse pattern.

## Sources Mirrored

- https://brainunit.readthedocs.io/apis/generated/brainunit.Quantity.html
- https://brainunit.readthedocs.io/apis/brainunit.math.html
- https://brainunit.readthedocs.io/unit_operations/einstein_operations.html
