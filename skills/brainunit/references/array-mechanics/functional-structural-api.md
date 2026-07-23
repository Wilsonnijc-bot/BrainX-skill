# Functional Structural API

Open this second-level reference from `../array-mechanics.md` when selecting a `brainunit.math` function for joining, stacking, splitting, broadcasting, reshaping, repeating, or selecting values. The module API categorizes these as functions that keep units.

```python
import brainunit as u
import jax.numpy as jnp
```

Examples use concise `Quantity(..., "unit")` results; the exact displayed representation can vary by backend and BrainUnit version.

## Contents

- [Joining and stacking](#joining-and-stacking)
- [Splitting](#splitting)
- [Minimum rank and broadcasting](#minimum-rank-and-broadcasting)
- [Shape and axis operations](#shape-and-axis-operations)
- [Repetition](#repetition)
- [Selection](#selection)

## Joining And Stacking

| Exact signature | One-line description | Example & result |
|---|---|---|
| `concatenate(arrays, axis=0, dtype=None, **kwargs)` | Join a sequence of quantities or arrays along an existing axis. | <pre><code>a = [1, 2] * u.second<br>b = [3, 4] * u.second<br>u.math.concatenate([a, b])<br><br>Quantity([1 2 3 4], "s")</code></pre> |
| `stack(arrays, axis=0, dtype=None, **kwargs)` | Join a sequence of quantities or arrays along a new axis. | <pre><code>a = [1, 2, 3] * u.second<br>b = [4, 5, 6] * u.second<br>u.math.stack([a, b])<br><br>Quantity([[1 2 3]<br>          [4 5 6]], "s")</code></pre> |
| `block(arrays, **kwargs)` | Assemble a quantity or an array from nested lists of blocks. | <pre><code>a = [[1, 2], [3, 4]] * u.second<br>u.math.block(a)<br><br>Quantity([[1 2]<br>          [3 4]], "s")</code></pre> |
| `append(arr, values, axis=None, **kwargs)` | Append values to the end of a quantity or an array. | <pre><code>a = [1, 2, 3] * u.second<br>u.math.append(a, 4 * u.second)<br><br>Quantity([1 2 3 4], "s")</code></pre> |
| `row_stack(tup, dtype=None, **kwargs)` | Stack quantities or arrays in sequence vertically (row wise). | <pre><code>a = [1, 2, 3] * u.meter<br>b = [4, 5, 6] * u.meter<br>u.math.row_stack([a, b])<br><br>Quantity([[1 2 3]<br>          [4 5 6]], "m")</code></pre> |
| `vstack(tup, dtype=None, **kwargs)` | Stack quantities or arrays in sequence vertically (row wise). | <pre><code>a = [1, 2, 3] * u.meter<br>b = [4, 5, 6] * u.meter<br>u.math.vstack([a, b])<br><br>Quantity([[1 2 3]<br>          [4 5 6]], "m")</code></pre> |
| `hstack(arrays, dtype=None, **kwargs)` | Stack quantities or arrays in sequence horizontally (column wise). | <pre><code>a = [1, 2, 3] * u.meter<br>b = [4, 5, 6] * u.meter<br>u.math.hstack([a, b])<br><br>Quantity([1 2 3 4 5 6], "m")</code></pre> |
| `dstack(arrays, dtype=None, **kwargs)` | Stack quantities or arrays in sequence depth wise (along the third axis). | <pre><code>a = [[1], [2], [3]] * u.meter<br>b = [[4], [5], [6]] * u.meter<br>u.math.dstack([a, b])<br><br>Quantity([[[1 4]]<br>          [[2 5]]<br>          [[3 6]]], "m")</code></pre> |
| `column_stack(tup, **kwargs)` | Stack 1-D arrays as columns into a 2-D array. | <pre><code>a = [1, 2, 3] * u.second<br>b = [4, 5, 6] * u.second<br>u.math.column_stack([a, b])<br><br>Quantity([[1 4]<br>          [2 5]<br>          [3 6]], "s")</code></pre> |

`row_stack` is documented as an alias of `vstack`; its callable signature therefore mirrors `vstack`.

## Splitting

| Exact signature | One-line description | Example & result |
|---|---|---|
| `split(a, indices_or_sections, axis=0, **kwargs)` | Split a quantity or array into a list of multiple sub-arrays. | <pre><code>a = jnp.arange(9.0) * u.second<br>u.math.split(a, 3)<br><br>[Quantity([0. 1. 2.], "s"),<br> Quantity([3. 4. 5.], "s"),<br> Quantity([6. 7. 8.], "s")]</code></pre> |
| `array_split(ary, indices_or_sections, axis=0, **kwargs)` | Split an array into multiple sub-arrays. | <pre><code>a = jnp.arange(9.0) * u.second<br>parts = u.math.array_split(a, 3)<br>[x.shape for x in parts]<br><br>[(3,), (3,), (3,)]</code></pre> |
| `dsplit(a, indices_or_sections, **kwargs)` | Split a quantity or an array into multiple sub-arrays along the third axis (depth). | <pre><code>a = jnp.arange(16.0).reshape(2, 2, 4) * u.meter<br>parts = u.math.dsplit(a, 2)<br>[x.shape for x in parts]<br><br>[(2, 2, 2), (2, 2, 2)]</code></pre> |
| `hsplit(a, indices_or_sections, **kwargs)` | Split a quantity or an array into multiple sub-arrays horizontally (column-wise). | <pre><code>a = jnp.arange(16.0).reshape(4, 4) * u.meter<br>parts = u.math.hsplit(a, 2)<br>[x.shape for x in parts]<br><br>[(4, 2), (4, 2)]</code></pre> |
| `vsplit(a, indices_or_sections, **kwargs)` | Split a quantity or an array into multiple sub-arrays vertically (row-wise). | <pre><code>a = jnp.arange(16.0).reshape(4, 4) * u.meter<br>parts = u.math.vsplit(a, 2)<br>[x.shape for x in parts]<br><br>[(2, 4), (2, 4)]</code></pre> |

## Minimum Rank And Broadcasting

| Exact signature | One-line description | Example & result |
|---|---|---|
| `atleast_1d(*arys, **kwargs)` | View inputs as quantities or arrays with at least one dimension. | <pre><code>u.math.atleast_1d(0 * u.second)<br><br>Quantity([0], "s")</code></pre> |
| `atleast_2d(*arys, **kwargs)` | View inputs as quantities or arrays with at least two dimensions. | <pre><code>a = [1, 2, 3] * u.second<br>u.math.atleast_2d(a)<br><br>Quantity([[1 2 3]], "s")</code></pre> |
| `atleast_3d(*arys, **kwargs)` | View inputs as quantities or arrays with at least three dimensions. | <pre><code>a = [[1, 2], [3, 4]] * u.meter<br>u.math.atleast_3d(a).shape<br><br>(2, 2, 1)</code></pre> |
| `broadcast_arrays(*args, **kwargs)` | Broadcast any number of arrays against each other. | <pre><code>a = [1, 2, 3] * u.second<br>b = [[4], [5]] * u.second<br>result = u.math.broadcast_arrays(a, b)<br>[x.shape for x in result]<br><br>[(2, 3), (2, 3)]</code></pre> |
| `broadcast_to(array, shape, **kwargs)` | Broadcast an array to a new shape. | <pre><code>a = [1, 2, 3] * u.meter<br>u.math.broadcast_to(a, (2, 3))<br><br>Quantity([[1 2 3]<br>          [1 2 3]], "m")</code></pre> |

## Shape And Axis Operations

| Exact signature | One-line description | Example & result |
|---|---|---|
| `reshape(a, shape, order='C', **kwargs)` | Gives a new shape to a quantity or an array without changing its data. | <pre><code>a = [1, 2, 3, 4] * u.second<br>u.math.reshape(a, (2, 2))<br><br>Quantity([[1 2]<br>          [3 4]], "s")</code></pre> |
| `moveaxis(a, source, destination, **kwargs)` | Moves axes of a quantity or an array to new positions. | <pre><code>a = jnp.zeros((3, 4, 5)) * u.meter<br>u.math.moveaxis(a, 0, -1).shape<br><br>(4, 5, 3)</code></pre> |
| `transpose(a, axes=None, **kwargs)` | Permute the dimensions of a quantity or an array. | <pre><code>a = jnp.ones((2, 3)) * u.second<br>u.math.transpose(a).shape<br><br>(3, 2)</code></pre> |
| `swapaxes(a, axis1, axis2, **kwargs)` | Interchange two axes of a quantity or an array. | <pre><code>a = jnp.zeros((3, 4, 5)) * u.meter<br>u.math.swapaxes(a, 0, 2).shape<br><br>(5, 4, 3)</code></pre> |
| `expand_dims(a, axis, **kwargs)` | Expand the shape of a quantity or an array. | <pre><code>a = [1, 2, 3] * u.meter<br>u.math.expand_dims(a, axis=0).shape<br><br>(1, 3)</code></pre> |
| `squeeze(a, axis=None, **kwargs)` | Remove single-dimensional entries from the shape of a quantity or an array. | <pre><code>a = [[[1], [2], [3]]] * u.second<br>u.math.squeeze(a).shape<br><br>(3,)</code></pre> |
| `flatten(x, start_axis=None, end_axis=None, **kwargs)` | Flattens input by reshaping it into a one-dimensional tensor. | <pre><code>a = [[1, 2], [3, 4]] * u.second<br>u.math.flatten(a)<br><br>Quantity([1 2 3 4], "s")</code></pre> |
| `unflatten(x, axis, sizes, **kwargs)` | Expands a dimension of the input tensor over multiple dimensions. | <pre><code>a = [1, 2, 3, 4, 5, 6] * u.meter<br>u.math.unflatten(a, 0, (2, 3))<br><br>Quantity([[1 2 3]<br>          [4 5 6]], "m")</code></pre> |

## Repetition

| Exact signature | One-line description | Example & result |
|---|---|---|
| `tile(A, reps, **kwargs)` | Construct a quantity or an array by repeating `A` the number of times given by `reps`. | <pre><code>a = [1, 2, 3] * u.meter<br>u.math.tile(a, 2)<br><br>Quantity([1 2 3 1 2 3], "m")</code></pre> |
| `repeat(a, repeats, axis=None, total_repeat_length=None, **kwargs)` | Repeat elements of a quantity or an array. | <pre><code>a = [1, 2, 3] * u.second<br>u.math.repeat(a, 2)<br><br>Quantity([1 1 2 2 3 3], "s")</code></pre> |

## Selection

| Exact signature | One-line description | Example & result |
|---|---|---|
| `take(a, indices, axis=None, mode=None, unique_indices=False, indices_are_sorted=False, fill_value=None, **kwargs)` | Take elements from an array along an axis. | <pre><code>a = [4, 3, 5, 7, 6, 8] * u.second<br>u.math.take(a, jnp.array([0, 1, 4]))<br><br>Quantity([4 3 6], "s")</code></pre> |
| `gather(input, dim, index, **kwargs)` | Gather values along an axis specified by `dim`, according to `index`. | <pre><code>a = jnp.array([[1, 2], [3, 4]]) * u.mV<br>index = jnp.array([[0, 0], [1, 0]])<br>u.math.gather(a, 1, index)<br><br>Quantity([[1 1]<br>          [4 3]], "mV")</code></pre> |
| `choose(a, choices, mode='raise', **kwargs)` | Construct a quantity or an array from an index array and a set of arrays to choose from. | <pre><code>choices = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6])]<br>u.math.choose(jnp.array([0, 1, 0]), choices)<br><br>Array([1, 5, 3], dtype=int32)</code></pre> |
| `compress(condition, a, axis=None, *, size=None, fill_value=None, **kwargs)` | Return selected slices of a quantity or an array along given axis. | <pre><code>a = [1, 2, 3, 4] * u.meter<br>u.math.compress(jnp.array([0, 1, 1, 0]), a)<br><br>Quantity([2 3], "m")</code></pre> |
| `extract(condition, arr, *, size=None, fill_value=None, **kwargs)` | Return the elements of an array that satisfy some condition. | <pre><code>a = jnp.array([1, 2, 3]) * u.meter<br>u.math.extract(a.mantissa &gt; 1, a)<br><br>Quantity([2 3], "m")</code></pre> |

## Sources Mirrored

- https://brainunit.readthedocs.io/apis/brainunit.math.html
- Generated function pages linked from the module index: `https://brainunit.readthedocs.io/apis/generated/brainunit.math.<function>.html`
