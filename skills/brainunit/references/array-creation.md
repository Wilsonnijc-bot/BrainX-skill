# Array Creation

Use this reference for specialized unit-aware ranges, grids, filled arrays, template-shaped arrays, matrix patterns, index constructors, and tree-shaped arrays. Basic `Quantity`, `asarray()`, and `arange()` construction remains in the parent skill.

```python
import brainunit as u
import jax.numpy as jnp
```

## Contents

- [Choose interval values](#choose-interval-values)
- [Build coordinate and power grids](#build-coordinate-and-power-grids)
- [Fill a new shape](#fill-a-new-shape)
- [Follow an existing shape](#follow-an-existing-shape)
- [Build matrices and masks](#build-matrices-and-masks)
- [Locate triangular entries](#locate-triangular-entries)
- [Mirror pytrees](#mirror-pytrees)
- [Convert NumPy input](#convert-numpy-input)
- [Routing](#routing)
- [Cross-family gotchas](#cross-family-gotchas)
- [Sources mirrored](#sources-mirrored)

## Choose Interval Values

### `linspace`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)` | Generate a fixed number of evenly spaced samples between two endpoints. | Return evenly spaced numbers over a specified interval. | <pre><code>u.math.linspace(0 * u.second, 10 * u.second, 5)<br><br>Quantity([ 0.   2.5  5.   7.5 10. ], "s")</code></pre> |

### `logspace`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)` | Generate values whose exponents are evenly spaced. | Return numbers spaced evenly on a log scale. | <pre><code>u.math.logspace(0, 2, 3)<br><br>Array([  1.,  10., 100.], dtype=float32)</code></pre> |

## Build Coordinate And Power Grids

### `meshgrid`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `meshgrid(*xi, copy=True, sparse=False, indexing='xy')` | Expand 1-D coordinate vectors into dense or sparse coordinate grids. | Return coordinate matrices from coordinate vectors. | <pre><code>x = jnp.array([1, 2]) * u.meter<br>t = jnp.array([0, 1, 2]) * u.second<br>x_grid, t_grid = u.math.meshgrid(x, t, indexing='ij')<br><br># both shapes: (2, 3)<br># units: meter and second</code></pre> |

### `vander`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `vander(x, N=None, increasing=False, unit=Unit('1'))` | Build a Vandermonde matrix from powers of a 1-D vector. | Generate a Vandermonde matrix. | <pre><code>u.math.vander(jnp.array([1, 2]), N=3)<br><br>Array([[1, 1, 1],<br>       [4, 2, 1]], dtype=int32)</code></pre> |

## Fill A New Shape

### Initialized arrays

| API | Description & result |
|---|---|
| `full(shape, fill_value, dtype=None)` | Fill a new shape with `fill_value`; a quantity fill produces a `Quantity` with that unit.<br><br><pre><code>u.math.full(2, 3 * u.second)<br># Quantity([3, 3], "s")</code></pre> |
| `ones(shape, dtype=None, unit=Unit('1'))` | Fill a new shape with ones. Returns a plain JAX array by default or a `Quantity` when a non-trivial unit is supplied.<br><br><pre><code>u.math.ones(2, unit=u.second)<br># Quantity([1., 1.], "s")</code></pre> |
| `zeros(shape, dtype=None, unit=Unit('1'))` | Fill a new shape with zeros. Returns a plain JAX array by default or a `Quantity` when a non-trivial unit is supplied.<br><br><pre><code>u.math.zeros(2)<br># plain JAX array</code></pre> |

### `empty`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `empty(shape, dtype=None, unit=Unit('1'))` | Allocate a shape when every entry will be overwritten before it is read. | Return a new quantity or array of given shape and type, without initializing entries. | <pre><code>buffer = u.math.empty((2, 2), unit=u.second)<br><br>buffer.shape  # (2, 2)<br># Entry values are unspecified.</code></pre> |

## Follow An Existing Shape

### `full_like`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `full_like(a, fill_value, dtype=None, shape=None)` | Fill an array using a prototype's shape and dtype. | Return a new quantity or array with the same shape and type as a given array or quantity, filled with `fill_value`. | <pre><code>prototype = jnp.zeros(2) * u.second<br>u.math.full_like(prototype, 500 * u.ms)<br><br>Quantity([0.5 0.5], "s")</code></pre> |

### `empty_like`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `empty_like(prototype, dtype=None, shape=None, unit=Unit('1'))` | Allocate an uninitialized buffer shaped and typed like a prototype. | Return a new quantity or array with the same shape and type as a given array. | <pre><code>prototype = jnp.ones(3) * u.second<br>buffer = u.math.empty_like(prototype)<br><br>buffer.shape  # (3,)<br># unit: second; entry values are unspecified</code></pre> |

### Create One or zero arrays

| API | Description & result |
|---|---|
| `ones_like(a, dtype=None, shape=None, unit=Unit('1'))` | Create ones with `a`'s shape and dtype. A `Quantity` prototype preserves its unit by default; a non-trivial `unit` attaches a unit to a plain prototype or requests a compatible conversion. |
| `zeros_like(a, dtype=None, shape=None, unit=Unit('1'))` | Create zeros with `a`'s shape and dtype. A `Quantity` prototype preserves its unit by default; a non-trivial `unit` attaches a unit to a plain prototype or requests a compatible conversion. |

## Build Matrices And Masks

### Identity-like and triangular constructors

| API | Description & result |
|---|---|
| `eye(N, M=None, k=0, dtype=None, unit=Unit('1'))` | Create an `N`-by-`M` array with ones on diagonal `k` and zeros elsewhere. Returns a plain JAX array by default or a `Quantity` with a non-trivial unit. |
| `identity(n, dtype=None, unit=Unit('1'))` | Create an `n`-by-`n` identity array. Returns a plain JAX array by default or a `Quantity` with a non-trivial unit. |
| `tri(N, M=None, k=0, dtype=None, unit=Unit('1'))` | Create an `N`-by-`M` array with ones at and below diagonal `k`. Returns a plain JAX array by default or a `Quantity` with a non-trivial unit. |

### `diag`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `diag(v, k=0, unit=Unit('1'))` | Construct a diagonal matrix or extract a diagonal based on input rank. | Extract a diagonal or construct a diagonal array. | <pre><code>vector = jnp.array([1, 2, 3]) * u.second<br>matrix = u.math.diag(vector)<br>extracted = u.math.diag(matrix)<br><br>matrix.shape, extracted.shape  # ((3, 3), (3,))</code></pre> |

### Keep one triangle of an existing array

| API | Description & result |
|---|---|
| `tril(m, k=0, unit=Unit('1'))` | Return `m` with entries above diagonal `k` zeroed, applying to the final two axes when `m.ndim > 2`. It preserves a `Quantity` input's unit by default; `unit` can attach or request a compatible output unit.<br><br><pre><code>m = jnp.arange(1, 5).reshape(2, 2)<br>u.math.tril(m)<br># [[1, 0], [3, 4]]</code></pre> |
| `triu(m, k=0, unit=Unit('1'))` | Return `m` with entries below diagonal `k` zeroed, applying to the final two axes when `m.ndim > 2`. It preserves a `Quantity` input's unit by default; `unit` can attach or request a compatible output unit.<br><br><pre><code>m = jnp.arange(1, 5).reshape(2, 2)<br>u.math.triu(m)<br># [[1, 2], [0, 4]]</code></pre> |

### `fill_diagonal`

| Exact signature | Use when | One-line description | Code and result |
|---|---|---|---|
| `fill_diagonal(a, val, wrap=False, inplace=False)` | Replace the main diagonal of an existing array with one value. | Fill the main diagonal of the given array of any dimensionality. | <pre><code>q = jnp.zeros((2, 2)) * u.second<br>filled = u.math.fill_diagonal(q, 500 * u.ms)<br><br>Quantity([[0.5 0. ]<br>          [0.  0.5]], "s")</code></pre> |

## Locate Triangular Entries

### Lower and upper triangle indices

| API | Description & result |
|---|---|
| `tril_indices(n, k=0, m=None)` | Return row and column indices for the lower triangle of an `(n, m)` array as two plain integer arrays. |
| `triu_indices(n, k=0, m=None)` | Return row and column indices for the upper triangle of an `(n, m)` array as two plain integer arrays. |
| `tril_indices_from(arr, k=0)` | Derive lower-triangle row and column indices from a 2-D array's shape. Returns two plain integer arrays; an input `Quantity` does not give the indices a unit. |
| `triu_indices_from(arr, k=0)` | Derive upper-triangle row and column indices from a 2-D array's shape. Returns two plain integer arrays; an input `Quantity` does not give the indices a unit. |

## Mirror Pytrees

### One or zero in every leaf

| API | Description & result |
|---|---|
| `tree_ones_like(tree)` | Replace every pytree leaf with ones while preserving structure, leaf shape, dtype, and unit; plain leaves remain plain and quantity leaves remain quantities. |
| `tree_zeros_like(tree)` | Replace every pytree leaf with zeros while preserving structure, leaf shape, dtype, and unit; plain leaves remain plain and quantity leaves remain quantities. |

## Convert NumPy Input

| API | Description & result |
|---|---|
| `from_numpy(x, unit=Unit('1'))` | Convert a NumPy array to the JAX backend. Returns a plain JAX array by default or a `Quantity` with the supplied non-trivial unit. |

## Routing

`array_split()` appears on the tutorial page, but it is a structural operation; route it to `array-mechanics.md`.

Backend extraction and `as_numpy()` belong in `array-mechanics.md` because they cross an existing array boundary rather than create a scientific quantity from source values.

## Cross-Family Gotchas

- For constructors with `unit=Unit('1')`, the unitless default produces a plain JAX array unless a quantity input supplies a unit; pass a non-trivial `Unit` to request a `Quantity`.
- The `dtype` and `shape` arguments on `*_like` functions override the prototype; otherwise its dtype and shape are retained.

## Sources Mirrored

- https://brainx.chaobrain.com/brainunit/unit_operations/array_creation.html
- https://brainx.chaobrain.com/brainunit/apis/brainunit.math.html
