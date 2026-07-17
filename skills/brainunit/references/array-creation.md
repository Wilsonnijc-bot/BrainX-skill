# Array Creation

Start with `asarray()` to normalize unit-bearing inputs and `arange()` to build physical ranges. Use the remaining sections for grids, filled arrays, template-shaped arrays, and matrix patterns; basic scalar or direct `Quantity` construction remains in the skill body.

## Contents

- [`asarray`: normalize and attach units](#asarray-normalize-and-attach-units)
- [`arange`: build unit-aware ranges](#arange-build-unit-aware-ranges)
- [Other ranges and grids](#other-ranges-and-grids)
- [Filled and template arrays](#filled-and-template-arrays)
- [Matrix constructors](#matrix-constructors)
- [Additional API constructors](#additional-api-constructors)
- [Source-backed gotchas](#source-backed-gotchas)

## `asarray`: Normalize And Attach Units

Use `asarray()` as the primary normalization boundary when input may be a plain sequence, an existing `Quantity`, or a sequence of compatible quantities. `brainunit.math.array` is an alias for `brainunit.math.asarray`.

```text
asarray(a, dtype=None, order=None, unit=None)
```

| Input and `unit` | Result |
|---|---|
| Plain data, no `unit` | Plain array with inferred `dtype` |
| Unit-bearing quantity data, no `unit` | `Quantity` with its unit inferred from the input |
| Compatible quantity data plus `unit=target` | `Quantity` converted to `target` |
| Incompatible quantity elements or target | `UnitMismatchError` |

All elements in a list of quantities must share the same dimension. Supplying `unit` makes the target scale explicit and converts compatible inputs before constructing the result.

```python
import brainunit as u

plain = u.math.asarray([1, 2, 3])
inferred = u.math.asarray([1 * u.second, 2 * u.second])
seconds = u.math.asarray(
    [1000 * u.ms, 2000 * u.ms],
    unit=u.second,
)

try:
    invalid = u.math.asarray([1 * u.second], unit=u.ampere)
except Exception as error:
    print("unit error:", error)
```

Use `dtype` to control the numerical representation and `order` for supported memory-layout requests. Do not strip units before normalization merely to satisfy an array constructor.

## `arange`: Build Unit-Aware Ranges

Use `u.math.arange()` as the default constructor for simulation time axes, parameter sweeps, and other half-open physical intervals.

```text
arange(start=None, stop=None, step=None, dtype=None)
```

Values lie in `[start, stop)`. When any supplied bound is a `Quantity`, `start`, `stop`, and `step` must share the same unit; incompatible or mixed physical dimensions raise `UnitMismatchError`. If neither `start` nor `stop` is given, `arange()` raises `TypeError`.

```python
# Plain index range.
indices = u.math.arange(5)

# Quantity stop with an explicit quantity step; start defaults to zero.
first_seconds = u.math.arange(5 * u.second, step=1 * u.second)

# Canonical simulation time axis.
times = u.math.arange(0.0 * u.ms, 100.0 * u.ms, 0.1 * u.ms)

# Parameter sweep with an explicit dtype.
currents = u.math.arange(
    0.0 * u.nA,
    1.0 * u.nA,
    0.1 * u.nA,
    dtype=float,
)
```

Prefer this unit-aware interval construction when endpoints already carry physical meaning; it validates the interval before returning a `Quantity` and avoids attaching a unit to an unchecked plain range afterward.

## Other Ranges And Grids

| Exact signature | Unit rule |
|---|---|
| `linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)` | Quantity endpoints must have the same unit; `endpoint=False` excludes `stop` |
| `logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)` | `start` and `stop` must be dimensionless; result is always a plain array |
| `meshgrid(*xi, copy=True, sparse=False, indexing='xy')` | Accepts coordinate vectors and supports `Quantity` inputs |
| `vander(x, N=None, increasing=False, unit=Unit('1'))` | `x` must be dimensionless if it is a `Quantity`; `unit` controls the returned `Quantity` |

```python
samples = u.math.linspace(0 * u.second, 10 * u.second, 5)
x_grid, y_grid = u.math.meshgrid(
    u.math.asarray([1, 2, 3], unit=u.second),
    u.math.asarray([4, 5], unit=u.second),
)
```

`array_split()` appears on the tutorial page, but it is a structural operation; route it to `array-mechanics.md`.

## Filled And Template Arrays

| Exact signature | Result selection |
|---|---|
| `full(shape, fill_value, dtype=None)` | A quantity fill value gives the result the same unit; otherwise returns a plain array |
| `empty(shape, dtype=None, unit=Unit('1'))` | `unit` selects a `Quantity`; entries are uninitialized |
| `ones(shape, dtype=None, unit=Unit('1'))` | `unit` selects a unit-aware array of ones |
| `zeros(shape, dtype=None, unit=Unit('1'))` | `unit` selects a unit-aware array of zeros |
| `full_like(a, fill_value, dtype=None, shape=None)` | Uses `a` for shape and dtype; a quantity prototype requires a compatible fill value |
| `empty_like(prototype, dtype=None, shape=None, unit=Unit('1'))` | Uses prototype shape and dtype, with optional overrides |
| `ones_like(a, dtype=None, shape=None, unit=Unit('1'))` | Preserves a quantity prototype's unit; `shape` can override shape |
| `zeros_like(a, dtype=None, shape=None, unit=Unit('1'))` | Preserves a quantity prototype's unit; `shape` can override shape |

```python
q = u.math.ones((2, 2), unit=u.second)
same_shape = u.math.zeros_like(q)
filled = u.math.full_like(q, 4 * u.second)
```

## Matrix Constructors

| Exact signature | Purpose |
|---|---|
| `eye(N, M=None, k=0, dtype=None, unit=Unit('1'))` | Identity-like 2-D result with diagonal offset `k` |
| `identity(n, dtype=None, unit=Unit('1'))` | Square identity result |
| `tri(N, M=None, k=0, dtype=None, unit=Unit('1'))` | Ones at and below diagonal `k` |
| `diag(v, k=0, unit=Unit('1'))` | Constructs a diagonal matrix from 1-D input or extracts from 2-D input |
| `tril(m, k=0, unit=Unit('1'))` | Zeros entries above diagonal `k`; applies to final two axes when `ndim > 2` |
| `triu(m, k=0, unit=Unit('1'))` | Zeros entries below diagonal `k`; applies to final two axes when `ndim > 2` |
| `fill_diagonal(a, val, wrap=False, inplace=False)` | Fills locations `a[i, i, ..., i]`; `val` must have a compatible unit |

```python
identity_seconds = u.math.eye(3, unit=u.second)
diagonal_seconds = u.math.diag(u.math.asarray([1, 2, 3]), unit=u.second)
updated = u.math.fill_diagonal(
    u.math.zeros((3, 3), unit=u.second),
    4 * u.second,
)
```

## Additional API Constructors

- `tril_indices(n, k=0, m=None)` and `triu_indices(n, k=0, m=None)` create triangular index tuples.
- `tril_indices_from(arr, k=0)` and `triu_indices_from(arr, k=0)` derive triangular indices from an array shape.
- `from_numpy(x, unit=Unit('1'))` converts a NumPy array to a JAX array and can attach a unit.
- `tree_ones_like(tree)` and `tree_zeros_like(tree)` retain tree structure and fill every leaf.

Backend extraction and `as_numpy()` belong in `array-mechanics.md` because they cross an existing array boundary rather than create a scientific quantity from source values.

## Source-Backed Gotchas

- `logspace()` rejects unit-bearing exponents because `base ** x` is intrinsically dimensionless.
- `vander()` likewise requires a dimensionless `x`; its columns use different powers of `x`.
- `full_like()` rejects a unit-bearing fill for a plain prototype and a plain fill for a unit-bearing prototype.
- The `unit` argument to `diag()` is ignored when `v` already carries a unit.
- `empty()` and `empty_like()` do not initialize entries.

## Sources Mirrored

- https://brainunit.readthedocs.io/unit_operations/array_creation.html
- https://brainunit.readthedocs.io/apis/brainunit.math.html
