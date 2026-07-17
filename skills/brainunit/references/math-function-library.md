# Math Function Library

Use this reference after the basic `brainunit.math` workflow is known. Route a function by what it requires from input units and what it does to output units. Array creation and purely structural manipulation are owned by `array-creation.md` and `array-mechanics.md`.

## Contents

- [Selection map](#selection-map)
- [Activation functions](#activation-functions)
- [Dimensionless evaluation](#dimensionless-evaluation)
- [Unit-changing operations](#unit-changing-operations)
- [Unit-preserving operations](#unit-preserving-operations)
- [Unit-removing operations](#unit-removing-operations)
- [Reductions and contractions](#reductions-and-contractions)
- [Einstein operations](#einstein-operations)
- [Source-backed gotchas](#source-backed-gotchas)

## Selection Map

| Question | Route |
|---|---|
| Must the mathematical argument be dimensionless? | Dimensionless-evaluation family |
| Does the operation multiply, divide, power, or contract values? | Unit-changing family |
| Does the result represent the same physical kind as the input? | Unit-preserving family |
| Is the result a boolean, count, or index? | Unit-removing family |
| Are named axes being reduced or contracted? | `einreduce()` or `einsum()` |

## Activation Functions

The API lists activation functions separately because their unit behavior is not uniform.

| Unit behavior | Exact API names |
|---|---|
| Preserves the input unit | `relu`, `leaky_relu` |
| Requires dimensionless input or an explicit compatible `unit_to_scale`; returns a plain array | `relu6`, `sigmoid`, `softplus`, `sparse_plus`, `sparse_sigmoid`, `soft_sign`, `silu`, `swish`, `log_sigmoid`, `hard_sigmoid`, `hard_silu`, `hard_swish`, `hard_tanh`, `elu`, `celu`, `selu`, `gelu`, `glu`, `squareplus`, `mish` |

Use `silu` and `swish` as the API's two names for the same activation family, and likewise `hard_silu` and `hard_swish`. Do not assume the unit-preserving behavior of `relu()` extends to sigmoid- or exponential-style activations.

## Dimensionless Evaluation

The API groups these under "Functions that Accepting Unitless":

- Exponential/logarithmic: `exprel`, `set_exprel_order`, `exp`, `exp2`, `expm1`, `log`, `log10`, `log1p`, `log2`.
- Trigonometric/hyperbolic: `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh`, `cos`, `cosh`, `sin`, `sinc`, `sinh`, `tan`, `tanh`.
- Angle conversion: `deg2rad`, `rad2deg`, `degrees`, `radians`, `angle`.
- Paired/statistical: `frexp`, `hypot`, `arctan2`, `logaddexp`, `logaddexp2`, `corrcoef`, `correlate`, `cov`, `ldexp`.
- Bitwise: `bitwise_not`, `invert`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`, `right_shift`.

Representative signatures that expose an explicit scale are:

```text
exp(x, unit_to_scale=None, **kwargs)
log(x, unit_to_scale=None, **kwargs)
hypot(x, y, unit_to_scale=None, **kwargs)
arctan2(x, y, unit_to_scale=None, **kwargs)
corrcoef(x, y=None, rowvar=True, unit_to_scale=None, **kwargs)
cov(m, y=None, rowvar=True, bias=False, ddof=None,
    fweights=None, aweights=None, unit_to_scale=None, **kwargs)
```

For `exp()`, the generated API explicitly requires `unit_to_scale` when `x` has physical units so that the evaluated exponent becomes dimensionless.

## Unit-Changing Operations

| Family | Exact API names |
|---|---|
| Reciprocal/product reductions | `reciprocal`, `prod`, `product`, `nancumprod`, `nanprod`, `cumprod`, `cumproduct` |
| Variance | `var`, `nanvar` |
| Powers and roots | `cbrt`, `square`, `sqrt`, `power`, `float_power`, `matrix_power` |
| Binary arithmetic | `multiply`, `divide`, `true_divide`, `floor_divide`, `divmod` |
| Products/contractions | `cross`, `convolve`, `trapezoid`, `dot`, `multi_dot`, `vdot`, `vecdot`, `inner`, `outer`, `kron`, `matmul`, `tensordot` |

The official generated pages state these representative rules:

| Signature | Unit result |
|---|---|
| `reciprocal(x, **kwargs)` | Inverse input unit |
| `prod(x, axis=None, dtype=None, keepdims=False, initial=None, where=None)` | Input unit raised to the number of multiplied elements |
| `var(a, axis=None, dtype=None, ddof=0, keepdims=False, *, where=None, **kwargs)` | Square of the input unit |
| `square(x, **kwargs)` | Square of the input unit |
| `sqrt(x, **kwargs)` | Square root of the input unit |
| `multiply(x, y, **kwargs)` | Product of input units |
| `divide(x, y, **kwargs)` | Quotient of input units |
| `power(x, y, **kwargs)` | `x.unit ** y`; `y` must be dimensionless |
| `dot(a, b, *, precision=None, preferred_element_type=None, **kwargs)` | `a.unit * b.unit` |
| `matmul(a, b, *, precision=None, preferred_element_type=None, **kwargs)` | `a.unit * b.unit` |
| `tensordot(a, b, axes=2, precision=None, preferred_element_type=None, **kwargs)` | `a.unit * b.unit` |
| `matrix_power(a, n, **kwargs)` | `a.unit ** n` |

## Unit-Preserving Operations

Exclude the structural names routed to `array-mechanics.md`. The remaining API category includes:

- Order/extrema: `sort`, `max`, `min`, `amax`, `amin`, `nanmin`, `nanmax`, `ptp`.
- Additive and cumulative reductions: `sum`, `nancumsum`, `nansum`, `cumsum`.
- Statistics: `median`, `average`, `mean`, `std`, `nanmedian`, `nanmean`, `nanstd`, `percentile`, `nanpercentile`, `quantile`, `nanquantile`.
- Unary same-kind operations: `real`, `imag`, `conj`, `conjugate`, `negative`, `positive`, `abs`, `absolute`, `fabs`, `round`, `around`, `rint`, `floor`, `ceil`, `trunc`, `fix`.
- Same-kind elementwise selection/arithmetic: `add`, `subtract`, `maximum`, `minimum`, `fmax`, `fmin`, `clip`, `interp`, `where`, `select`.
- Other preserving operations: `ediff1d`, `diff`, `rot90`, `intersect1d`, `nan_to_num`, `modf`, `fmod`, `mod`, `copysign`, `remainder`, `lcm`, `gcd`, `trace`, `nextafter`, `promote_dtypes`, `histogram`, `compress`, `extract`, `take`, `choose`, `unique`, `gather`.

Representative reduction signatures:

```text
sum(x, axis=None, dtype=None, keepdims=False, initial=None, where=None)
mean(x, axis=None, dtype=None, keepdims=False, *, where=None, **kwargs)
std(x, axis=None, dtype=None, ddof=0, keepdims=False, *, where=None, **kwargs)
percentile(a, q, axis=None, method='linear', keepdims=False, **kwargs)
```

`sum()`, `mean()`, and `std()` return a `Quantity` when their input is a `Quantity`. In contrast, `var()` belongs to the unit-changing category because variance squares the unit.

`histogram(x, bins=10, range=None, weights=None, density=None, **kwargs)` returns histogram values and bin edges; the API return type allows the bin edges to be a `Quantity`.

## Unit-Removing Operations

| Result kind | Exact API names |
|---|---|
| Boolean/comparison | `all`, `any`, `logical_not`, `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`, `array_equal`, `isclose`, `allclose`, `logical_and`, `logical_or`, `logical_xor`, `alltrue`, `sometrue` |
| Indices | `argsort`, `argmax`, `argmin`, `nanargmax`, `nanargmin`, `argwhere`, `nonzero`, `flatnonzero`, `searchsorted`, `diag_indices_from` |
| Counts/digits/signs | `iscomplexobj`, `heaviside`, `signbit`, `sign`, `bincount`, `digitize`, `count_nonzero`, `get_promote_dtypes` |

Important signatures and behaviors:

```text
equal(x, y, *args, **kwargs)
isclose(x, y, rtol=None, atol=None, equal_nan=False, **kwargs)
allclose(x, y, rtol=None, atol=None, equal_nan=False, **kwargs)
argsort(a, axis=-1, *, kind=None, order=None, stable=True,
        descending=False, **kwargs)
searchsorted(a, v, side='left', sorter=None, *, method='scan', **kwargs)
nonzero(a, *, size=None, fill_value=None, **kwargs)
count_nonzero(a, axis=None, keepdims=None, **kwargs)
```

`equal()` converts `y` to `x`'s unit when both are quantities and raises `TypeError` when only one operand has units. `argsort()`, `argmax()`, `nonzero()`, and `count_nonzero()` strip units because their results are indices or counts.

## Reductions And Contractions

Use the output's physical meaning to choose among similar reducers:

```python
import brainunit as u

x = u.math.asarray([1.0, 2.0, 3.0], unit=u.meter)

total = u.math.sum(x)           # keeps metre
center = u.math.mean(x)         # keeps metre
spread = u.math.std(x)          # keeps metre
spread2 = u.math.var(x)         # metre ** 2
area_terms = u.math.square(x)   # metre ** 2
```

For comparisons with tolerances, `rtol` and `atol` may be `Quantity` values according to the generated API. Match absolute tolerance to the compared physical dimension.

## Einstein Operations

| Exact signature | Use |
|---|---|
| `einreduce(x, pattern, reduction, **axes_lengths)` | Named-axis reduction plus reordering |
| `einsum(subscripts, /, *operands, optimize='optimal', precision=None, preferred_element_type=None)` | General reductions, inner/outer products, and contractions |

An axis absent from the `einreduce()` output is reduced. The tutorial demonstrates `mean`, `min`, `max`, `sum`, and `prod`:

```python
pooled = u.math.einreduce(
    x,
    'b (h h2) (w w2) c -> b h w c',
    'mean',
    h2=2,
    w2=2,
)
```

Use `einrearrange()` and `einrepeat()` through `array-mechanics.md` when element count is preserved or increased rather than reduced.

## Source-Backed Gotchas

- `all()` and `any()` require dimensionless input and raise `TypeError` for physical units.
- `power()` requires a dimensionless exponent.
- `prod()` changes the unit exponent according to the number of factors; it is not interchangeable with additive reductions.
- `where(condition, x, y)` requires `x`, `y`, and `condition` to be broadcastable; use unit-compatible branches such as `a` and `0 * u.meter`.
- `searchsorted()` converts a quantity `v` to the unit of quantity `a` before searching.
- `isclose()` and `allclose()` return booleans, not quantities.

## Sources Mirrored

- https://brainunit.readthedocs.io/apis/brainunit.math.html
- https://brainunit.readthedocs.io/unit_operations/einstein_operations.html
