---
name: brainunit
description: Enforce BrainUnit physical-quantity, dimensional, conversion, unit-aware math, JAX-transformation, and external-library boundary safety. Use when Codex works with BrainUnit or BrainCell values involving voltage, current, time, conductance, capacitance, length, concentration, temperature, physical constants, unit errors, dimensional mismatches, or suspicious bare numbers.
---

# BrainUnit Quantity Safety

## Purpose And Boundary

Use BrainUnit to keep physical meaning attached to numerical values throughout scientific computations. Keep the canonical path here; open only the narrow reference needed for specialized constructors, array mechanics, function semantics, custom units, temperature, prefixes, or constants.

## Core Model

- `brainunit` provides physical units and a unit-aware mathematical system in JAX for general AI-driven scientific computing.
- BrainUnit combines unit names with prefixes and appends a number for predefined squared and cubed forms. Examples include `msiemens`, `siemens2`, and `usiemens3`. Import `brainunit as u` instead of guessing a name or case-sensitive prefix.

## Canonical Imports

Use these imports for the bundled patterns below.

```python
import brainunit as u
import jax
import jax.numpy as jnp
from brainunit import constants
```

## Create And Inspect Quantities

A `Quantity` is a numeric value plus a physical unit. Attach units at creation and retain them through BrainUnit and BrainCell code. BrainCell enforces explicit units and raises `TypeError` when a physical parameter receives a bare number.

```python
# Scalars, arrays, and direct construction.
mass = 5.0 * u.kilogram
speed = 10.0 * u.meter / u.second
voltages = jnp.array([1.0, 2.5, 3.7]) * u.mV
current = u.Quantity(jnp.array([0.1, 0.2, 0.3]), unit=u.ampere)

# Inspect without discarding physical metadata.
q = jnp.array([[1.0, 2.0], [3.0, 4.0]]) * u.volt
print(q.mantissa, q.unit, q.dim, q.shape, q.dtype)

# BrainCell-facing physical parameters must also carry units.
resting_voltage = -65.0 * u.mV
time_step = 0.1 * u.ms
current_density = 5.0 * u.uA / u.cm**2
conductance_density = 0.03 * u.mS / u.cm**2
```

Do not infer a unit from a parameter name. Inspect `.unit` and `.dim` when debugging; `.mantissa` exposes the current stored scale without converting it.

## Compute With Dimensions

BrainUnit represents units through seven irreducible SI dimensions: length, mass, time, electric current, temperature, amount of substance, and luminous intensity. Units are tracked automatically: addition and subtraction require matching dimensions and align compatible scales, while multiplication, division, powers, and supported functions combine dimension exponents. Treat an incompatibility as a model error rather than stripping units.

```python
t1 = 500.0 * u.ms
t2 = 1.5 * u.second
elapsed = t1 + t2

work = (10.0 * u.newton) * (3.0 * u.meter)
average_speed = (100.0 * u.meter) / (10.0 * u.second)

try:
    invalid = 5.0 * u.meter + 3.0 * u.second
except Exception as error:
    print("dimension error:", error)
```

## Convert At Explicit Boundaries

Use `in_unit(target)` to rescale while retaining a `Quantity`. Use `to_decimal(target)` only when an external API requires raw numbers, and make its expected unit explicit in the variable or parameter name.

```python
distance = 2.5 * u.kmeter
distance_m = distance.in_unit(u.meter)
distance_m_raw = distance.to_decimal(u.meter)

times = u.math.arange(0.0 * u.ms, 10.0 * u.ms, 0.1 * u.ms)
plot_times_ms = times.to_decimal(u.ms)
```

Do not substitute `.mantissa` for conversion.

## Use Unit-Aware Math And Constants

Use `u.math` where unit semantics matter: functions may preserve units, change them, require dimensionless input, or return indices or booleans. Import predefined physical constants as quantities rather than recreating bare values.

```python
data = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]) * u.newton
total = u.math.sum(data)                             # newton
mean = u.math.mean(data)                             # newton
length = u.math.sqrt(4.0 * u.meter2)                 # meter
ordered = u.math.sort(jnp.array([3.0, 1.0, 2.0]) * u.volt)

avogadro = constants.avogadro
boltzmann = constants.boltzmann
elementary_charge = constants.elementary_charge
electron_mass = constants.electron_mass
```

## Transform And Validate Unit-Aware Functions

BrainUnit integrates with automatic differentiation, JIT compilation, vectorization, and parallel computation. Its strict physical-unit type checking and dimensional inference perform unit conversion and analysis at compilation time in compiled workflows; eager invalid operations raise when evaluated. Use `jax.jit` and `jax.vmap` with quantities, `u.autograd.grad` for unit-aware derivatives, and `@u.check_units` at scientific function boundaries.

```python
@jax.jit
def kinetic_energy(m, v):
    return 0.5 * m * v**2


energy = kinetic_energy(2.0 * u.kilogram, 3.0 * u.meter / u.second)
velocities = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]) * u.meter / u.second
energies = jax.vmap(lambda v: kinetic_energy(2.0 * u.kilogram, v))(velocities)

denergy_dv = u.autograd.grad(
    lambda v: 0.5 * (2.0 * u.kilogram) * v**2
)
momentum = denergy_dv(3.0 * u.meter / u.second)


@u.check_units(v=u.meter / u.second, t=u.second)
def displacement(v, t):
    return v * t


distance_traveled = displacement(
    10.0 * u.meter / u.second,
    5.0 * u.second,
)
```

## Reference Routing

| Reference | Open when |
|---|---|
| `references/quantity-inspection-and-conversion.md` | Inspecting compatibility, dimensions, conversions, decomposition, formatting, or raw-value boundaries beyond the canonical pattern. |
| `references/array-creation.md` | Normalizing inputs with `asarray`, constructing ranges with `arange`, or creating grids, templates, and specialized arrays. |
| `references/array-mechanics.md` | Indexing, functional updates, reshaping, broadcasting, joining, splitting, repeating, backend conversion, or named-axis rearrangement. |
| `references/math-function-library.md` | Selecting functions by dimensionless-input, unit-preserving, unit-changing, reduction, contraction, comparison, boolean, or index-returning semantics. |
| `references/unit-structure-and-definition.md` | Inspecting unit structure, comparing scale and dimension, composing units, or defining named, derived, or scaled custom units. |
| `references/temperature-conversions.md` | Converting absolute temperatures or temperature differences where Celsius offsets matter. |
| `references/prefix-library.md` | Looking up predefined SI base or derived units, generated unit names, prefix symbols, and prefix scales. |
| `references/physical-constant-library.md` | Looking up predefined constant names, values, dimensions, and canonical units. |

## Official Sources

- https://brainx.chaobrain.com/brainunit/
- https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html
- https://brainx.chaobrain.com/brainunit/physical_units/quantity.html
- https://brainx.chaobrain.com/brainunit/physical_units/standard_units.html
- https://brainx.chaobrain.com/braincell/concepts/units.html
