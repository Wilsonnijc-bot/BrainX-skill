---
name: brainunit-quantity-safety
description: Explains BrainUnit quantity safety, including physical units, dimensional checks, conversions, scalar and array quantities, and safe boundaries with external libraries. Use when working with voltage, current, time, concentration, conductance, BrainCell physical quantities, unit errors, dimensional mismatches, or suspicious bare numbers.
---

brainunit-quantity-safety/

Concepts

• Why BrainUnit exists
brainunit is a unit-aware scientific computing library built on JAX; it tracks physical units through computations and catches dimension errors at runtime.
Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

• Creating Quantities
A Quantity = numeric value + physical unit. Create one by multiplying a value with a unit.
Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

• Arithmetic
Units are tracked automatically. Incompatible operations raise errors.
Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

• Unit conversion
Use to_decimal() to extract the numeric value in a target unit, or in_unit() to get a new Quantity in the target unit.
Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

• Quantity attributes
A quantity exposes mantissa, unit, dim, shape, and dtype.
Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

• Unit-aware math
brainunit.math provides 500+ functions that understand units.
Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

• BrainCell connection
In braincell, every physical quantity carries an explicit unit; passing a bare number where a quantity is expected raises TypeError.
Source: https://brainx.chaobrain.com/braincell/concepts/units.html

Evoke case / task boundary

• use this skill whenever code involves voltage, current, conductance, capacitance, time, length, concentration, temperature, or physical model parameters.
• use this skill before BrainCell / BrainState biological or dynamical simulation code.
• do not silently strip units to raw floats.
• do not mix dimensions and hope the math works.
• do not use plain jnp math when u.math is the documented unit-aware path.
• convert to raw numbers only at explicit formula / API boundaries with target units.

Creating quantities

Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

Script

import brainunit as u
import jax
import jax.numpy as jnp
# Scalars
mass = 5.0 * u.kilogram
speed = 10.0 * u.meter / u.second
print('mass:', mass)
print('speed:', speed)
# Arrays
voltages = jnp.array([1.0, 2.5, 3.7]) * u.mV
print('voltages:', voltages)
print('shape:', voltages.shape, 'dtype:', voltages.dtype)
# Direct construction
current = u.Quantity(jnp.array([0.1, 0.2, 0.3]), unit=u.ampere)
print('current:', current)

Explanation text

Core mental model: numeric value + physical unit. Attach units at creation, not after simulation.

Arithmetic and dimension mismatch

Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

Script

# Addition: same dimension required
t1 = 500.0 * u.ms
t2 = 1.5 * u.second
print('t1 + t2:', t1 + t2)  # auto-aligns to first unit
# Multiplication: units multiply
F = 10.0 * u.newton
d = 3.0 * u.meter
print('work = F * d:', F * d)  # N * m = J
# Division: units divide
print('speed = d / t:', (100.0 * u.meter) / (10.0 * u.second))  # m/s
# Dimension mismatch raises error
try:
    result = 5.0 * u.meter + 3.0 * u.second
except Exception as e:
    print('Error:', e)

Explanation text

Arithmetic is the guardrail: same-dimension addition, unit multiplication/division, and immediate error on incompatible dimensions.

Unit conversion and raw-value extraction

Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

Script

distance = 2.5 * u.kmeter
print('In meters:', distance.to_decimal(u.meter))       # 2500.0
print('In cm:', distance.to_decimal(u.cmeter))           # 250000.0
print('As Quantity:', distance.in_unit(u.meter))          # 2500.0 m

Explanation text

Use to_decimal(target_unit) only when raw numeric values are actually needed; otherwise prefer keeping a Quantity with in_unit(...).

Unit-aware math and transforms

Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

Script

data = jnp.array([2., 4., 6., 8., 10.]) * u.newton
print('sum:', u.math.sum(data))           # keeps unit
print('mean:', u.math.mean(data))         # keeps unit
print('sqrt:', u.math.sqrt(4.0 * u.meter2))  # changes unit: m^2 -> m
print('sort:', u.math.sort(jnp.array([3., 1., 2.]) * u.volt))
# JIT compilation
@jax.jit
def kinetic_energy(m, v):
    return 0.5 * m * v**2
KE = kinetic_energy(2.0 * u.kilogram, 3.0 * u.meter / u.second)
print('KE =', KE)  # kg * m^2 / s^2 = J

Explanation text

Quantities work with JAX transforms, and u.math keeps functions unit-aware.

Function unit contracts

Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html

Script

@u.check_units(v=u.meter / u.second, t=u.second)
def displacement(v, t):
    return v * t
print('displacement:', displacement(10.0 * u.meter / u.second, 5.0 * u.second))
# Wrong units raise an error
try:
    displacement(10.0 * u.kilogram, 5.0 * u.second)
except Exception as e:
    print('Error:', e)

Explanation text

Use @check_units to enforce unit contracts on function arguments.

Full bundled script references

brainunit-quickstart-quantity-safety.py

Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html
Purpose: complete quickstart workflow: quantities, arithmetic, conversion, attributes, math, constants, transforms, decorators.

unit-aware-jax-transform.py

Source: https://brainx.chaobrain.com/brainunit/getting_started/quickstart.html
Purpose: complete JIT / vmap / grad example with unit tracking.

braincell-unit-parameter-pattern.py

Source: https://brainx.chaobrain.com/braincell/concepts/units.html
Purpose: official BrainCell quantity creation pattern for biological parameters.

Common mistakes -> Fix

• using raw numbers for voltage/current/time -> multiply by u.mV, u.nA, u.ms, etc.
• adding incompatible quantities -> let BrainUnit raise; fix dimensions, not just units.
• extracting .mantissa or raw values early -> use to_decimal(target_unit) only at boundary.
• using jnp.sum / generic math where units must survive -> use u.math functions.
• missing function-level unit checks -> use @u.check_units(...) for public helper functions.
• passing bare numbers into BrainCell -> attach explicit units before constructing mechanisms or parameters.

