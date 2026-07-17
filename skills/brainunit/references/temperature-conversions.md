# Temperature Conversions

Use this reference when an interface supplies Celsius values, a model stores absolute temperature in kelvin, or code must distinguish an absolute temperature from a temperature difference.

## Affine Boundary

Celsius is an offset scale. BrainUnit uses kelvin for temperature `Quantity` objects and provides two explicit conversion functions:

| Function | Required input | Output | Formula |
|---|---|---|---|
| `u.celsius2kelvin(x)` | Plain numeric Celsius value or array | Kelvin `Quantity` | `K = C + 273.15` |
| `u.kelvin2celsius(q)` | Temperature `Quantity` in kelvin | Plain Celsius value or array | `C = K - 273.15` |

```python
import brainunit as u
import jax.numpy as jnp

celsius = jnp.array([-40.0, 0.0, 20.0, 37.0, 100.0])
kelvin = u.celsius2kelvin(celsius)
roundtrip = u.kelvin2celsius(kelvin)
```

The official examples include:

```text
0 C       -> 273.15 K
100 C     -> 373.15 K
37 C      -> 310.15 K
-273.15 C -> 0 K
```

## Absolute Temperature Versus Difference

Store absolute temperatures as kelvin quantities:

```python
room_temp = 293.15 * u.kelvin
delta_T = 10.0 * u.kelvin
warmer = room_temp + delta_T
difference = 373.15 * u.kelvin - room_temp
```

In the official output, `warmer` is `303.15 K` and `difference` is `80 K`. The conversion offset is for absolute Celsius-to-kelvin boundaries, not for adding or subtracting a temperature difference already expressed in kelvin.

## Zero Celsius Constant

`u.constants.zero_celsius` is the predefined `273.15 K` quantity. The tutorial uses it for manual conversion:

```python
celsius_25 = 25.0
kelvin_25 = celsius_25 * u.kelvin + u.constants.zero_celsius
```

Prefer `celsius2kelvin()` and `kelvin2celsius()` at ordinary affine boundaries; use the constant when the explicit offset is part of a larger formula.

## Validation And Error Cases

- `celsius2kelvin()` rejects a `Quantity`; it expects a plain number interpreted as Celsius.
- `kelvin2celsius()` rejects a plain number; it expects a temperature `Quantity`.
- `kelvin2celsius()` rejects a `Quantity` with a non-temperature dimension.
- `kelvin2celsius()` returns a plain value rather than a `Quantity`.

```python
try:
    u.celsius2kelvin(100.0 * u.kelvin)
except TypeError:
    pass

try:
    u.kelvin2celsius(100.0 * u.meter)
except TypeError:
    pass
```

## Roundtrip Check

Use a roundtrip when validating an input adapter:

```python
original_c = 37.0
encoded = u.celsius2kelvin(original_c)
decoded_c = u.kelvin2celsius(encoded)
```

The official tutorial reports `37.0 C -> 310.15 K -> 37.0 C`.

## Sources Mirrored

- https://brainunit.readthedocs.io/physical_units/temperature.html
