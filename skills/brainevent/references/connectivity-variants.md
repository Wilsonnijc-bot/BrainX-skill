# BrainEvent Connectivity Variants

Use this reference only after selecting the connectivity family in the skill body. It contains the subvariants intentionally omitted from the canonical path.

## JITC Weight Distribution

| Variant family | Stored weight rule | Use when |
|---|---|---|
| `JITCScalarR` / `JITCScalarC` | One shared connection weight | All realized connections use the same weight |
| `JITCNormalR` / `JITCNormalC` | Normal distribution with location and scale | Generated weights need unbounded variability around a mean |
| `JITCUniformR` / `JITCUniformC` | Uniform distribution between low and high bounds | Generated weights must remain in a bounded interval |

```python
import brainevent

shape = (1000, 500)

scalar = brainevent.JITCScalarR(
    (0.1, 0.10, 42),                 # weight, probability, seed
    shape=shape,
)
normal = brainevent.JITCNormalR(
    (0.0, 0.1, 0.10, 42),           # mean, std, probability, seed
    shape=shape,
)
uniform = brainevent.JITCUniformR(
    (-0.1, 0.3, 0.10, 42),          # low, high, probability, seed
    shape=shape,
)
```

All JITC variants regenerate connectivity from their parameters and seed. Do not use them when individual edges must be inspected, mutated, or learned.

## Row And Column Orientation

JITC uses `R` and `C` class suffixes for row- and column-oriented variants. The official selection guide says to choose the orientation matching the indexed dimension and contiguous contraction layout; benchmark both when uncertain.

```python
row_oriented = brainevent.JITCScalarR((0.1, 0.05, 7), shape=(1000, 500))
column_oriented = brainevent.JITCScalarC((0.1, 0.05, 7), shape=(1000, 500))
```

Use `brainevent.benchmark_function()` to compare the actual workload rather than inferring performance only from the suffix.

## Fixed Fan-In: `FixedPreNumConn`

`FixedPreNumConn` gives every postsynaptic neuron the same number of input connections. Its `data` and `indices` arrays have shape `(num_post, connections_per_post)`; each index identifies a presynaptic neuron.

```python
import brainevent
import jax.numpy as jnp

# Four presynaptic neurons, three postsynaptic neurons,
# and exactly two inputs received by every postsynaptic neuron.
indices = jnp.array([
    [0, 2],
    [1, 3],
    [0, 3],
])
weights = jnp.array([
    [0.5, 0.3],
    [0.4, 0.7],
    [0.2, 0.2],
])
connectivity = brainevent.FixedPreNumConn(
    (weights, indices),
    shape=(4, 3),
)

spikes = brainevent.BinaryArray([1, 0, 1, 0])
postsynaptic_input = spikes @ connectivity
```

Use `FixedPreNumConn` when fixed fan-in is the topology. Use the body-level `FixedPostNumConn` pattern when fixed fan-out is the topology.

## Sources

- Tutorial 3: JIT Connection Matrices: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/03_jit_connectivity.html
- Tutorial 4: Fixed Connection Count Structures: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/04_fixed_connections.html
- Choose a connectivity format: https://brainx.chaobrain.com/brainevent/how-to/data-structures/choosing-a-sparse-format.html
- Sparse Matrix Data Structures API: https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html
- Utility Functions API: https://brainx.chaobrain.com/brainevent/reference/apis/utilities.html
