---
name: brainevent-event-driven-connectivity
description: Use when representing binary spike events with BrainEvent, multiplying BinaryArray values through dense, CSR, JIT-generated, or fixed-degree connectivity, choosing among BrainEvent connectivity formats, applying event-driven plasticity, or routing custom operator work.
---

# BrainEvent Event-Driven Connectivity

## Purpose And Boundary

Use this skill to represent binary spike events and route them through BrainEvent connectivity. Keep neuron, synapse, channel, and neural-mass dynamics in their owning BrainX packages; BrainEvent supplies the event representation and event-driven communication layer.

Do not use `BinaryArray` for continuous analog activity. Do not use BrainEvent as the default coupling path for BrainMass models unless the model explicitly communicates with binary spike events.

## P0 Concept: `BinaryArray`

`BinaryArray` is BrainEvent's core data structure for binary event arrays. Wrapping spike data in `BinaryArray` makes subsequent matrix operations use event-driven optimization.

Source: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/01_eventarray_basics.html

```python
import brainevent
import jax.numpy as jnp

# True/1 means that the presynaptic neuron emitted an event.
spikes = brainevent.BinaryArray(
    jnp.array([True, False, True, False, True])
)
```

The left dimension of the connectivity must match the spike-vector length. The canonical call site is always:

```python
postsynaptic_input = spikes @ connectivity
```

## Choose A Connectivity Representation

| Representation | Connection rule | Use when | Avoid when |
|---|---|---|---|
| Dense JAX/NumPy array | Store every matrix entry | The matrix is small or genuinely dense (roughly more than 25% nonzero), or arbitrary per-edge weights matter more than sparse storage | A large matrix is mostly zeros |
| `CSR` | Store an explicit, fixed sparse matrix by row | Connectivity is known, reusable, and sparse; use for the normal forward `BinaryArray @ matrix` path | Random connectivity is too large to materialize |
| `JITCScalarR` | Regenerate fixed-probability random connections from weight, probability, and seed | The random network is extremely large and individual edges need not be inspected or changed | Individual weights must be inspected, mutated, or learned |
| `FixedPostNumConn` | Give every presynaptic neuron the same number of outputs | Fixed fan-out is part of the topology | Connection counts vary by neuron or an explicit general sparse matrix is required |

Representative rule: explicit and reusable -> `CSR`; random and huge -> `JITCScalarR`; fixed fan-out -> `FixedPostNumConn`. For COO/CSC, JITC distributions and orientations, or fixed fan-in, open the routed references instead of expanding the body.

Source: https://brainx.chaobrain.com/brainevent/how-to/data-structures/choosing-a-sparse-format.html

## Dense Connectivity

A dense JAX/NumPy array stores every weight. `BinaryArray @ dense_weights` still uses the event representation, but the matrix itself does not save sparse storage.

```python
import brainevent
import jax.numpy as jnp

spikes = brainevent.BinaryArray([1, 0, 1, 0, 1])
weights = jnp.array([
    [0.5, 0.2, 0.1],
    [0.3, 0.4, 0.2],
    [0.1, 0.5, 0.3],
    [0.2, 0.1, 0.4],
    [0.4, 0.3, 0.5],
])
postsynaptic_input = spikes @ weights
```

Use when: the matrix is small or genuinely dense, or the simplest arbitrary per-entry representation is required.

## Explicit Sparse Connectivity: `CSR`

CSR stores nonzero values, column indices, and row pointers. BrainEvent's tutorial recommends CSR with `BinaryArray` for most spiking-network forward propagation.

```python
import brainevent
import jax.numpy as jnp

shape = (4, 5)
row = jnp.array([0, 1, 2, 3])
col = jnp.array([1, 3, 0, 4])
data = jnp.array([1.5, 2.0, 0.5, 3.0])

indptr, indices, order = brainevent.coo2csr(row, col, shape=shape)
connectivity = brainevent.CSR(
    (data[order], indices, indptr),
    shape=shape,
)

spikes = brainevent.BinaryArray([1, 0, 1, 0])
postsynaptic_input = spikes @ connectivity
```

Use when: the sparse edges are explicit and reusable, per-edge values must remain accessible, or weights may later be updated. Keep `data[order]`: `coo2csr()` returns the permutation that aligns values with the compressed indices.

Open [references/sparse-formats.md](references/sparse-formats.md) for COO construction, CSR, CSC, and their selection rules.

Sources:

- https://brainx.chaobrain.com/brainevent/tutorials/data-structures/02_sparse_matrices.html
- https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html

## Generated Random Connectivity: `JITCScalarR`

JIT connectivity stores generation rules instead of pre-storing a weight matrix. A fixed seed reproduces the generated connectivity.

```python
import brainevent
import jax.numpy as jnp

n_pre = 100_000
n_post = 100_000

# (shared weight, connection probability, random seed)
connectivity = brainevent.JITCScalarR(
    (0.5, 0.01, 0),
    shape=(n_pre, n_post),
)

spikes = brainevent.BinaryArray(
    jnp.zeros(n_pre, dtype=bool).at[::1000].set(True)
)
postsynaptic_input = spikes @ connectivity
```

Use when: connectivity is random with fixed probability, the materialized matrix would not fit in memory, and individual edges do not need inspection or mutation. Keep the seed stable when the realized random graph must remain reproducible.

Open [references/connectivity-variants.md](references/connectivity-variants.md) for scalar/normal/uniform weights and row/column orientations.

Sources:

- https://brainx.chaobrain.com/brainevent/tutorials/data-structures/03_jit_connectivity.html
- https://brainx.chaobrain.com/brainevent/how-to/data-structures/jit-connectivity-large-networks.html

## Fixed Fan-Out Connectivity: `FixedPostNumConn`

`FixedPostNumConn` encodes a fixed number of output connections for every presynaptic neuron. Its `data` and `indices` arrays both have shape `(num_pre, connections_per_pre)`.

```python
import brainevent
import jax.numpy as jnp

# Four presynaptic neurons, three postsynaptic neurons,
# and exactly two outputs from every presynaptic neuron.
indices = jnp.array([
    [0, 2],
    [1, 2],
    [0, 1],
    [1, 2],
])
weights = jnp.array([
    [0.5, 0.2],
    [0.4, 0.1],
    [0.3, 0.6],
    [0.7, 0.2],
])
connectivity = brainevent.FixedPostNumConn(
    (weights, indices),
    shape=(4, 3),
)

spikes = brainevent.BinaryArray([1, 0, 1, 0])
postsynaptic_input = spikes @ connectivity
```

Use when: fixed fan-out is the intended topology and directly encoding it is clearer than a general sparse matrix.

Open [references/connectivity-variants.md](references/connectivity-variants.md) for `FixedPreNumConn`, which fixes the number of inputs received by each postsynaptic neuron.

Source: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/04_fixed_connections.html

## JAX Transform Pattern

Keep the connectivity object explicit when compiling the event-driven product.

```python
import jax

@jax.jit
def communicate(spikes, connectivity):
    return spikes @ connectivity

postsynaptic_input = communicate(spikes, connectivity)
```

## Synaptic Plasticity Overlay

Plasticity is not part of the P0 communication path. Use it as an overlay when spikes must update stored synaptic weights. BrainEvent provides pre- and post-synaptic event-driven update operators for CSR and dense storage; these operators touch only weights connected to neurons that fired.

Open [references/synaptic-plasticity.md](references/synaptic-plasticity.md) for operator selection, CSR versus dense routing, and the official CSR STDP update pattern.

Sources:

- https://brainx.chaobrain.com/brainevent/tutorials/data-structures/05_synaptic_plasticity.html
- https://brainx.chaobrain.com/brainevent/how-to/data-structures/synaptic-plasticity.html

## Custom Operator Routing

When built-in BrainEvent operations do not cover the required kernel, BrainEvent can be extended from high-level Numba/Warp decorators down to hand-written C++ and CUDA.

Open [references/custom-operators.md](references/custom-operators.md) to choose the official CPU or GPU custom-operator tutorial. Keep custom kernels out of the canonical path unless extension work is explicitly required.

Source: https://brainx.chaobrain.com/brainevent/tutorials/custom-operators/index.html

## Script References

Keep the two selected complete applications in the bundled scripts. Sparse-format, plasticity, and custom-operator examples stay inline in their Markdown references; do not promote them into additional standalone Python scripts.

- `references/scripts/102_EI_net_1996.py` is the direct full-script reference for a high-level E/I network. Use it when the surrounding model already uses `brainpy.state.AlignPostProj` and `brainstate.nn.EventFixedProb`; it does not construct a low-level `brainevent` matrix directly. Preserve its unit-aware E/I weights, state initialization, compiled time loop, and visualization. Upstream source: https://raw.githubusercontent.com/chaobrain/brainpy.state/main/examples/brainpy_like/102_EI_net_1996.py
- `references/scripts/204_joglekar_2018_propagation.py` belongs to the connectivity-variants branch for delayed spikes, JIT connectivity, and area mapping. Use it when delayed spikes must be wrapped with `BinaryArray`, multiplied by `JITCScalarC`, and vmapped across areas. Preserve its delays, area-specific weights, seeds, refractory masks, external data dependencies, and BrainPy version assumptions. Upstream source: https://raw.githubusercontent.com/chaobrain/brainpy.state/main/examples/brainpy_like/204_joglekar_2018_propagation.py

## Common Issues

- **Plain spike array at the event boundary:** wrap binary spikes with `brainevent.BinaryArray` before multiplying by connectivity.
- **Huge random matrix is materialized:** use a JITC representation if individual edges do not need inspection or learning.
- **JITC used for mutable per-edge weights:** use `CSR` or a dense matrix instead.
- **Wrong fixed-degree direction:** `FixedPostNumConn` fixes outputs per presynaptic neuron; `FixedPreNumConn` fixes inputs per postsynaptic neuron.
- **COO treated as a BrainEvent matrix class:** BrainEvent uses coordinate triplets as construction input; convert them with `coo2csr()`.
- **Orientation guessed from the class suffix:** route to the orientation reference and benchmark the actual contraction when performance matters.

## Reference Routing

These four files are the complete skill-local Markdown reference inventory:

| Reference | Open when |
|---|---|
| [references/sparse-formats.md](references/sparse-formats.md) | Explicit sparse connectivity needs COO construction, CSR/CSC storage, conversion, or format selection. |
| [references/connectivity-variants.md](references/connectivity-variants.md) | Generated connectivity needs JITC distributions or row/column orientation, or fixed-degree connectivity needs fan-in/fan-out selection. This single route owns both JIT and fixed-connection variants. |
| [references/synaptic-plasticity.md](references/synaptic-plasticity.md) | Pre- or postsynaptic events must update CSR or dense weights, including the STDP overlay. |
| [references/custom-operators.md](references/custom-operators.md) | Built-in operations are insufficient and the task requires a Numba, Numba CUDA, Warp, C++, or CUDA extension path. |

Keep `BinaryArray`, dense connectivity, the representative CSR/JITC/fixed fan-out workflows, and the JAX transform pattern in this body. Do not create separate Markdown routes for those canonical paths.

Exact API sources:

- https://brainx.chaobrain.com/brainevent/reference/apis/events.html — exact event-array APIs.
- https://brainx.chaobrain.com/brainevent/reference/apis/sparsedata.html — exact sparse and generated connectivity classes.
- https://brainx.chaobrain.com/brainevent/reference/apis/operations.html — event-driven products and plasticity operators.
