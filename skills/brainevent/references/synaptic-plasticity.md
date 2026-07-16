# BrainEvent Synaptic Plasticity

Use this reference when binary pre- or postsynaptic events must modify stored synaptic weights. Plasticity is an overlay on the core `BinaryArray @ connectivity` workflow, not a prerequisite for ordinary event-driven communication.

BrainEvent's event-driven plasticity operators touch only weights connected to neurons that fired, following the same event-sparsity principle as its matrix products.

## Choose The Update Operator

| Weight storage | Presynaptic event triggers update | Postsynaptic event triggers update | Use when |
|---|---|---|---|
| CSR | `update_csr_on_binary_pre()` | `update_csr_on_binary_post()` | Connectivity is sparse and fixed; only stored synapses should be visited |
| Dense | `update_dense_on_binary_pre()` | `update_dense_on_binary_post()` | A small fully connected layer already uses a dense weight matrix |

Use the `*_on_binary_pre` direction when the source neuron firing triggers the rule. Use the `*_on_binary_post` direction when the target neuron firing triggers it.

## Canonical CSR STDP Overlay

The official tutorial maintains exponentially decaying pre/post traces, applies a spike-triggered update to the CSR `data`, and rebuilds the CSR object while preserving its structural arrays.

```python
import brainevent
import jax.numpy as jnp

# Existing CSR connectivity and traces:
# csr_weights: brainevent.CSR
# pre_spike: shape (num_pre,)
# post_spike: shape (num_post,)
# pre_trace: shape (num_pre,)
# post_trace: shape (num_post,)

pre_trace = pre_trace * decay_pre + pre_spike.astype(jnp.float32)
post_trace = post_trace * decay_post + post_spike.astype(jnp.float32)

# Presynaptic event-triggered update using the postsynaptic trace.
new_data = brainevent.update_csr_on_binary_pre(
    weight=csr_weights.data,
    indices=csr_weights.indices,
    indptr=csr_weights.indptr,
    pre_spike=pre_spike,
    post_trace=post_trace * A_plus,
    w_min=0.0,
    w_max=1.0,
    shape=csr_weights.shape,
)

csr_weights = brainevent.CSR(
    (new_data, csr_weights.indices, csr_weights.indptr),
    shape=csr_weights.shape,
)
```

Use when: a presynaptic spike should update its outgoing stored CSR synapses according to a postsynaptic trace. Preserve `indices`, `indptr`, and `shape`; the plasticity operator updates weight values, not the sparse topology.

For a bidirectional STDP rule, combine the corresponding pre- and post-triggered operators with the appropriate traces and signs from the learning rule. Do not silently substitute the pre-triggered operator for a post-triggered update.

## Storage Boundary

- Prefer CSR plasticity for large sparse networks with fixed connectivity.
- Prefer dense plasticity for small fully connected layers already represented densely.
- Do not choose JITC when individual connection weights must be persistently updated; use stored CSR or dense weights.

## Exact API Routing

Open the Matrix Operations API for exact signatures and associated primitives:

- CSR: `update_csr_on_binary_pre`, `update_csr_on_binary_post`, and their `*_p` primitives.
- Dense: `update_dense_on_binary_pre`, `update_dense_on_binary_post`, and their `*_p` primitives.

API: https://brainx.chaobrain.com/brainevent/reference/apis/operations.html

## Sources

- Tutorial 5: Synaptic Plasticity Modeling - Foundation of Learning and Memory: https://brainx.chaobrain.com/brainevent/tutorials/data-structures/05_synaptic_plasticity.html
- Apply event-driven synaptic plasticity: https://brainx.chaobrain.com/brainevent/how-to/data-structures/synaptic-plasticity.html
