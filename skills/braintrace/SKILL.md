---
name: braintrace
description: Use when working with BrainTrace, ETP primitives, D-RTRL, ES-D-RTRL, pp-prop, online learning, eligibility traces, BrainTrace RNNs/SNNs, compiler graphs, hidden states, vmap batching, or BrainTrace debugging.
---

# BrainTrace Online Learning

## Purpose And Boundary

Use this skill when the user is designing, implementing, training, debugging, or extending a `braintrace` online-learning model.

Do not teach general ML, JAX, Python, or neuroscience background. Use official BrainX / BrainTrace snippets only. For exact APIs or lower-level internals, open the relevant reference markdown.

## P0 Concepts

- **Online learning**
  Weights are updated at each time step using eligibility traces that summarize the gradient history.
  Source: https://brainx.chaobrain.com/braintrace/quickstart/concepts.html

- **Eligibility traces**
  Eligibility traces accumulate the information needed for gradient computation incrementally.
  Source: https://brainx.chaobrain.com/braintrace/quickstart/concepts.html

- **braintrace.nn first**
  BrainTrace provides pre-built layers in the `braintrace.nn` module that already use ETP primitives internally.
  Source: https://brainx.chaobrain.com/braintrace/quickstart/concepts.html

- **braintrace.compile**
  `braintrace.compile(model, algo, x0, batch_size=B)` is the one-call entry point that initialises states, builds the trace graph, and returns a ready learner.
  Source: https://brainx.chaobrain.com/braintrace/quickstart/concepts.html

- **Hidden states**
  The compiler identifies which state variables are both read and written during a forward pass, then groups related hidden states into hidden groups.
  Source: https://brainx.chaobrain.com/braintrace/tutorials/hidden_states.html

- **Compiled graph**
  Models are compiled into an `ETraceGraph`, an intermediate representation that captures relationships between weight parameters, ETP primitives, and hidden state groups.
  Source: https://brainx.chaobrain.com/braintrace/tutorials/graph_visualization.html

- **Algorithm choice**
  `D_RTRL` is for RNNs and general-purpose use; `ES_D_RTRL` is for large SNNs and memory-constrained use.
  Source: https://brainx.chaobrain.com/braintrace/apis/algorithms.html

## Primitive Boundary

Default path:

1. Prefer `braintrace.nn` modules.
2. Compile with `braintrace.compile`.
3. Inspect `learner.report` / `show_graph()` if inclusion is unclear.

Only use primitive references when:

1. the user writes a custom layer,
2. the user directly calls `braintrace.matmul`, `element_wise`, `conv`, `sparse_matmul`, or `lora_matmul`,
3. the user needs parameter transforms such as mask / sign / standardization / squash,
4. the user registers a custom ETP primitive,
5. the user asks why a parameter is or is not eligibility-traced.

## Canonical BrainTrace Workflow

Use this when the user asks for the default BrainTrace online-learning path.

```python
import jax
import jax.numpy as jnp
import brainstate
import braintrace

class GRUNet(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.rnn = braintrace.nn.GRUCell(n_in, n_rec)
        self.readout = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.readout(self.rnn(x))

# Step 1: Define model
model = GRUNet(10, 64, 10)

# Step 2: Compile -- initialises states, builds the trace graph, returns a ready learner.
# The example input carries the batch axis: shape (batch_size, n_in) = (1, 10).
trainer = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros((1, 10)), batch_size=1)

# Step 3: Use standard JAX gradient computation
# The eligibility traces are updated inside trainer(x)
weights = model.states(brainstate.ParamState)

def loss_fn(x):
    out = trainer(x)
    return jnp.mean(out ** 2)

grad_fn = brainstate.transform.grad(loss_fn, weights)
grads = grad_fn(jnp.ones((1, 10)))   # input carries the batch axis (1, 10)
```

Source: https://brainx.chaobrain.com/braintrace/quickstart/concepts.html

Using BrainTrace for online learning follows a simple three-step workflow: define the model using `braintrace.nn` modules or manual ETP primitives; compile with `braintrace.compile`; train with standard JAX gradient computation.

## Hidden State Pattern

Use this when the user writes a custom recurrent module.

```python
import jax
import jax.numpy as jnp
import brainstate
import braintrace

class SimpleNeuron(brainstate.nn.Module):
    """A minimal recurrent neuron with a single hidden state."""
    def __init__(self, size):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(size, size) * 0.01)
        self.h = brainstate.HiddenState(jnp.zeros(size))

    def update(self, x):
        # braintrace.matmul marks w as participating in online learning
        self.h.value = jax.nn.tanh(x + braintrace.matmul(self.h.value, self.w.value))
        return self.h.value
```

Source: https://brainx.chaobrain.com/braintrace/tutorials/hidden_states.html

`brainstate.HiddenState` manages exactly one state tensor; use it when the neuron or synapse has a single recurrent variable.

For multiple correlated same-shape states, use `brainstate.HiddenGroupState`; for named same-shape state collections, use `brainstate.HiddenTreeState`.

## Inspect Graph Before Trusting Gradients

Use this when the model is nontrivial, custom, recurrent, multi-layer, or has unexpected excluded weights.

```python
import jax
import jax.numpy as jnp
import brainstate
import braintrace

class SingleLayerRNN(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.rnn = braintrace.nn.ValinaRNNCell(n_in, n_rec)
        self.out = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.out(self.rnn(x))

model = SingleLayerRNN(10, 32, 5)

# braintrace.compile initialises states, compiles the ETP graph, and returns a ready learner.
# We compile for a single unbatched sample (no batch_size), so the hidden state is (32,) and
# the recurrent op is the matrix-vector primitive etp_mv. verbose=2 prints full diagnostics.
learner = braintrace.compile(model, braintrace.D_RTRL, jnp.zeros(10), verbose=2)
learner.show_graph()

# report.show(level) prints a structured summary at the requested verbosity.
# level=1 shows hidden groups, etrace weights, and excluded weights.
learner.report.show(1)

# Programmatic access to the summary counts
print("Counts:", learner.report.counts)

# Which weights participate in online learning?
print("ETrace weights:", learner.report.etrace_weights)

# Which weights were excluded (e.g., non-temporal readouts)?
print("Excluded weights:", learner.report.excluded_weights)
```

Source: https://brainx.chaobrain.com/braintrace/tutorials/graph_visualization.html

`show_graph()` visualizes hidden groups, weight-primitive-hidden connections, and non-ETP weights.

Every learner returned by `braintrace.compile` carries a `CompilationReport` at `learner.report`.

## Vmap Batching Default

Use this when the user has batched data.

```python
import jax
import jax.numpy as jnp
import brainstate
import braintools
import braintrace

class SimpleGRU(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.rnn = braintrace.nn.GRUCell(n_in, n_rec)
        self.out = braintrace.nn.Linear(n_rec, n_out)

    def update(self, x):
        return self.out(self.rnn(x))

model = SimpleGRU(10, 64, 5)
batch_size = 16

# braintrace.compile with vmap=True:
#   - initialises per-sample hidden states (batch_size independent copies)
#   - compiles the ETP graph from a single batched time step: axis 0 is the batch
#     axis, which compile strips internally to recover the per-sample example
#   - wraps the result in brainstate.nn.Vmap for parallel execution
algo_vmapped = braintrace.compile(
    model, braintrace.D_RTRL, jnp.zeros((batch_size, 10)),
    batch_size=batch_size, vmap=True,
)

# Run on batched input -- the returned learner handles the batch axis transparently
x_batch = jnp.ones((batch_size, 10))
out = algo_vmapped(x_batch)
print("Output shape:", out.shape)  # (16, 5)
```

Source: https://brainx.chaobrain.com/braintrace/tutorials/batching.html

This pattern keeps the model definition simple, with single-sample logic, while gaining efficient batch parallelism automatically.

## Algorithm Routing

Use only this routing in the skill body. Open `skills/braintrace/references/algorithms-and-customization.md` for detailed choices.

- **braintrace.D_RTRL**
  Use for rate-based RNNs where accurate temporal gradient propagation is needed and parameter-count memory is acceptable.
  Source: https://brainx.chaobrain.com/braintrace/quickstart/concepts.html

- **braintrace.ES_D_RTRL / braintrace.pp_prop**
  Use for spiking neural networks or very large recurrent networks.
  Source: https://brainx.chaobrain.com/braintrace/quickstart/concepts.html

- **SNN paper algorithms**
  Paper-faithful algorithms tailored to spiking neural networks are approximate except where a regime makes them exact; know the regime before relying on their gradients.
  Source: https://brainx.chaobrain.com/braintrace/apis/algorithms.html

## References

Reference quick map:

```text
skills/braintrace/references/primitive-ops-and-transforms.md
skills/braintrace/references/algorithms-and-customization.md
skills/braintrace/references/compiler-graph-debugging.md
skills/braintrace/references/state-batching-workflows.md
skills/braintrace/references/braintools-metrics.md
skills/braintrace/references/braintools-optimizer.md
```

### `skills/braintrace/references/primitive-ops-and-transforms.md`

Source:

- https://brainx.chaobrain.com/braintrace/quickstart/concepts.html
- https://brainx.chaobrain.com/braintrace/advanced/compiler_internals.html
- https://brainx.chaobrain.com/braintrace/tutorials/etp_primitives.html
- https://brainx.chaobrain.com/braintrace/tutorials/customizing_primitive_transforms.html
- https://brainx.chaobrain.com/braintrace/apis/primitives.html

Purpose: one operation-level reference for built-in ETP primitives, `braintrace.matmul`, conv/sparse/LoRA/element-wise ops, parameter transform hooks, and custom ETP primitive registration.

### `skills/braintrace/references/algorithms-and-customization.md`

Source:

- https://brainx.chaobrain.com/braintrace/apis/algorithms.html
- https://brainx.chaobrain.com/braintrace/advanced/custom_algorithms.html

Purpose: sharply explains each algorithm choice one by one, then covers the advanced algorithm hierarchy and custom algorithm extension points.

### `skills/braintrace/references/compiler-graph-debugging.md`

Source:

- https://brainx.chaobrain.com/braintrace/advanced/compiler_internals.html
- https://brainx.chaobrain.com/braintrace/advanced/limitations.html
- https://brainx.chaobrain.com/braintrace/tutorials/graph_visualization.html

Purpose: one compiler/debugging reference for `ETraceGraph`, module-info extraction, hidden-group discovery, ETP relations, diagnostics, excluded weights, limitations, and workarounds.

### `skills/braintrace/references/state-batching-workflows.md`

Source:

- https://brainx.chaobrain.com/braintrace/tutorials/hidden_states.html
- https://brainx.chaobrain.com/braintrace/tutorials/batching.html

Purpose: deeper reference for hidden-state variants, state initialization/reset, single-sample mode, vmap batching, and multi-step data handling.

### `skills/braintrace/references/braintools-metrics.md`

Source:

- https://brainx.chaobrain.com/braintools/apis/metric.html

Purpose: select losses and evaluation metrics for online-learning workflows.

### `skills/braintrace/references/braintools-optimizer.md`

Source:

- https://brainx.chaobrain.com/braintools/apis/optim.html
- https://brainx.chaobrain.com/braintools/optim/index.html

Purpose: select optimizers and learning-rate schedules for online parameter updates.

## Script References

### Core Quickstart Workflows (2)

#### rnn-online-learning.py

Source: https://brainx.chaobrain.com/braintrace/quickstart/rnn_online_learning.html

Purpose: complete executable GRU copying-memory workflow using `braintrace.nn`, `braintrace.D_RTRL`, online training, and BPTT comparison.

#### snn-online-learning.py

Source: https://brainx.chaobrain.com/braintrace/quickstart/snn_online_learning.html

Purpose: complete executable recurrent SNN workflow using `brainstate` neurons, `braintrace.nn` layers, and `braintrace.ES_D_RTRL`.

### Default Scripts (6)

#### Algorithm Family Representative Workflows

These are the two main algorithm-family scripts.

##### examples/drtrl/09-classification-mnist.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/09-classification-mnist.py

Category: D_RTRL representative workflow

Purpose: flagship D_RTRL classification script; treats MNIST as row-scan sequence data and compares D_RTRL with BPTT using matched hyperparameters.

##### examples/pp_prop/12-classification-neuromorphic.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/12-classification-neuromorphic.py

Category: pp_prop / ES_D_RTRL representative workflow

Purpose: flagship pp_prop classification script; trains a LIF RSNN on Poisson-MNIST and compares pp_prop with BPTT.

#### Batching Modes

Keep exactly two batching scripts: one canonical vmap pattern and one batched-primitive alternative.

##### examples/drtrl/02-batching-vmap.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/02-batching-vmap.py

Category: batching / vmap

Purpose: canonical per-sample-state batching pattern; shows explicit state initialization, graph compilation, and wrapping the online model in `brainstate.nn.Vmap`.

##### examples/pp_prop/06-batching-batched.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/06-batching-batched.py

Category: batching / batched primitive

Purpose: important alternative to vmap; inputs carry a batch dimension directly and the ETP primitive path handles batched execution.

#### Temporal-Credit / VJP Method Choice

Keep one contrast script. Do not keep separate single-step and multi-step scripts.

##### examples/pp_prop/14-knob-vjp-method-contrast.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/14-knob-vjp-method-contrast.py

Category: VJP method contrast

Purpose: compares `vjp_method='single-step'`, `vjp_method='multi-step'`, and BPTT head-to-head on delayed match-to-sample.

#### Performance Knob

Keep one performance-knob script.

##### examples/drtrl/11-knob-fast-solve.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/11-knob-fast-solve.py

Category: D_RTRL performance knob

Purpose: compares `fast_solve=True` and `fast_solve=False`, checks numerical equivalence, and measures wall-clock speedup.

### Operator Scripts (3)

These are valuable, but they should be opened only when the user is working below `braintrace.nn` or asking about custom layers, ETP operators, LoRA, sparse connectivity, or convolutional online learning.

Do not treat these as default workflow scripts.

#### examples/drtrl/07-operator-lora.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/07-operator-lora.py

Category: operator variation / LoRA

Purpose: best primitive-selection script; freezes a base recurrent weight through plain `x @ w` and trains only LoRA factors routed through `braintrace.lora_matmul`.

#### examples/pp_prop/09-operator-sparse.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/09-operator-sparse.py

Category: operator variation / sparse or masked recurrent connectivity

Purpose: shows sparse recurrent connectivity through a fixed mask and explains the masked-dense fallback path.

#### examples/pp_prop/11-operator-conv.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/11-operator-conv.py

Category: operator variation / convolution

Purpose: convolutional SNN reference using `braintrace.nn.Conv2d`, where pp_prop dispatches to the convolutional ETP primitive for kernel gradients.

### Optional Specialized Scripts (2)

Do not bundle these by default. Add only if the skill is expected to support benchmarking or biophysical SNN architectures often.

#### examples/003-snn-memory-and-speed-evaluation-all.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/003-snn-memory-and-speed-evaluation-all.py

Category: benchmark / memory / speed

Purpose: heavy benchmark comparing BPTT, ES_D_RTRL, and D_RTRL across sequence lengths, training time, and memory usage.

#### examples/pp_prop/04-neurons-coba-ei-rsnn.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/04-neurons-coba-ei-rsnn.py

Category: specialized SNN architecture

Purpose: Dale-law excitatory/inhibitory recurrent SNN example; useful only when the user asks for COBA/EI or biologically constrained SNN architecture.

### Helper Dependencies (2)

Bundle only if selected scripts import them. Do not present these as teaching scripts.

#### examples/pp_prop/_shared.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/pp_prop/_shared.py

Purpose: support file for pp_prop scripts: shared cells, datasets, readouts, training helpers, BPTT helpers, and plotting utilities.

#### examples/drtrl/_shared.py

Source: https://github.com/chaobrain/braintrace/blob/main/examples/drtrl/_shared.py

Purpose: support file for D_RTRL scripts: shared task generators, model helpers, training helpers, BPTT helpers, and plotting utilities.

### Final Default Bundle

Use this as the default script bundle:

```text
examples/drtrl/09-classification-mnist.py
examples/pp_prop/12-classification-neuromorphic.py
examples/drtrl/02-batching-vmap.py
examples/pp_prop/06-batching-batched.py
examples/pp_prop/14-knob-vjp-method-contrast.py
examples/drtrl/11-knob-fast-solve.py
```

## Common Mistakes -> Fix

- Using `x @ w` for temporal recurrent weights -> use `braintrace.matmul`; regular JAX ops exclude the weight from online learning.
  Source: https://brainx.chaobrain.com/braintrace/quickstart/concepts.html

- Rewriting a raw parameter for masks/sign constraints/standardization -> use the ETP op's shape-preserving `*_fn` hook so the forward pass uses transformed `V = f(W)` while traces and gradients stay attached to raw `W`.
  Source: https://brainx.chaobrain.com/braintrace/tutorials/customizing_primitive_transforms.html

- Putting `jax.lax.cond`, `jax.lax.scan`, `jax.lax.while_loop`, or nested `jax.vmap` inside `update()` -> move them outside the model update path or use supported selection/masking workarounds.
  Source: https://brainx.chaobrain.com/braintrace/advanced/limitations.html

- Calling `braintrace.compile` inside the training loop -> call it once before the training loop; the compiled graph is cached and reused for inputs of the same shape.
  Source: https://brainx.chaobrain.com/braintrace/advanced/limitations.html

- Expecting readout weights to always be eligibility-traced -> readout output may be the final output and never flow back into a hidden state; the compiler can report it as non-temporal but still trained.
  Source: https://brainx.chaobrain.com/braintrace/tutorials/graph_visualization.html

- Skipping graph inspection for a custom model -> use `learner.show_graph()` and `learner.report.show(1)` to verify hidden groups, ETrace weights, and excluded weights.
  Source: https://brainx.chaobrain.com/braintrace/tutorials/graph_visualization.html

- Manually designing batched recurrent state first -> prefer `braintrace.compile(..., batch_size=B, vmap=True)`; the model itself only sees single-sample inputs.
  Source: https://brainx.chaobrain.com/braintrace/tutorials/batching.html
