---
name: brainstate
description: Use when a task involves BrainState State, ParamState, HiddenState, `.value` mutation, PyTree state values, Module composition, parameter collection, collection utilities, model-wide lifecycle operations, mixins, parameter descriptors, State hooks, constrained or fixed parameters, training loops, losses, metrics, optimizers, StateFinder, graph inspection and reconstruction, Flax or Equinox interoperation, prebuilt layers, size inference, reproducible randomness, or state-aware `jit`, `grad`, and `vmap`.
---

# BrainState

## Purpose

Use this as the backbone for BrainState stateful programming, Module graphs, state traversal, parameter registration, reproducible randomness, and state-aware transformations. Keep the canonical path inline and open the routed references only for advanced traversal, collection organization, model-wide lifecycle operations, extension mechanisms, observability, architectures, dynamics, control flow, diagnostics, or performance work.

## Core Mental Model

BrainState programs are stateful Module graphs:

`small reusable Modules -> nested Module tree -> State objects at the leaves -> brainstate.transform handles stateful execution`

Core ideas:

- **State is where mutation happens.**
- Read and write ordinary State, including `ParamState`, through `.value`.
- `ParamState` is the standard trainable BrainState `State`: a mutable State leaf in the Module graph used for ordinary trainable weights, biases, and other unconstrained values.
- Use semantic subclasses such as `ParamState` for trainable parameters and `HiddenState` for dynamical or internal state.
- Build models from small reusable and composable `brainstate.nn.Module` objects.
- A model is a tree; `State` objects are usually the leaves.
- Assign a `State` or child `Module` to an attribute so BrainState can find it.
- Collect trainable parameters by type:

```python
params = model.states(brainstate.ParamState)
```

Pass this collection to `brainstate.transform.grad` and register the same collection with an optimizer. An `nn.Param` contributes its underlying `.val` `ParamState`; an `nn.Const` does not appear in the collection.

- Use `brainstate.transform`, not raw `jax.jit`, `jax.grad`, or `jax.vmap`, when code touches BrainState State.

## State Essentials

`State` is the container for mutable values in BrainState.

```python
import brainstate
import jax.numpy as jnp

s = brainstate.State(jnp.ones(3))
x = s.value
s.value = x + 1
```

Important rules:

- Only values inside State are mutable in BrainState-transformed code.
- Regular Python attributes are static configuration unless stored as State.
- Update State explicitly through `.value`.
- Keep state updates simple and local.
- Preserve the intended tree structure when a State contains a PyTree.
- Prefer semantic subclasses over plain `State` when the role is known.

### Create State values

```python
# Array state
v = brainstate.State(jnp.zeros(10))

# Scalar state
counter = brainstate.State(jnp.array(0))

# PyTree state
neuron_state = brainstate.State({
    "V": jnp.zeros(100),
    "u": jnp.zeros(100),
})
```

### Update PyTree State

A State value can be a JAX array or a PyTree such as a dictionary or list of arrays.

```python
state = brainstate.State({
    "neurons": {
        "V": jnp.zeros(5),
        "u": jnp.ones(5),
    },
    "synapses": {
        "g": jnp.zeros((5, 5)),
    },
})

value = state.value
state.value = {
    "neurons": {
        "V": value["neurons"]["V"] + 0.1,
        "u": value["neurons"]["u"],
    },
    "synapses": {
        "g": value["synapses"]["g"],
    },
}
```

Use tree checking while debugging structure-changing updates:

```python
with brainstate.check_state_value_tree():
    state.value = {
        "neurons": {
            "V": jnp.zeros(5),
            "u": jnp.ones(5),
        },
        "synapses": {
            "g": jnp.zeros((5, 5)),
        },
    }
```

### State subclasses

State subclasses are semantic markers used by traversal, filtering, gradient collection, debugging, and model organization.

| State subclass | Use for |
|---|---|
| `ParamState` | Trainable parameters such as weights and biases |
| `HiddenState` | Internal dynamical state such as membrane voltage or RNN hidden state |
| `ShortTermState` | Transient runtime state such as current input or last spike time |
| `LongTermState` | Persistent non-parameter state such as running statistics |

```python
w = brainstate.ParamState(brainstate.random.randn(10, 5) * 0.1)
b = brainstate.ParamState(jnp.zeros(5))
h = brainstate.HiddenState(jnp.zeros(5))
last_spike = brainstate.ShortTermState(jnp.full(5, -1e7))
running_mean = brainstate.LongTermState(jnp.zeros(5))
```

### Parameter choice

Use the smallest parameter abstraction that expresses the model:

```text
ordinary trainable value
    -> ParamState

trainable value requiring a valid domain, transform, or prior
    -> brainstate.nn.Param

fixed model value
    -> brainstate.nn.Const
```

Access is intentionally different:

- Ordinary `State`, including `ParamState`, stores its value in `.value`.
- `brainstate.nn.Param` exposes the constrained value used by computation through `.value()` and stores its underlying trainable `ParamState` in `.val`.
- `brainstate.nn.Const` represents a fixed value in the Module graph and is excluded from `model.states(brainstate.ParamState)`.

Keep transform choices, constraints, and regularization details in `references/brainstate/parameter-constraints-regularization.md` and its nested transform catalog.

## Fundamental Randomness

### The default `RandomState`

All functions in `brainstate.random` use the global `brainstate.random.DEFAULT` instance unless a separate key or `RandomState` is supplied.

```python
print("Default RandomState:")
print(brainstate.random.DEFAULT)
print(f"\nType: {type(brainstate.random.DEFAULT)}")
```

### Seed management and reproducibility

Set a seed before constructing randomly initialized Modules or generating random data. Resetting to the same seed reproduces the same sequence.

```python
brainstate.random.seed(42)
print("Seed set to 42")

x1 = brainstate.random.rand(5)
print(f"First sequence: {x1}")

brainstate.random.seed(42)
x2 = brainstate.random.rand(5)
print(f"Second sequence: {x2}")
print(f"\nIdentical? {jnp.allclose(x1, x2)}")
```

Use independent `RandomState` instances or explicit key management for separate streams, stochastic `vmap`, or checkpointed randomness; route those cases to the randomness references.

## Module Essentials

Use `brainstate.nn.Module` to build small reusable components, then compose or nest them into larger models. Modules organize computation and child Modules; State objects hold mutable values at the leaves.

### Add State to a Module

```python
class Counter(brainstate.nn.Module):
    """A Module that counts how many times it is called."""

    def __init__(self):
        super().__init__()
        self.count = brainstate.ShortTermState(jnp.array(0))

    def update(self, x):
        self.count.value = self.count.value + 1
        return x * self.count.value


counter = Counter()
print("Initial count:", counter.count.value)

for i in range(5):
    result = counter(jnp.array(10.0))
    print(f"Call {i + 1}: count={counter.count.value}, result={result}")
```

Expected result:

```text
Initial count: 0
Call 1: count=1, result=10.0
Call 2: count=2, result=20.0
Call 3: count=3, result=30.0
Call 4: count=4, result=40.0
Call 5: count=5, result=50.0
```

### Basic neural-network layers

| Layer type | Use for |
|---|---|
| `Linear` | Dense vector-to-vector mapping, MLPs, and classifier heads |
| `Conv1d` / `Conv2d` / `Conv3d` | Spatial or temporal local feature extraction |
| Pooling | Reducing spatial or temporal size |
| `Flatten` | Converting spatial features before dense classifier layers |
| Activations | Adding nonlinearity between learned layers |
| Normalization | Stabilizing activations or statistics during training |
| Dropout | Regularization during training |

Minimal prebuilt layer:

```python
brainstate.random.seed(42)
linear = brainstate.nn.Linear(in_size=(10,), out_size=(5,))
x = brainstate.random.randn(10)
y = linear(x)
```

Minimal activation layer:

```python
relu = brainstate.nn.ReLU()
sigmoid = brainstate.nn.Sigmoid()
tanh = brainstate.nn.Tanh()

x = brainstate.random.randn(5)
y = relu(x)
```

Activation functions determine a neuron or layer output from its input, usually by adding nonlinearity.

### Size inference, `Sequential`, and `.desc()`

- `in_size` is the input feature shape without the batch dimension.
- `out_size` is the output feature shape without the batch dimension.
- When `in_size` is known, output size is inferred automatically.
- In `Sequential`, one layer's `out_size` becomes the next layer's input size.
- Use `.desc()` when a layer should infer its input size from the preceding layer.

The following `ComplexNet` is the canonical inline example for `Sequential`, `.desc()`, and automatic size propagation. Use it directly when a user asks for the canonical example or its inferred sizes; do not ask whether to generalize it unless customization was requested. Route convolution formulas, pooling reduction, flatten-size inference, and edge cases to the planned consolidated `references/size-inference-variations.md` reference.

```python
class ComplexNet(brainstate.nn.Module):
    """Network demonstrating mixed layer types and size propagation."""

    def __init__(self, in_size):
        super().__init__()

        self.features = brainstate.nn.Sequential(
            brainstate.nn.Conv2d(
                in_size,
                out_channels=16,
                kernel_size=3,
                padding="SAME",
            ),
            brainstate.nn.ReLU(),
            brainstate.nn.Conv2d.desc(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding="SAME",
            ),
            brainstate.nn.ReLU(),
            brainstate.nn.Conv2d.desc(
                out_channels=64,
                kernel_size=3,
                padding="SAME",
            ),
            brainstate.nn.ReLU(),
            brainstate.nn.MaxPool2d.desc(
                kernel_size=(2, 2),
                stride=(2, 2),
                channel_axis=-1,
            ),
        )

        self.classifier = brainstate.nn.Sequential(
            brainstate.nn.Flatten(in_size=self.features.out_size),
            brainstate.nn.Linear.desc(out_size=(256,)),
            brainstate.nn.ReLU(),
            brainstate.nn.Linear.desc(out_size=(10,)),
        )

    def update(self, x):
        x = self.features(x)
        return self.classifier(x)


brainstate.random.seed(42)
net = ComplexNet(in_size=(32, 32, 3))

print("Complex Network:")
print(f"Input size: {net.features.in_size}")
print(f"After features: {net.features.out_size}")
print(f"After flatten: {net.classifier.layers[0].out_size}")
print(f"Final output: {net.classifier.out_size}")

x = brainstate.random.randn(2, 32, 32, 3)
y = net(x)
print(f"Forward pass: {x.shape} -> {y.shape}")
```

Expected size propagation and output:

```text
Complex Network:
Input size: (32, 32, 3)
After features: (8, 8, 64)
After flatten: (4096,)
Final output: (10,)
Forward pass: (2, 32, 32, 3) -> (2, 10)
```

## Transformation Essentials

`brainstate.transform` mirrors common JAX transforms but tracks State reads and writes so mutable BrainState programs remain correct under transformation.

Use:

- `brainstate.transform.jit` for a whole forward pass, simulation step, or training step.
- `brainstate.transform.grad` for differentiation with respect to a State collection.
- `brainstate.transform.vmap` for batching a single-example function.
- `jit(grad(...))` as the standard compiled training-step backbone.

### Minimal state-aware JIT

The first compatible call traces and compiles the function; later compatible calls reuse the compiled executable while BrainState writes State updates back.

```python
forward = brainstate.transform.jit(model)
y = forward(x)
```

### Minimal gradient and parameter update

Gradients use the same keys as the State collection passed to `grad`, so apply each `grads[key]` to the matching `params[key]`.

```python
params = model.states(brainstate.ParamState)


def loss_fn():
    pred = model(x)
    return jnp.mean((pred - target) ** 2)


grads, loss = brainstate.transform.grad(
    loss_fn,
    params,
    return_value=True,
)()

for key in params:
    params[key].value -= 0.1 * grads[key]
```

The collected `ParamState` mapping is the canonical gradient target. Reuse it for optimizer registration in a complete training loop; do not collect all State types.

### Composed training-step transform

```python
@brainstate.transform.jit
def train_step():
    grads, loss = brainstate.transform.grad(
        loss_fn,
        params,
        return_value=True,
    )()
    for key in params:
        params[key].value -= 0.1 * grads[key]
    return loss
```

### Minimal `vmap`

```python
def predict_one(x_row):
    return model(x_row[None, :])[0]


predict_batch = brainstate.transform.vmap(predict_one)
y = predict_batch(x)
```

## References and Routing

Open references only when the task exceeds this backbone.

### Canonical first-layer references

- `skills/brainstate/references/state-graph-operations.md`
  Use to find, extract, split, replace, and reconstruct State graphs.
- `skills/brainstate/references/model-interop-and-migration.md`
  Use for Flax or Equinox interoperation and PyTorch migration.
- `skills/brainstate/references/state_collections_and_utilities.md`
  Use to filter, organize, freeze, flatten, configure, and print nested collections.
- `skills/brainstate/references/collective_model_operations.md`
  Use for model-wide initialization, reset, method invocation, batched lifecycle operations, and State restoration.
- `skills/brainstate/references/extension_mechanisms.md`
  Use for mixins, descriptors, runtime modes, and State hooks.
- `skills/brainstate/references/size-inference-variations.md` [planned]
  Use for convolution formulas and edge cases, pooling reduction, and flatten-size inference. This consolidated target is not yet present; do not substitute either legacy split size-inference article.
- `skills/brainstate/references/brainstate/parameter-constraints-regularization.md`
  Use as the parameter parent for `ParamState` versus `nn.Param`, constraints and transforms, `nn.Const`, penalties, prior regularizers, and loss integration.
- `skills/brainstate/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`
  Use as the only first-layer randomness parent for streams, stochastic transforms, trials, dropout or noise, and checkpointed RNG State.
- `skills/brainstate/references/libraries/prebuilt-layer-library.md`
  Use for the full prebuilt layer catalog.
- `skills/brainstate/references/libraries/prebuilt-activation-library.md`
  Use for activation functions and normalization selection.
- `skills/brainstate/references/brainstate/transformation-jit-expansion.md`
  Use for State write-back, cache and static arguments, compilation boundaries, and benchmarking.
- `skills/brainstate/references/brainstate/transformation-grad-expansion.md`
  Use for autodiff, differentiable simulation, fitting, `return_value`, and `has_aux`.
- `skills/brainstate/references/brainstate/transformation-vmap-expansion.md`
  Use for State axes, ensembles, sweeps, stochastic `vmap`, `in_states`, and `out_states`.
- `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md`
  Use for transform-safe loops, scans, branches, and checkpointed control flow.
- `skills/brainstate/references/diagnostics/brainstate-transformed-diagnostics.md`
  Use as the diagnostics parent for runtime checks, transformed debugging, NaN or Inf checks, callbacks, and traced values.
- `skills/brainstate/references/braintools-optimizer-reference.md`
  Use for optimizer families, learning-rate schedulers, and external optimizer wrappers.
- `skills/brainx-acceleration-audit/SKILL.md`
  Route performance, batching, sweeps, memory, GPU, and multi-device work to the acceleration skill.

### Parent-only nested references

- Only `skills/brainstate/references/brainstate/parameter-constraints-regularization.md` may select `skills/brainstate/references/brainstate/parameter-containers-transforms-catalog.md` for exhaustive parameter-container and transform-class selection.
- Only `skills/brainstate/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` may select `skills/brainstate/references/brainstate-randomness-reproducibility/advanced-randomness.md` for advanced streams, mapped randomness, key restoration, and checkpoint behavior.
- Only `skills/brainstate/references/diagnostics/brainstate-transformed-diagnostics.md` may select `skills/brainstate/references/diagnostics/common-failures-index.md`; establish transformed diagnostics before opening the recurring-failure router.
- Do not select these three nested children directly from the skill router or another first-layer reference.

### Script references

- `skills/brainstate/references/lif_neuron_model.py`
  Direct skill script for combined State roles and explicit `.value` updates. Source: `https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html`.
- `skills/brainstate/references/modern_cnn.py`
  Script selected through the prebuilt layer or activation branch for convolution, normalization, pooling, dropout, and dense composition. Source: `https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html`.
- `skills/brainstate/references/resnet.py`
  Direct skill script for residual Modules and dynamic child registration. Its source is unresolved, so do not treat it as canonical until the source is established.

## Common Mistakes and Fixes

- Mutating raw Python attributes inside transformed code -> put mutable values in State.
- Reading or writing ordinary State without `.value` -> use `state.value`; for `nn.Param`, use `.value()` for the effective value and `.val` for its underlying `ParamState`.
- Changing a State PyTree structure accidentally -> preserve the structure or use `brainstate.check_state_value_tree()` while debugging.
- Using generic State for every role -> use semantic State subclasses when the role is known.
- Forgetting to register child Modules -> assign each child to a Module attribute.
- Forgetting to collect parameters -> use `model.states(brainstate.ParamState)`.
- Treating `nn.Const` as trainable -> keep it out of the `ParamState` collection; use `nn.Param` or `ParamState` for trainable values.
- Using raw `jax.jit` on stateful BrainState code -> use `brainstate.transform.jit`.
- Differentiating positional arguments like plain JAX -> pass a State collection to `brainstate.transform.grad`.
- Compiling tiny operations -> JIT the whole forward, simulation, or training step.
- Python-looping over batches -> write a single-example function and wrap it with `brainstate.transform.vmap`.
