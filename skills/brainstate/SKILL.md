---
name: brainstate
description: Use for BrainState mutable State and `.value`, ParamState and other State roles, Module graphs, state collection and lifecycle operations, size-aware neural-network composition, reproducible randomness, state-aware jit/grad/vmap, or a BrainState training step.
---

# BrainState

## Purpose And Boundary

Use this skill for BrainState mutable `State`, `brainstate.nn.Module` graphs, state collection and initialization, reproducible randomness, state-aware transformations, and the general structure of stateful training. Activate it for `State`, `.value`, `ParamState`, `HiddenState`, Module traversal, `in_size` / `out_size`, `.desc()`, or `brainstate.transform`.

Use this canonical path:

`classify State roles -> construct Modules -> register State and children -> initialize -> transform the whole operation -> validate State and outputs`

Keep the canonical path here. Open references only for graph editing, collection utilities, lifecycle operations, extension hooks, constraints, full layer catalogs, transform variants, diagnostics, interoperation, or optimizer selection. Route performance, memory, accelerator, and multi-device work to `skills/brainx-acceleration-audit/SKILL.md`.

Advanced branches include training, dynamics, randomness, parameter constraints, model graphs, diagnostics, interoperation, layers, and acceleration. This skill keeps the general stateful training structure; route specialized neuronal or network dynamics to the matching BrainCell or BrainPy-State skill instead of the legacy BrainState dynamics drafts.

## P0 Concepts And Core Scripts

### 1. State Is The Mutation Boundary

`State` encapsulates model values that change over time. It can wrap Python scalars, arrays, `jax.Array` values, dictionaries, lists, or another stable PyTree structure; its value remains mutable after compilation. Read and write it through `.value`.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html

State provides three operational guarantees:

- Its value can be updated inside JIT-compiled functions.
- State checks value type, shape, and, when requested, PyTree structure.
- BrainState transformations can discover and manage its reads and writes.

Only values inside `State` are mutable in transformed code. Keep ordinary Python attributes for static configuration. Preserve the original PyTree structure when assigning `.value`.

#### Create scalar, array, and PyTree State

```python
import brainstate
import brainstate.nn as nn
import jax.numpy as jnp

counter = brainstate.State(jnp.array(0))
vector = brainstate.State(jnp.zeros(10))
neuron = brainstate.State({
    "V": jnp.zeros(5),
    "u": jnp.ones(5),
})

value = neuron.value
neuron.value = {
    "V": value["V"] + 0.1,
    "u": value["u"],
}

with brainstate.check_state_value_tree():
    neuron.value = {
        "V": jnp.zeros(5),
        "u": jnp.ones(5),
    }
```

#### Choose the State or parameter role

State subclasses are functionally State containers and act as semantic markers for filtering and model organization.

| Role | Use |
|---|---|
| `ParamState` | Bare trainable weights, biases, or other unconstrained values |
| `HiddenState` | Internal activations or dynamical state retained between updates |
| `ShortTermState` | Transient runtime values such as current input or last spike time |
| `LongTermState` | Persistent non-parameter values such as running statistics |
| `nn.Param` | A trainable value requiring a constraint transform or regularizer |
| `nn.Const` | A fixed value kept inside the Module graph |

`nn.Param` wraps an underlying trainable `ParamState`: computation reads the constrained value with `.value()`, while optimizers update `.val`. `nn.Const` is excluded when collecting `ParamState` objects.

Sources:

- https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html

```python
w = brainstate.ParamState(brainstate.random.randn(10, 5) * 0.1)
b = brainstate.ParamState(jnp.zeros(5))
h = brainstate.HiddenState(jnp.zeros(5))
last_spike = brainstate.ShortTermState(jnp.full(5, -1e7))
running_mean = brainstate.LongTermState(jnp.zeros(5))

positive_tau = nn.Param(jnp.array(2.0), t=nn.SoftplusT(lower=0.1))
fixed_scale = nn.Const(jnp.array(10.0))

print(positive_tau.value())  # constrained value used by the model
print(positive_tau.val)      # underlying trainable ParamState
print(fixed_scale.value())
```

### 2. Modules Form Registered State Graphs

`brainstate.nn.Module` is the base class for BrainState modules. It provides automatic child registration, State collection, inspection, and integration with BrainState transformations. Assign each `State` and child `Module` to an attribute so the model becomes a nested Module graph with State objects at the leaves.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/02_modules_and_graph.html

Modules keep related State and computation together, can be reused after construction, and compose into larger graphs. Collect only the semantic State role required by the operation:

```python
params = model.states(brainstate.ParamState)
```

#### Add State to a Module

```python
class Counter(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.count = brainstate.ShortTermState(jnp.array(0))

    def update(self, x):
        self.count.value = self.count.value + 1
        return x * self.count.value


counter_module = Counter()
for _ in range(5):
    print(counter_module(jnp.array(10.0)), counter_module.count.value)
```

The output advances through `(10, 1)`, `(20, 2)`, ..., `(50, 5)` because the registered State is updated explicitly.

#### Use basic prebuilt layers

```python
brainstate.random.seed(42)

linear = nn.Linear(in_size=(10,), out_size=(5,))
relu = nn.ReLU()
conv = nn.Conv2d(
    in_size=(28, 28, 3),
    out_channels=32,
    kernel_size=3,
    padding="SAME",
)
pool = nn.MaxPool2d(
    in_size=conv.out_size,
    kernel_size=(2, 2),
    stride=(2, 2),
    channel_axis=-1,
)

x = brainstate.random.randn(8, 10)
y = relu(linear(x))
assert y.shape == (8, 5)
```

Open the layer and activation catalogs instead of guessing an uncommon class name or signature.

### 3. Size Inference Drives Composition

Every size-aware `brainstate.nn.Module` exposes `in_size` and `out_size` as feature shapes without the batch dimension. When the input size is known, the Module computes its output size. `nn.Sequential` propagates one layer's `out_size` into the next layer, and `.desc()` creates a descriptor that is instantiated when that input size becomes available.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/02_modules_and_graph.html

#### Compose `ComplexNet` with `Sequential` and `.desc()`

```python
class ComplexNet(brainstate.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_size,
                out_channels=16,
                kernel_size=3,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.Conv2d.desc(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.Conv2d.desc(
                out_channels=64,
                kernel_size=3,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.MaxPool2d.desc(
                kernel_size=(2, 2),
                stride=(2, 2),
                channel_axis=-1,
            ),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(in_size=self.features.out_size),
            nn.Linear.desc(out_size=256),
            nn.ReLU(),
            nn.Linear.desc(out_size=10),
        )

    def update(self, x):
        return self.classifier(self.features(x))


brainstate.random.seed(42)
net = ComplexNet(in_size=(32, 32, 3))
x_image = brainstate.random.randn(2, 32, 32, 3)
y_image = net(x_image)

assert net.features.out_size == (8, 8, 64)
assert net.classifier.layers[0].out_size == (4096,)
assert net.classifier.out_size == (10,)
assert y_image.shape == (2, 10)
```

Open `references/size-inference-variations.md` for convolution formulas, padding/stride edge cases, pooling reduction, and flatten-size variants.

### 4. Use State-Aware Transforms And Reproducible Randomness

`brainstate.transform` mirrors JAX's `jit`, `grad`, and `vmap`, but tracks the `State` objects a model reads and writes. Raw `jax.jit` can discard State writes; wrap the complete stateful operation in `brainstate.transform` and prefer whole forward, simulation, or training steps over fragmented transforms.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

All `brainstate.random` functions use the global `brainstate.random.DEFAULT` `RandomState` unless a separate stream or key is supplied. Seed before random initialization or data generation when the run must be reproducible.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

#### Inspect the default `RandomState`

```python
print(brainstate.random.DEFAULT)
print(type(brainstate.random.DEFAULT))
```

#### Seed a reproducible sequence

```python
brainstate.random.seed(42)
x1 = brainstate.random.rand(5)

brainstate.random.seed(42)
x2 = brainstate.random.rand(5)

assert jnp.allclose(x1, x2)
```

Use independent `RandomState` instances, key save/restore, stochastic mapping, or checkpoint-aware randomness only through the randomness parent reference.

## Canonical Transformation Workflow

The scripts below share one model and dataset so the `jit`, `grad`, composed training-step, and `vmap` decisions are not re-explained four times.

```python
brainstate.random.seed(0)
model = nn.Linear(in_size=(3,), out_size=(1,))
x_train = brainstate.random.randn(64, 3)
y_train = brainstate.random.randn(64, 1)
params = model.states(brainstate.ParamState)
```

### Minimal State-Aware JIT

```python
forward = brainstate.transform.jit(model)
prediction = forward(x_train)
assert prediction.shape == (64, 1)
```

The first compatible call traces and compiles the complete forward pass; later compatible calls reuse it while BrainState handles State effects.

### Minimal Gradient And Parameter Update

`grad` differentiates with respect to a State collection and returns gradients keyed by the same State paths. `return_value=True` returns the loss from the same pass.

```python
def loss_fn():
    return jnp.mean((model(x_train) - y_train) ** 2)


grads, loss = brainstate.transform.grad(
    loss_fn,
    params,
    return_value=True,
)()

for key in params:
    params[key].value -= 0.1 * grads[key]
```

### Composed Training-Step Transform

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


loss = train_step()
```

Use `jit(grad(...))` as the default compiled training-step backbone. Open the optimizer reference when manual updates should become optimizer-managed updates.

### Minimal `vmap`

```python
def predict_one(x_row):
    return model(x_row[None, :])[0]


predict_batch = brainstate.transform.vmap(predict_one)
batched_prediction = predict_batch(x_train)
assert batched_prediction.shape == (64, 1)
```

This maps a function written for one example over a batch. Open the `vmap` expansion for mapped State axes, ensembles, parameter sweeps, or the documented `state_in_axes` / `state_out_axes` controls. The routed tutorial does not define the rough draft's `in_states` / `out_states` names.

## Reference Routing

Open only the smallest reference that owns the requested variant.

| Reference | Open when |
|---|---|
| `references/state-graph-operations.md` | Find, extract, split, replace, or reconstruct State graphs |
| `references/model-interop-and-migration.md` | Interoperate with Flax/Equinox or migrate PyTorch concepts |
| `references/state_collections_and_utilities.md` | Filter, organize, freeze, flatten, configure, or print nested collections |
| `references/collective_model_operations.md` | Initialize, reset, invoke methods, batch lifecycle calls, or restore State model-wide |
| `references/extension_mechanisms.md` | Use mixins, descriptors, runtime modes, or State hooks |
| `references/size-inference-variations.md` | Handle convolution, pooling, padding, or flatten size variants |
| `references/braintools-optimizer-reference.md` | Select optimizers, schedules, Optax bridges, or external wrappers |
| `references/brainstate/parameter-constraints-regularization.md` | Add constraints, transforms, `nn.Const`, regularizers, or loss penalties |
| `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` | Use independent streams, stochastic transforms, trials, noise/dropout, or RNG checkpoints |
| `references/libraries/prebuilt-layer-library.md` | Select an exact prebuilt layer and signature |
| `references/libraries/prebuilt-activation-library.md` | Select activation or normalization behavior |
| `references/brainstate/transformation-jit-expansion.md` | Control State write-back, cache/static arguments, boundaries, or benchmarks |
| `references/brainstate/transformation-grad-expansion.md` | Use advanced autodiff, fitting, differentiable simulation, `return_value`, or `has_aux` |
| `references/brainstate/transformation-vmap-expansion.md` | Map State axes, ensembles, sweeps, or stochastic work |
| `references/brainstate/brainstate-control-flow-patterns.md` | Use transform-safe loops, scans, branches, or checkpointed control flow |
| `references/diagnostics/brainstate-transformed-diagnostics.md` | Debug traced values, runtime checks, NaN/Inf, or callbacks |

Nested references have one inbound route each:

- Only `parameter-constraints-regularization.md` may open `parameter-containers-transforms-catalog.md`.
- Only `randomness-and-reproducibility.md` may open `advanced-randomness.md`.
- Only `brainstate-transformed-diagnostics.md` may open `common-failures-index.md`.

Do not route to the existing dynamics, deep-learning training, solver, or legacy split size-inference files from this skill; they are outside the architecture supplied for this BrainState skill.

## Script References

- `references/lif_neuron_model.py`: extended combination of `HiddenState`, `ShortTermState`, and `ParamState` with explicit `.value` updates. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html
- `references/modern_cnn.py`: full convolution, normalization, activation, pooling, dropout, and dense Module composition. Select it through the layer or activation branch. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/04_activations_and_normalization.html
- `references/resnet.py`: residual Modules and dynamically registered child blocks. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/02_modules_and_graph.html

## Common Failures

- Mutable raw attribute inside transformed code -> store the value in an appropriate `State`.
- Ordinary State read or write omits `.value` -> use `.value`; reserve `.value()` and `.val` for `nn.Param`.
- State assignment changes the PyTree structure -> preserve it or debug under `check_state_value_tree()`.
- Every value uses generic `State` -> choose semantic subclasses so filtering and training target the correct leaves.
- Child Module is kept only in an unregistered local container -> assign each child to a Module attribute.
- Gradient targets all State roles -> collect `model.states(brainstate.ParamState)` only.
- `nn.Const` is expected to train -> replace it with `ParamState` or `nn.Param`.
- Stateful function uses raw `jax.jit`, `jax.grad`, or `jax.vmap` -> use the matching `brainstate.transform` operation.
- JIT wraps tiny fragments -> compile the whole forward, simulation, or training step.
- Python loops over a batch -> write the single-example operation and use state-aware `vmap`.
