---
name: brainstate-transformations-core
description: Guides BrainState-aware program transformations for stateful code, including jit, grad, vmap, transformed state handling, and routing to transformation expansion references. Use when the user asks about compilation, gradients, batching, sweeps, vectorization, differentiable simulation, or replacing raw JAX transforms; route transformed loops and branches to BrainState control flow.
---

# brainstate-transformations-core/

## Concepts

• what this skill is for
Use when the task needs BrainState-aware jit, grad, vmap, or composed transforms over modules/states. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

• BrainState transformation difference from raw JAX
brainstate.transform mirrors the JAX transformation API — jit, grad, vmap — but every transform is state-aware: it tracks the State objects your model reads and writes. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

• state-aware transformation
Wrap a model in a brainstate transform and its state is handled for you. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

• raw JAX boundary
A BrainState model keeps mutable State; raw jax.jit can silently discard the State write. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

• jit
jit traces a function the first time it is called, compiles it with XLA, and reuses the compiled version afterwards. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

• grad
grad differentiates a function with respect to a collection of States — not its positional arguments, as in plain JAX. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

• vmap
vmap adds a batch dimension to a function written for a single example, turning Python-level looping into a single vectorized call. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

• composing transformations
Transforms compose; the common pattern is jit(grad(...)): differentiate, then compile the whole gradient computation. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

### Mini title

Minimal state-aware JIT canonical path

#### Script

```python
class Linear(brainstate.nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.randn(din, dout) * 0.1)
        self.b = brainstate.ParamState(jnp.zeros(dout))
    def __call__(self, x):
        return x @ self.w.value + self.b.value
model = Linear(3, 1)
x = brainstate.random.randn(64, 3)
y = brainstate.random.randn(64, 1)
forward = brainstate.transform.jit(model)
forward(x).shape
```

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

#### Explanation text

Use jit on whole steps — a forward pass, a training step — not on tiny operations. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

### Mini title

Minimal grad + parameter update canonical path

#### Script

```python
params = model.states(brainstate.ParamState)
def loss_fn():
    return jnp.mean((model(x) - y) ** 2)
grads = brainstate.transform.grad(loss_fn, params)()
grads, loss = brainstate.transform.grad(loss_fn, params, return_value=True)()
for key in params:
    params[key].value -= 0.1 * grads[key]
```

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

#### Explanation text

The gradient keys match the parameter keys exactly, so applying an update is a simple loop. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

### Mini title

Minimal vmap canonical path

#### Script

```python
def predict_one(x_row):
    return jnp.tanh(model(x_row[None, :]))[0]
predict_batch = brainstate.transform.vmap(predict_one)
predict_batch(x).shape
```

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

#### Explanation text

Because it is state-aware, vmap can also map over the states themselves through in_states / out_states. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

### Mini title

Composed training-step transform

#### Script

```python
@brainstate.transform.jit
def train_step():
    grads, loss = brainstate.transform.grad(loss_fn, params, return_value=True)()
    for key in params:
        params[key].value -= 0.1 * grads[key]
    return loss
```

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html

## Reference

references/brainstate/transformation-jit-expansion.md

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html, https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html
**Purpose:** Expands BrainState-aware JIT compilation, state write-back, cache, and static-argument guidance.

references/brainstate/transformation-grad-expansion.md

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html, https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html, https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
**Purpose:** Expands gradient and autodiff teaching for differentiable simulation and parameter fitting.

references/brainstate/transformation-vmap-expansion.md

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html, https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html, https://brainx.chaobrain.com/brainstate/tutorials/transformations/04_advanced_batching.html
**Purpose:** Expands BrainState vectorization, batching, state axes, sweeps, and stochastic vmap patterns.

references/brainstate/brainstate-control-flow-patterns.md

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html
**Purpose:** Collects loop and branch patterns that remain valid under BrainState and JAX transformations.

references/diagnostics/brainstate-transformed-diagnostics.md

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html, https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html
**Purpose:** Collects runtime debugging, checking, and error-handling patterns for transformed BrainState code.

## Full bundled script references

transformations-essentials.py

**Source:** https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html
**Purpose:** complete representative workflow for state-aware jit, grad, vmap, and jit(grad(...)).

## Common mistakes -> Fix

• using raw jax.jit on a mutable BrainState model -> use brainstate.transform.jit so state reads/writes persist.
• differentiating positional args like plain JAX -> collect params = model.states(brainstate.ParamState) and pass states to grad.
• compiling tiny operations -> apply jit to whole forward/training steps.
• Python-looping over a batch -> write a single-example function and wrap with brainstate.transform.vmap.
• updating params outside the gradient key structure -> update by matching params keys to grads keys.

⸻
