---
name: brainstate-deeplearning-training
description: Guides BrainState training workflows, including losses, optimizers, metrics, evaluation, gradient updates, transformed training steps, ParamState updates, randomness, seeding, and stochastic layers. Use when training BrainState or BrainCell models, fitting parameters, computing losses, managing dropout or random batches, or combining training loops with grad and jit.
---

brainstate-deeplearning-training/

Concepts

• what this skill is for
Use when the task is a BrainState supervised-training loop: model, loss, optimizer, registered parameters, compiled train step, metrics, and evaluation. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

• training loop structure
This tutorial assembles Module, ParamState, grad, and jit into a complete training loop. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

• model / loss / optimizer / metrics
A BrainState training loop is built from count_parameters, braintools.optim, grad + clip_grad_norm + jit, and MultiMetric. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

• parameter collection
Collect exactly the ParamState instances; other state types are left untouched. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

• gradient update
A single training step differentiates the loss, clips gradients, then applies optimizer.update(grads) in place. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

• compiled training step
Wrapping the whole step in brainstate.transform.jit compiles it once and tracks state reads/writes automatically across the transform boundary. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

• metrics lifecycle
MultiMetric bundles several metrics; lifecycle is always reset → update per batch → compute. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

• fit/eval environment note
Stochastic modules may read fit = brainstate.environ.get('fit', False) before applying randomness. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Mini title

Minimal self-contained dataset

Script

DIM, N_CLASSES = 8, 3
centers = brainstate.random.randn(N_CLASSES, DIM) * 2.0
def make_blobs(n_per):
    xs = [brainstate.random.randn(n_per, DIM) + centers[c] for c in range(N_CLASSES)]
    ys = [jnp.full((n_per,), c, dtype=jnp.int32) for c in range(N_CLASSES)]
    return jnp.concatenate(xs), jnp.concatenate(ys)
x_train, y_train = make_blobs(200)
x_test, y_test = make_blobs(50)
mean, std = x_train.mean(axis=0), x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Explanation text

The statistics come from the training set only; the test set is transformed with those same numbers, never its own. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Mini title

Minimal model

Script

class MLP(brainstate.nn.Module):
    def __init__(self, din, dhidden, dout):
        super().__init__()
        self.fc1 = brainstate.nn.Linear(din, dhidden)
        self.fc2 = brainstate.nn.Linear(dhidden, dout)
    def __call__(self, x):
        return self.fc2(brainstate.nn.relu(self.fc1(x)))
model = MLP(DIM, 32, N_CLASSES)

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Explanation text

A two-layer perceptron composes two brainstate.nn.Linear layers with a ReLU nonlinearity; the final layer emits one logit per class. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Mini title

Parameter count + optimizer registration

Script

n_params = count_parameters(model)
optimizer = braintools.optim.Adam(lr=1e-2)
optimizer.register_trainable_weights(model.states(brainstate.ParamState))

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Explanation text

After constructing an optimizer, register the states it is allowed to update. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Mini title

Canonical train step

Script

params = model.states(brainstate.ParamState)
@brainstate.transform.jit
def train_step(x, y):
    def loss_fn():
        logits = model(x)
        return softmax_cross_entropy_with_integer_labels(logits, y).mean()
    grads, loss = brainstate.transform.grad(loss_fn, params, return_value=True)()
    grads = clip_grad_norm(grads, max_norm=1.0)
    optimizer.update(grads)
    return loss

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Explanation text

grad differentiates the loss, clip_grad_norm rescales gradients, and optimizer.update(grads) applies the update in place. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Mini title

Metrics + evaluation step

Script

metrics = MultiMetric(
    loss=AverageMetric('loss'),
    accuracy=AccuracyMetric(),
)
@brainstate.transform.jit
def eval_step(x, y):
    logits = model(x)
    loss = softmax_cross_entropy_with_integer_labels(logits, y).mean()
    metrics.update(loss=loss, logits=logits, labels=y)

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Explanation text

Each sub-metric reads the keyword arguments it needs from a single update(...) call. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Mini title

Training loop skeleton

Script

def iter_batches(x, y, batch_size):
    order = brainstate.random.permutation(len(x))
    for i in range(0, len(x), batch_size):
        idx = order[i:i + batch_size]
        yield x[idx], y[idx]
for epoch in range(15):
    for xb, yb in iter_batches(x_train, y_train, batch_size=32):
        train_step(xb, yb)
    metrics.reset()
    eval_step(x_test, y_test)
    stats = metrics.compute()

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Explanation text

Iterate in mini-batches, reshuffle each epoch with brainstate.random.permutation, reset metrics, run evaluation, then compute statistics. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html  

Reference

references/brainstate/transformation-grad-expansion.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html, https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html, https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
Purpose: Expands gradient and autodiff teaching for differentiable simulation and parameter fitting.

references/brainstate/transformation-jit-expansion.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html, https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html
Purpose: Expands BrainState-aware JIT compilation, state write-back, cache, and static-argument guidance.

references/brainstate/parameter-constraints-regularization.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html, https://brainx.chaobrain.com/brainstate/how_to/constrain_and_regularize_parameters.html
Purpose: Collects constrained parameter, parameter transform, and regularization notes for trainable BrainState values.

references/brainstate/advanced-randomness.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html
Purpose: Catalogs advanced BrainState RNG streams, stochastic layers, and transformed-randomness patterns.

references/brainstate/brainstate-control-flow-patterns.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html
Purpose: Collects loop and branch patterns that remain valid under BrainState and JAX transformations.

references/libraries/prebuilt-layer-library.md

Source: https://brainx.chaobrain.com/brainstate/apis/nn/linear.html, https://brainx.chaobrain.com/brainstate/apis/nn/conv.html, https://brainx.chaobrain.com/brainstate/apis/nn/normalization.html, https://brainx.chaobrain.com/brainstate/apis/nn/pooling.html, https://brainx.chaobrain.com/brainstate/apis/nn/padding.html, https://brainx.chaobrain.com/brainstate/apis/nn/dropout.html
Purpose: Catalogs BrainState prebuilt layers so agents reuse existing components before writing custom modules.

references/libraries/prebuilt-activation-library.md

Source: https://brainx.chaobrain.com/brainstate/apis/nn/activation.html
Purpose: Catalogs BrainState activation and normalization components for module-building tasks.

references/brain-dynamics/brain-dynamics-snn-workflows.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html, https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html
Purpose: Routes build, simulate, and train workflows for BrainState-style spiking neural networks.

Full bundled script references

training-and-metrics-classification.py

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html
Purpose: complete executable synthetic classification training loop with optimizer and metrics.

training-snn.py

Source: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html
Purpose: complete executable representative workflow for training a spiking neural network with surrogate gradients.

snn-training-example.py

Source: https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/snn_training.html
Purpose: longer example-gallery script for SNN training variants.

Common mistakes -> Fix

• training plain arrays or all states -> collect/register only model.states(brainstate.ParamState).
• calling optimizer before registration -> construct optimizer, then register_trainable_weights(...).
• forgetting to JIT the whole step -> wrap the full loss/grad/update step in brainstate.transform.jit.
• leaking test statistics into preprocessing -> compute mean/std on training data only.
• metric values accumulating across evaluations -> metrics.reset() before evaluation.
• stochastic modules behave same in train/eval -> check brainstate.environ fit/eval behavior when dropout/noise is involved.

⸻
