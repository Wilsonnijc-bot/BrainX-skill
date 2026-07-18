# BrainState Randomness and Reproducibility

This is BrainState's only first-layer randomness reference. Open it after `skills/brainstate/SKILL.md` when a task needs independent streams, repeated stochastic trials, random calls inside a transformed operation, dropout or noise, or RNG state stored with a checkpoint. Do not route directly to `advanced-randomness.md` from the skill or another reference.

The skill owns the global `DEFAULT` generator and ordinary seeding path; do not repeat those basics here. The supplemental rule is that `brainstate.random` wraps JAX random generation in a stateful interface with automatic key splitting, while its random functions remain JIT-compatible.

Source: https://brainx.chaobrain.com/brainstate/apis/random.html

## Independent streams and repeated trials

"For advanced use cases, you can create custom `RandomState` instances with independent random streams." Use one `RandomState` per concern whose sequence must advance independently, such as data augmentation and model initialization. Consecutive calls on one instance are successive stochastic trials from that stream.

```python
import brainstate
import jax.numpy as jnp

rng1 = brainstate.random.RandomState(42)
rng2 = brainstate.random.RandomState(123)

samples1 = rng1.randn(5)
samples2 = rng2.randn(5)

assert not jnp.allclose(samples1, samples2)
```

The API also supports the explicit keyword form `brainstate.random.RandomState(seed=123)` and instance methods such as `rng.normal(0, 1, size=(10, 10))`. Keep ordinary independent streams at this level; open the nested child when the task requires direct key manipulation.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Stochastic calls, transforms, dropout, and noise

BrainState random functions are JIT-compatible. Apply the whole-operation `brainstate.transform` rule from `skills/brainstate/SKILL.md` to stochastic code just as you do to other stateful code. A normal compiled stochastic call stays on this parent path; mapped randomness under `vmap`, parallel key assignment, or explicit key control belongs in `advanced-randomness.md`.

The tutorial's dropout pattern bundles fit-mode gating, mask generation, and inverse-keep scaling in the stochastic call:

```python
class Dropout(brainstate.nn.Module):
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate

    def __call__(self, x):
        fit = brainstate.environ.get('fit', False)
        if not fit:
            return x

        keep_prob = 1.0 - self.drop_rate
        mask = brainstate.random.bernoulli(keep_prob, x.shape)
        return x * mask / keep_prob
```

When `fit` is false, the module returns `x` unchanged. When fitting, `bernoulli` draws the mask with mean `keep_prob`, and division by `keep_prob` applies the tutorial's scaling.

The tutorial's noisy-layer pattern keeps trainable values in `ParamState` and draws fresh Gaussian weight noise inside every call. Two consecutive calls are two stochastic trials because automatic key management advances the random state.

```python
class NoisyLayer(brainstate.nn.Module):
    def __init__(self, d_in, d_out, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.w = brainstate.ParamState(
            brainstate.random.randn(d_in, d_out) * 0.1
        )
        self.b = brainstate.ParamState(jnp.zeros(d_out))

    def __call__(self, x):
        w_noisy = self.w.value + brainstate.random.normal(
            0,
            self.noise_std,
            self.w.value.shape,
        )
        return x @ w_noisy + self.b.value


layer = NoisyLayer(5, 3, noise_std=0.01)
x = jnp.ones(5)
y1 = layer(x)
y2 = layer(x)
```

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Checkpoint the RNG with the model

When exact stochastic continuation matters, the tutorial saves the model and current RNG key in the same checkpoint, then restores the key before continuing:

```python
checkpoint = {
    'model': model.state_dict(),
    'rng_key': brainstate.random.get_key()
}

brainstate.random.set_key(checkpoint['rng_key'])
```

Keep this compact checkpoint requirement here because it determines what must be saved. Open `advanced-randomness.md` before adapting key restoration, restoring multiple streams, or defining more elaborate checkpoint behavior.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Open `advanced-randomness.md` only from this parent

Use the nested child for:

- manual key inspection, splitting, assignment, or restoration;
- mapped or parallel randomness, including `vmap` key behavior;
- restoring custom streams or exact intermediate key state;
- advanced checkpoint behavior beyond the tutorial's model-plus-RNG pair.

## Official sources

- https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html
- https://brainx.chaobrain.com/brainstate/apis/random.html
