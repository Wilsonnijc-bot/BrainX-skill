---
name: brainstate-randomness-reproducibility
description: Use when the user asks about BrainState random sources, stochasticity, seeds, reproducibility, random initialization, random sampling, dropout, noisy models, random batches, random connectivity, random spike trains, or independent RNG streams.
---

brainstate-randomness-reproducibility/

Concepts

• what this skill is for
Use when the task needs BrainState random generation, reproducibility, seed management, automatic key handling, stochastic modules, or RNG checkpointing. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

• random source
BrainState provides a random number generation system built on JAX with NumPy-like API, reproducibility, JIT-compatible random functions, and stateful management. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

• automatic key management
JAX uses explicit random keys; BrainState handles keys automatically after brainstate.random.seed(...). Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

• default random state
All random functions in brainstate.random use a global DEFAULT instance of RandomState. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

• seed management
For reproducible experiments, always set a seed at the beginning. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

• key management
For advanced use cases, directly access and manipulate keys with get_key, split_key, and set_key. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

• stochasticity in training
Mini-batch sampling uses brainstate.random.permutation; dropout uses brainstate.random.bernoulli; noisy layers can use brainstate.random.normal. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

• independent streams
Custom RandomState instances create independent random streams. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

• checkpoint reproducibility
Save and restore the RNG key with brainstate.random.get_key() and brainstate.random.set_key(...). Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Mini title

Minimal automatic key management

Script

brainstate.random.seed(0)
x = brainstate.random.normal(0, 1, (100,))

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Explanation text

BrainState approach: automatic key management; keys are handled automatically. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Mini title

Minimal reproducible generation

Script

brainstate.random.seed(42)
x1 = brainstate.random.rand(5)
brainstate.random.seed(42)
x2 = brainstate.random.rand(5)
jnp.allclose(x1, x2)

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Explanation text

Resetting to the same seed reproduces the same sequence. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Mini title

Minimal key save / split / restore

Script

current_key = brainstate.random.get_key()
keys = brainstate.random.split_key(n=4)
saved_key = brainstate.random.get_key()
v1 = brainstate.random.randn(3)
v2 = brainstate.random.randn(3)
brainstate.random.set_key(saved_key)
v3 = brainstate.random.randn(3)
jnp.allclose(v1, v3)

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Explanation text

Use key save/restore when reproducible continuation matters, such as checkpointing. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Mini title

Minimal sampling functions

Script

brainstate.random.seed(42)
uniform_samples = brainstate.random.rand(5)
normal_samples = brainstate.random.randn(5)
int_samples = brainstate.random.randint(0, 10, size=5)

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Explanation text

Random sampling functions include rand, randn, randint, random, and choice. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Mini title

Independent streams

Script

rng1 = brainstate.random.RandomState(42)
rng2 = brainstate.random.RandomState(123)
samples1 = rng1.randn(5)
samples2 = rng2.randn(5)
rng1.seed(999)
samples3 = rng1.randn(5)

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Explanation text

Use custom RNGs for independent streams such as data augmentation and model initialization. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Mini title

Where randomness matters in training

Script

def create_mini_batches(X, y, batch_size=32, shuffle=True):
    n_samples = len(X)
    if shuffle:
        indices = brainstate.random.permutation(n_samples)
    else:
        indices = jnp.arange(n_samples)
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))
    return batches

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Explanation text

Mini-batch sampling uses brainstate.random.permutation to shuffle indices. Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html  

Reference

references/brainstate/advanced-randomness.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html
Purpose: Catalogs advanced BrainState RNG streams, stochastic layers, and transformed-randomness patterns.

references/brainstate/transformation-vmap-expansion.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html, https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html, https://brainx.chaobrain.com/brainstate/tutorials/transformations/04_advanced_batching.html
Purpose: Expands BrainState vectorization, batching, state axes, sweeps, and stochastic vmap patterns.

references/brainstate/transformation-jit-expansion.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html, https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html
Purpose: Expands BrainState-aware JIT compilation, state write-back, cache, and static-argument guidance.

references/brainstate/brainstate-control-flow-patterns.md

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html
Purpose: Collects loop and branch patterns that remain valid under BrainState and JAX transformations.

Full bundled script references

random-number-generation.py

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html
Purpose: complete executable workflow for seeding, sampling, key management, custom RandomState, stochastic modules, and checkpoint RNG state.

Common mistakes -> Fix

• not setting a seed -> call brainstate.random.seed(...) at the start of the script/notebook.
• manually threading JAX PRNG keys through BrainState code -> use brainstate.random automatic key management unless advanced key control is required.
• expecting identical stochastic results after consuming random numbers -> save key with get_key() before the stochastic block and restore with set_key(...).
• using one global stream for unrelated randomness -> create custom RandomState instances for independent streams.
• checkpointing model state but not RNG state -> include rng_key = brainstate.random.get_key() and restore it.
• dropout/noisy layers active in eval by accident -> gate stochastic behavior with fit/eval environment where the source module pattern requires it.
