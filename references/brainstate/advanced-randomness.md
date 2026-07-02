# Advanced Randomness Blueprint

## Purpose

Catalog advanced BrainState random number generation, RNG stream, stochastic layer, and transformed-randomness patterns after the core randomness skill has established the boundary.

## Used by

- `skills/brainstate-randomness-reproducibility/SKILL.md`
- `skills/brainstate-module-building/SKILL.md`
- `skills/brainstate-deeplearning-training/SKILL.md`
- `skills/brainstate-brain-dynamics/SKILL.md`
- `skills/brainstate-transformations-core/SKILL.md`
- `skills/brainstate-control-flow/SKILL.md`
- `skills/brainx-debugging-diagnostics/SKILL.md`
- `references/brainstate/transformation-vmap-expansion.md`

## Primary source pages to expand from

- https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Should eventually cover

- Direct `get_key` / `set_key` / `split_key`.
- `RandomState` and `DEFAULT`.
- Custom independent RNG streams.
- Distribution catalog.
- Random mini-batch creation.
- Dropout.
- Noisy layers.
- Random connectivity.
- Random spike trains.
- Checkpointing RNG state.
- Randomness with `jit` / `vmap` / `grad` / control flow.

## Common mistakes to document

- Not setting a seed for reproducible examples.
- Reusing the same random sample accidentally.
- Assuming stochastic code behaves the same under `vmap`.
- Debugging stochastic failures without fixing RNG first.
- Forgetting independent RNG streams for independent stochastic processes.
- Forgetting RNG save/restore in checkpoints.

## Placeholder examples

- Global seed setup.
- Basic sampling functions.
- Independent RandomState streams.
- Random init in Module.
- Dropout training/eval reproducibility.
- Random mini-batch creation.
- Randomness inside `scan` / `for_loop`.
- Shared versus independent randomness under `vmap`.
- Checkpoint save/restore of RNG state.
