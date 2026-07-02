# Transformation VMAP Expansion Blueprint

## Purpose

Detail how BrainState vectorization should be taught after the Transformations Core skill routes here.

## Used by

- `skills/brainstate-transformations-core/SKILL.md`
- `skills/brainstate-deeplearning-training/SKILL.md`
- `skills/braincell-singlecell/SKILL.md`
- `skills/braincell-multicompartment/SKILL.md`

## Primary source pages to expand from

- BrainState vectorization tutorial.
- BrainState randomness tutorial for stochastic vectorization.

## Should eventually cover

- `vmap` over function arguments.
- `state_in_axes` and `state_out_axes`.
- Mapping over stateful Modules.
- `vmap` versus `vmap2` if both are relevant.
- Vectorized parameter sweeps.
- Vectorized populations or ensembles.
- Composition with JIT and grad.
- Randomness splitting/sharing behavior under vectorization.
- When BrainCell built-in vectorized construction is enough.

## Randomness route

Open `brainstate-randomness-reproducibility` when vectorized stochastic functions need independent or shared random streams.

## Keep out

- Full training-loop tutorial.
- Full control-flow tutorial.
- Detailed random-number tutorial except routing notes. Use `references/brainstate/advanced-randomness.md` after the core randomness skill.

## Common mistakes to document

- Vectorizing inputs but forgetting state axes.
- Accidentally sharing state across mapped instances.
- Accidentally sharing RNG samples.
- Using vmap where a built-in BrainCell batch/population pattern is simpler.

## Placeholder examples

- Batched input vmap.
- Stateful vmap.
- Parameter sweep.
- Vectorized stochastic function.
- FI-curve style sweep blueprint.
