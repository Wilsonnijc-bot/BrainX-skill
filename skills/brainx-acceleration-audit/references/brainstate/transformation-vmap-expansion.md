# Transformation VMAP Expansion Blueprint

## Purpose

Detail how BrainState vectorization should be taught after the Transformations Core skill routes here.

## Used by

- `skills/brainstate/SKILL.md`
- `skills/braincell/SKILL.md`

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

Open `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` when vectorized stochastic functions need independent or shared random streams.

## Keep out

- Full training-loop tutorial.
- Full control-flow tutorial.
- Detailed random-number tutorial except routing notes. Route to `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`; only that parent may select its advanced child.

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
