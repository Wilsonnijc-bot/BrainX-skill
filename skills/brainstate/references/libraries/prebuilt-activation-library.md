# Prebuilt Activation Library Blueprint

## Purpose

Catalog BrainState activation and normalization components for module-building tasks.

## Used by

- `skills/brainstate/SKILL.md`
- `skills/brainstate/references/deeplearning-training/supervised-training-workflows.md`

## Primary source pages to expand from

- BrainState activation functions and normalization tutorial.

## Should eventually cover

- Common activation functions.
- Normalization layers.
- Feature standardization patterns.
- When activation functions are stateless.
- When normalization owns State.
- Composition with prebuilt layers.

## Common mistakes to document

- Treating stateful normalization like a pure function.
- Forgetting train/eval distinction where relevant.
- Reimplementing standard activations.
- Mixing incompatible feature axes.

## Placeholder examples

- Linear + activation.
- Conv + normalization + activation.
- Train/eval normalization note.
