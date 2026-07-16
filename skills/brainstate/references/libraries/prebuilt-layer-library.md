# Prebuilt Layer Library Blueprint

## Purpose

Catalog BrainState prebuilt layers so the agent can reuse existing components instead of writing custom layers unnecessarily.

## Used by

- `skills/brainstate/SKILL.md`
- `skills/brainstate/references/deeplearning-training/supervised-training-workflows.md`

## Primary source pages to expand from

- BrainState common layers tutorial.

## Should eventually cover

- Linear.
- LoRA or low-rank variants if available.
- Sparse linear variants if available.
- Convolution layers: 1D, 2D, 3D.
- Pooling layers: average/adaptive variants.
- Dropout.
- Flatten.
- Containers/sequential composition.
- Basic size-inference notes.

## Common mistakes to document

- Rewriting standard layers manually.
- Forgetting state/parameter registration.
- Misunderstanding tensor shape conventions.
- Using dropout without handling randomness and train/eval mode.

## Placeholder examples

- Minimal Linear.
- Conv + activation.
- Pool + flatten.
- Dropout route to randomness.
