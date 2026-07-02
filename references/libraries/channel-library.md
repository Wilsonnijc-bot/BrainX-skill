# Channel Library Blueprint

## Purpose

Catalog available BrainCell channels and channel-modeling patterns.

## Used by

- `skills/braincell-singlecell/SKILL.md`
- `skills/braincell-multicompartment/SKILL.md`
- `skills/braincell-ion-channel-authoring/SKILL.md`

## Primary source pages to expand from

- BrainCell ions and channels concept page.
- BrainCell channel tutorial.
- BrainCell example pages for gating, ablation, adaptation, rebound, and thalamic neurons.

## Should eventually cover

- Sodium channels.
- Potassium channels.
- Calcium channels.
- Leak/passive channels where applicable.
- AHP/adaptation channels.
- T-type/rebound-related channels.
- HH-style channels.
- Markov channels.
- Existing versus custom channel decision tree.
- Required ion dependencies.

## Common mistakes to document

- Writing a custom channel before checking the library.
- Ignoring ion dependency and reversal potential.
- Mixing units incorrectly in conductance/current.
- Not validating steady-state gating or current direction.

## Placeholder examples

- Existing channel selection table.
- Channel ablation blueprint.
- Calcium gating blueprint.
- T-current rebound blueprint.
