# Ion Library Blueprint

## Purpose

Catalog available BrainCell ion species and ion-related modeling patterns.

## Used by

- `skills/braincell-singlecell/SKILL.md`
- `skills/braincell-multicompartment/SKILL.md`
- `references/braincell/braincell-custom-ion-channel-authoring.md`

## Primary source pages to expand from

- BrainCell ions and channels concept page.
- BrainCell ions tutorial.

## Should eventually cover

- Sodium ion patterns.
- Potassium ion patterns.
- Calcium ion patterns.
- Reversal potential handling.
- Concentration dynamics.
- Nernst-style templates where applicable.
- `MixIons`.
- How ions couple to channels.
- Unit expectations.

## Common mistakes to document

- Treating ion and channel as the same object.
- Hard-coding reversal potentials in the wrong layer.
- Forgetting concentration dynamics when needed.
- Using `MixIons` only after a channel requires multiple ion dependencies.

## Placeholder examples

- Existing ion selection table.
- Calcium dynamics mini blueprint.
- `MixIons` usage blueprint.
