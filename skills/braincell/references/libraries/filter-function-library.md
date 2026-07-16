# Filter Function Library Blueprint

## Purpose

Catalog BrainCell region and locset filters for spatially targeting mechanisms, probes, and clamps.

## Routing ownership

For BrainCell morphology tasks, this is a nested leaf opened only through `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md` or its nested manual-construction reference.

## Used by

- `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`
- `references/braincell/braincell-manual-morphology-construction.md`

## Primary source pages to expand from

- BrainCell regions and locsets concept page.
- BrainCell region and locset filters tutorial.

## Should eventually cover

- Region selectors.
- Locset selectors.
- Branch/type filters.
- Soma/dendrite/axon targeting.
- Proximal/distal targeting if supported.
- Combining filters.
- Visual or runtime verification of targets.
- Paint versus place routing.

## Common mistakes to document

- Using a locset where a region is required.
- Using a region where a locset is required.
- Targeting the wrong branch type after import.
- Not verifying selected targets before simulation.

## Placeholder examples

- Soma region filter.
- Dendrite region filter.
- Terminal locset filter.
- Combined selector blueprint.
