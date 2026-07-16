# CV Policy Reference Blueprint

## Purpose

Catalog BrainCell control-volume policies and how discretization choices affect multicompartment simulation.

## Routing ownership

This is a nested multicompartment leaf opened only through `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md` or its nested manual-construction reference.

## Used by

- `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`
- `references/braincell/braincell-manual-morphology-construction.md`

## Primary source pages to expand from

- BrainCell discretization concept page.
- BrainCell cell tutorial.

## Should eventually cover

- What a control volume is.
- Available CV policies.
- How CV policy affects resolution.
- How CV policy affects runtime cost.
- CV count inspection.
- Branch/cable discretization tradeoffs.
- Interaction with probes and locsets.
- When to refine or simplify discretization.

## Common mistakes to document

- Choosing CV policy blindly.
- Equating morphology points with CVs.
- Using too few CVs for spatial gradients.
- Using too many CVs without computational reason.
- Debugging probe location without checking CV mapping.

## Placeholder examples

- CV policy comparison table.
- CV count inspection.
- Soma/dendrite discretization blueprint.
