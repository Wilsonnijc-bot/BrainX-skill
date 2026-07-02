# Solver Library With Effects Blueprint

## Purpose

Catalog BrainCell/BrainState integration and solver choices with their modeling consequences.

## Used by

- `skills/brainunit-quantity-safety/SKILL.md`
- `skills/brainx-general-guard/SKILL.md`
- `skills/braincell-singlecell/SKILL.md`
- `skills/braincell-multicompartment/SKILL.md`
- `references/braincell/braincell-custom-ion-channel-authoring.md`

## Primary source pages to expand from

- BrainCell integration concept page.
- BrainCell integration methods example.

## Should eventually cover

- Available integration methods.
- Solver tradeoffs.
- Stability.
- Stiffness.
- Accuracy.
- Runtime cost.
- Time-step units.
- Differentiable integration considerations.
- Effects on spike timing and gating traces.

## Common mistakes to document

- Choosing solver blindly.
- Treating solver differences as biological effects.
- Ignoring time units.
- Using too-large time steps.
- Comparing traces from different solvers without noting method differences.

## Placeholder examples

- Solver comparison table.
- HH integration-method comparison blueprint.
- Stiffness diagnostic checklist.
