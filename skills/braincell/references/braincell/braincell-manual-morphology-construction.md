# BrainCell Manual Morphology Construction

## Routing ownership

This is a nested morphology leaf owned only by `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`. Open the multicompartment parent first. Do not route here directly from `skills/braincell/SKILL.md`, the bundle router, or the global reference index.

## Purpose

Teach the agent how to hand-build, validate, and inspect neuronal branch geometry and topology before returning to the multicompartment `Cell` workflow.

## Open when

Open when the user explicitly needs manual branch/point construction, soma/dendrite/axon typing, parent-child topology, coordinates, radii, or validation of a hand-built morphology.

## Do not use for

Do not use for mechanism dynamics, solver choice, channel kinetics, or training unless morphology is the main issue.

Do not use for single-compartment models where geometry is intentionally ignored.

Do not use for SWC, ASC, NeuroML2, NeuroMorpho, or checkpoint loading. Return to the multicompartment parent and let it select the nested morphology-IO reference.

## What information this reference should eventually cover

- Morphology as branch geometry and topology.
- Branch points and typed cables.
- Manual morphology construction.
- Parent-child branch relationships.
- Point coordinates, radii, and branch types.
- Validation of a hand-built tree.
- Inspecting before simulation.

## Expected workflow

1. Define points, radii, branch types, and parent-child relationships.
2. Build the branch tree and verify its root and topology.
3. Validate structure, geometry, and branch types.
4. Inspect or visualize the constructed morphology.
5. Return to `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md` for CV policy, mechanisms, placement, and simulation.
6. Route to diagnostics only if the runtime topology still differs from the validated morphology.

## Required / useful reference markdowns

- `references/libraries/filter-function-library.md`
- `references/libraries/cv-policy-reference.md`
- `references/braincell/topology-building-and-visualization.md`
- `references/diagnostics/common-failures-index.md`

## Common mistakes this reference should prevent

- Creating disconnected or cyclic parent-child topology.
- Losing soma/dendrite/axon type information during construction.
- Supplying inconsistent point coordinates or radii.
- Not validating the hand-built tree before simulation.
- Confusing declaration-time morphology with runtime CV topology.
- Building mechanisms before confirming morphology targets.

## Placeholder for future examples or validation checks

- Manual branch construction blueprint.
- Typed soma/dendrite/axon tree blueprint.
- Manual topology validation blueprint.
- Hand-built morphology visualization blueprint.
