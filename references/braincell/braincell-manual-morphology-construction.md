# BrainCell Manual Morphology Construction

## Purpose

Teach the agent how to build, load, validate, and inspect neuronal morphology before BrainCell multicompartment modeling.

## Open when

Open when the user asks about morphology files, SWC, ASC, NeuroML2, NeuroMorpho, branches, points, topology, soma/dendrite/axon structure, geometry, morphology validation, or morphology visualization.

## Do not use for

Do not use for mechanism dynamics, solver choice, channel kinetics, or training unless morphology is the main issue.

Do not use for single-compartment models where geometry is intentionally ignored.

## What information this reference should eventually cover

- Morphology as branch geometry and topology.
- Branch points and typed cables.
- Manual morphology construction.
- File loading paths.
- SWC specifics.
- Neurolucida ASC specifics.
- NeuroML2 morphology import.
- NeuroMorpho search/download/cache path.
- Validation reports.
- Checkpointing processed morphology.
- Inspecting before simulation.

## Expected workflow

1. Determine whether morphology is built manually or loaded.
2. Load/build branches and topology.
3. Validate structure and branch types.
4. Inspect geometry.
5. Save/checkpoint processed morphology when useful.
6. Route to Multicompartment Concept/API for mechanisms and simulation.
7. Route to Debugging Diagnostics if runtime topology does not match expectations.

## Required / useful reference markdowns

- `references/braincell/morphology-io-loading-validation.md`
- `references/libraries/filter-function-library.md`
- `references/libraries/cv-policy-reference.md`
- `references/braincell/topology-building-and-visualization.md`
- `references/diagnostics/common-failures-index.md`

## Common mistakes this reference should prevent

- Assuming file metadata guarantees valid morphology.
- Assuming NeuroML2 morphology import also imports mechanisms.
- Ignoring SWC branch type mapping.
- Not validating before simulation.
- Confusing declaration-time morphology with runtime CV topology.
- Building mechanisms before confirming morphology targets.

## Placeholder for future examples or validation checks

- Manual branch construction blueprint.
- SWC load and validation blueprint.
- ASC load and validation blueprint.
- NeuroML2 import blueprint.
- NeuroMorpho download/cache blueprint.
- Morphology checkpoint blueprint.
