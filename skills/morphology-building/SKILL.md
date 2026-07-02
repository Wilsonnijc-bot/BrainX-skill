---
name: morphology-building
description: Guides BrainCell morphology construction and loading, including branches, points, topology, typed cables, SWC, ASC, NeuroML2, NeuroMorpho imports, validation, and checkpoints. Use when the user asks to build, inspect, import, repair, validate, visualize, or prepare neuronal morphology before creating a BrainCell Cell.
---

# Morphology Building Skill

## Purpose

Teach the agent how to build, load, validate, and inspect neuronal morphology before BrainCell multicompartment modeling.

## When to use this skill

Use when the user asks about morphology files, SWC, ASC, NeuroML2, NeuroMorpho, branches, points, topology, soma/dendrite/axon structure, geometry, morphology validation, or morphology visualization.

## When not to use this skill

Do not use for mechanism dynamics, solver choice, channel kinetics, or training unless morphology is the main issue.

Do not use for single-compartment models where geometry is intentionally ignored.

## What information this skill should eventually cover

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

- `references/braincell/morphology-loading-paths.md`
- `references/braincell/morphology-io-validation.md`
- `references/libraries/filter-function-library.md`
- `references/libraries/cv-policy-reference.md`
- `references/braincell/topology-building-and-visualization.md`
- `references/diagnostics/common-failures-index.md`

## Common mistakes this skill should prevent

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
