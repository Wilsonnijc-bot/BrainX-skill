# Common Failures Index Blueprint

## Purpose

Provide a compact second-level index of recurring BrainCell morphology, topology, and probe mistakes and the canonical reference route that should fix each one.

## Used by

- `references/braincell/braincell-manual-morphology-construction.md`
- `references/braincell/topology-building-and-visualization.md`
- `references/braincell/probe-reference.md`

This file is not a first-layer reference and is not selected by `skills/braincell/SKILL.md` or the root bundle skill.

## Should eventually cover

| Failure pattern | Route |
|---|---|
| Painted point mechanism or placed density mechanism | BrainCell multicompartment parent |
| Morphology loaded but not validated | BrainCell multicompartment parent → nested morphology IO |
| CV policy chosen blindly | BrainCell multicompartment parent → nested CV policy |
| Probe missing or wrong trace key | BrainCell multicompartment parent → nested probe reference |
| NeuroML2 morphology expected to include mechanisms | BrainCell multicompartment parent → nested morphology IO |
| Runtime topology inspected before `init_state()` | BrainCell multicompartment parent → nested topology reference |
| Region or locset resolves to the wrong target | BrainCell multicompartment parent → nested filter reference |
| Custom channel authored before checking built-ins | Main BrainCell skill → first-layer channel library → first-layer custom authoring if needed |

## Expansion rules

- Keep each failure entry short.
- Route morphology failures back through the multicompartment parent and one exclusive child.
- Route custom-authoring failures through the first-layer channel library and first-layer custom-authoring reference.
- Add minimal symptom wording.
- Add minimal fix wording.
- Do not turn this into a tutorial.

## Placeholder examples

- Failure-entry template.
- Symptom → cause → route table.
- New-failure harvesting checklist.
