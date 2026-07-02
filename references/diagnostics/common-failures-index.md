# Common Failures Index Blueprint

## Purpose

Provide a compact cross-bundle index of recurring BrainX mistakes and the skill/reference that should fix each one.

## Used by

All skills.

## Should eventually cover

| Failure pattern | Route |
|---|---|
| Raw `jax.jit` around stateful BrainState code | `brainstate-transformations-core` + JIT expansion |
| Gradients miss trainable State | Grad expansion + training |
| `vmap` ignores state axes | VMAP expansion |
| Python loop/branch breaks under transform | Control Flow |
| Plain `print` shows tracers | Debugging Diagnostics |
| Python assert fails under transform | Error Checks |
| Bare numbers passed to physical model | BrainUnit Quantity Safety |
| Painted point mechanism or placed density mechanism | Multicompartment Concept/API |
| Morphology loaded but not validated | Morphology Building + IO Validation |
| CV policy chosen blindly | CV Policy Reference |
| Probe missing or wrong trace key | Probe Debugging |
| NeuroML2 morphology expected to include mechanisms | Morphology IO Validation |
| Randomness not reproducible | Randomness Reference |
| Static module dataflow misrouted as control flow | Module Building + Control Flow |

## Expansion rules

- Keep each failure entry short.
- Link to one primary skill and optional references.
- Add minimal symptom wording.
- Add minimal fix wording.
- Do not turn this into a tutorial.

## Placeholder examples

- Failure-entry template.
- Symptom → cause → route table.
- New-failure harvesting checklist.
