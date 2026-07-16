# Common Failures Index Blueprint

## Purpose

Provide a compact BrainState-owned index of recurring BrainX mistakes and the skill or reference that should fix each one.

## Selected by

Only `references/diagnostics/brainstate-transformed-diagnostics.md` may select this second-level router. `skills/brainstate/SKILL.md` and other first-layer references must establish transformed diagnostics through that parent instead of routing here directly.

## Should eventually cover

| Failure pattern | Route |
|---|---|
| Raw `jax.jit` around stateful BrainState code | `brainstate` + JIT expansion |
| Gradients miss trainable State | Grad expansion + training |
| `vmap` ignores state axes | VMAP expansion |
| Python loop/branch breaks under transform | Control Flow |
| Plain `print` shows tracers | Debugging Diagnostics |
| Python assert fails under transform | Error Checks |
| Bare numbers passed to physical model | BrainUnit Quantity Safety |
| Painted point mechanism or placed density mechanism | BrainCell multicompartment parent |
| Morphology loaded but not validated | BrainCell multicompartment parent → nested morphology IO |
| CV policy chosen blindly | BrainCell multicompartment parent → nested CV policy |
| Probe missing or wrong trace key | BrainCell multicompartment parent → nested probe reference |
| NeuroML2 morphology expected to include mechanisms | BrainCell multicompartment parent → nested morphology IO |
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
