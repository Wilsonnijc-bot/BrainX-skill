---
name: brainx-debugging-diagnostics
description: Diagnoses BrainX runtime and transformation problems across BrainState and BrainCell, including NaNs, assertions, tracers, transformed printing, callbacks, breakpoints, probes, traces, NodeTree, CVs, locsets, and runtime topology. Use when code fails, records nothing, produces wrong states or currents, or behaves unexpectedly under jit, grad, vmap, or BrainCell simulation.
---

# BrainX Debugging and Diagnostics Skill

## Purpose

Provide a cross-cutting diagnostic skill for BrainState transformed-code debugging and BrainCell runtime/probe debugging.

## When to use this skill

Use when the user asks why something fails, returns NaN/Inf, produces tracers, does not update state, prints confusing values, breaks under `jit`/`grad`/`vmap`, has assertion/runtime-check issues, records no BrainCell traces, attaches probes incorrectly, has wrong locset/region/CV behavior, or shows unexpected BrainCell runtime values.

## When not to use this skill

Do not use for first-pass model design unless the user is already debugging behavior. Route design questions to the relevant BrainState or BrainCell skill.

## What information this skill should eventually cover

- Transformed-code runtime checks.
- Assertions under transformations.
- NaN/Inf debugging.
- Tracer-aware debugging.
- `jax.debug.print`-style runtime printing.
- Callbacks.
- Conditional breakpoints.
- BrainCell probes and traces.
- BrainCell runtime topology and NodeTree inspection.
- CV/region/locset debugging.
- Common failures index routing.

## Expected workflow

1. Classify the failure layer: install, unit, state, transform, control flow, BrainCell runtime, or morphology.
2. Reproduce with the smallest failing case.
3. Use transformed error checks for JIT/grad/vmap runtime issues.
4. Use transformed debugging tools instead of plain `print` when inside transforms.
5. Use BrainCell probes to observe states/currents/mechanisms.
6. Inspect runtime topology when morphology/CV/locset behavior is suspicious.
7. Open randomness when behavior is non-reproducible or two runs unexpectedly differ.
8. Route back to the relevant build skill after diagnosis.

## Required / useful reference markdowns

- `references/diagnostics/brainstate-transformed-diagnostics.md`
- `references/braincell/topology-building-and-visualization.md`
- `references/braincell/probe-reference.md`
- `references/diagnostics/common-failures-index.md`
- `references/brainstate/advanced-randomness.md` when stochastic behavior is involved.

## Common mistakes this skill should prevent

- Using plain Python `print` inside transformed code and expecting runtime values.
- Using Python `assert` where transform-compatible checks are needed.
- Ignoring NaNs until after long simulations.
- Running BrainCell simulations without probes.
- Debugging morphology before checking CV/runtime topology.
- Treating missing traces as solver failure before checking probe attachment.
- Forgetting randomness/reproducibility when failures are stochastic.

## Placeholder for future examples or validation checks

- JIT tracer debugging example.
- Runtime NaN check example.
- Conditional breakpoint example.
- Probe attachment validation example.
- Empty trace diagnostic checklist.
- Runtime topology inspection checklist.
