# BrainState Control-Flow Patterns Blueprint

## Purpose

Collect detailed loop and branch patterns for BrainState transformations and acceleration workflows.

## Used by

- `skills/brainstate-module-building/SKILL.md`
- `skills/brainstate-transformations-core/SKILL.md`
- `references/brainx-acceleration-audit/`

## Primary source pages to expand from

- BrainState control-flow tutorial.

## Core trigger

Use BrainState control-flow APIs when the module/function needs looping or branching whose execution must remain valid under JAX/BrainState transformations, especially `jit`, `grad`, and `vmap`.

Ordinary module-internal data passing is just static dataflow. Control-flow APIs are for dynamic or repeated execution structure.

## Should eventually cover

- Static dataflow versus dynamic/repeated execution.
- `for_loop`.
- `scan`.
- `while_loop`.
- Bounded while loops.
- `cond`.
- `switch`.
- `ifelse`.
- Control flow with State.
- Control flow under JIT, grad, and vmap.
- Randomness inside loop bodies.
- Memory/checkpoint patterns.

## Common mistakes to document

- Python loop where transform-safe loop is required.
- Python `if` on traced arrays.
- Reusing random numbers across iterations.
- Treating a fixed layer chain as control flow.

## Placeholder examples

- Time-step scan.
- Dynamic while.
- Conditional branch.
- Recurrent module loop.
