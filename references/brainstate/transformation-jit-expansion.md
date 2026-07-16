# Transformation JIT Expansion Blueprint

## Purpose

Detail how BrainState-aware JIT compilation should be taught after the Transformations Core skill routes here.

## Used by

- `skills/brainstate/SKILL.md`
- `skills/brainstate/references/deeplearning-training/supervised-training-workflows.md`
- `skills/braincell/SKILL.md`
- `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`

## Primary source pages to expand from

- BrainState JIT and compilation tutorial.
- BrainState transformation essentials tutorial.

## Should eventually cover

- `brainstate.transform.jit` versus raw `jax.jit`.
- State tracking and write-back under JIT.
- Static arguments and cache behavior.
- Recompilation triggers.
- Explicit compile/cache utilities if useful.
- Interop patterns with pure JAX only when code is stateless.
- JIT in simulation loops and training steps.

## Keep out

- Full gradient tutorial.
- Full vectorization tutorial.
- Control-flow patterns except as routing notes.

## Common mistakes to document

- Raw `jax.jit` around stateful BrainState Modules.
- Expecting Python side effects to behave normally under JIT.
- Shape-changing inputs causing unexpected recompilation.
- Plain `print` inside JIT showing tracers.

## Placeholder examples

- Minimal state-aware JIT.
- JIT training step.
- JIT simulation step.
- JIT cache/recompile diagnostic.
