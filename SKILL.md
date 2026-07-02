---
name: brainx-skill-bundle
description: Routes BrainX requests across BrainUnit, BrainState, BrainCell, morphology, diagnostics references, performance acceleration, and installation. Use when a request involves BrainX generally, requires choosing the correct BrainX package or skill, or should use BrainX tools instead of ad hoc NumPy/JAX simulation code.
---

# BrainX Skill Bundle Router

## Objective

This bundle helps an AI agent master the BrainX ecosystem cleanly and correctly, especially **BrainCell**, **BrainState**, and **BrainUnit**.

The agent should use BrainX packages for simulations, stateful computation, physical quantities, morphology handling, and transformed execution instead of writing random NumPy/JAX code when BrainX tools are appropriate.

## Progressive-disclosure rule

Start with the smallest skill that answers the user’s intent.

Use this order:

1. Route to the right domain.
2. Open the smallest relevant skill.
3. Open references only when the skill needs details.
4. Use examples or validation checks only after the concept/API boundary is clear.

Do not over-explain. Assume the agent is smart. Give the next correct page, invariant, or workflow.

## Shared reference map

Shared markdown references live under `references/` by category:

- `references/brainstate/` for BrainState transforms, control flow, parameter constraints, and advanced randomness.
- `references/brain-dynamics/` for Dynamics, delays, event-driven operators, and SNN workflows.
- `references/braincell/` for morphology, probes, and runtime topology.
- `references/libraries/` for reusable catalogs such as ions, channels, solvers, CV policies, filters, layers, and activations.
- `references/diagnostics/` for common failures and transformed-code diagnostics.
- `references/policy/` for future cross-bundle policy notes.

Use `references/index.md` as the global map before opening a detailed reference.

## Top-level routing

### Installation and infrastructure

Use `skills/brainx-install/SKILL.md` when the user asks about:

- Installing BrainX packages.
- Choosing CPU/GPU/TPU environment.
- Import errors.
- Python/JAX backend validation.
- First-time setup.

Use `skills/brainx-general-guard/SKILL.md` when the task is broad, ambiguous, or likely to suffer from common infrastructure mistakes.

### BrainUnit

Use `skills/brainunit-quantity-safety/SKILL.md` when the user asks about:

- Physical units.
- Voltage, current, time, conductance, capacitance, concentration.
- Unit conversion.
- Dimensional mismatch.
- Bare numbers passed into BrainCell code.

### BrainState

Use `skills/brainstate-state-management/SKILL.md` for mutable values, `.value`, `State`, `ParamState`, hidden state, and state tracing.

Use `skills/brainstate-module-building/SKILL.md` for Modules, layers, composition, nested models, static dataflow, state traversal, and prebuilt layers.

Use `skills/brainstate-brain-dynamics/SKILL.md` for BrainState / brainpy.state-style neural dynamics: Dynamics modules, `update()`, LIF-like models, state evolution, spiking populations, synaptic delays, event-driven spike operators, or SNN simulation/training.

Use `references/brainstate/parameter-constraints-regularization.md` for learnable parameters, constrained parameters, parameter transforms, and regularization.

Use `skills/brainstate-transformations-core/SKILL.md` for JIT, gradients, vectorization, batching, sweeps, transformed execution, and state-aware BrainState transforms.

Use `skills/brainx-accelerate/SKILL.md` when auditing or refactoring BrainX/BrainState simulation code for performance, replacing slow Python/NumPy loops with state-aware BrainState transform patterns, or benchmarking `jit`, `scan`, `vmap`, `grad`, checkpointed scans, multi-device mapping, shape stability, RNG safety, and warm runtime.

Use `references/brainstate/brainstate-control-flow-patterns.md` when looping or branching must remain valid under JAX/BrainState transformations.

Use `skills/brainstate-deeplearning-training/SKILL.md` for losses, optimizers, metrics, training steps, evaluation, and training loops.

Use `skills/brainstate-randomness-reproducibility/SKILL.md` for seeds, stochasticity, random initialization, dropout, random batches, random connectivity, random spike trains, independent RNG streams, reproducibility, or checkpointed RNG state.

BrainState routing shortcuts:

- LIF / `update()` / neural dynamics / spiking population / delays / event-driven SNN:
  → `skills/brainstate-brain-dynamics/SKILL.md`

- Looping or branching that must remain valid under `jit` / `grad` / `vmap`:
  → `skills/brainstate-transformations-core/SKILL.md`
  → `references/brainstate/brainstate-control-flow-patterns.md`

- Performance audit or acceleration of BrainX/BrainState simulation code:
  → `skills/brainx-accelerate/SKILL.md`
  → `skills/brainstate-transformations-core/SKILL.md` only when exact transform semantics are needed.

- Training feedforward model:
  → `skills/brainstate-deeplearning-training/SKILL.md`

- Training RNN / SNN / time-unrolled dynamics:
  → `skills/brainstate-deeplearning-training/SKILL.md`
  → `skills/brainstate-transformations-core/SKILL.md`
  → `references/brainstate/brainstate-control-flow-patterns.md`
  → `skills/brainstate-brain-dynamics/SKILL.md` when spiking/neural-dynamics-specific.

- Seeds / stochasticity / random initialization / dropout / random batches / random connectivity / random spike trains / reproducibility:
  → `skills/brainstate-randomness-reproducibility/SKILL.md`

- Constrained or regularized learnable parameters:
  → `references/brainstate/parameter-constraints-regularization.md`

### BrainCell

- Single-compartment HH-style neuron, point cell, no morphology:
  → `skills/braincell-singlecell/SKILL.md`

- Morphology, soma, dendrite, axon, branch, region, locset, CV, SWC, ASC, NeuroML2:
  → `skills/braincell-multicompartment/SKILL.md`

- Custom ion/channel mechanism:
  → `skills/braincell-singlecell/SKILL.md` or `skills/braincell-multicompartment/SKILL.md` based on whether geometry matters.
  → `references/braincell/braincell-custom-ion-channel-authoring.md` for custom channel authoring details.

- Morphology construction/loading/validation before full Cell creation:
  → `skills/braincell-multicompartment/SKILL.md`
  → `references/braincell/braincell-manual-morphology-construction.md`
  → `references/braincell/morphology-io-loading-validation.md` for file loading and validation.

- Probes, traces, NodeTree, transformed debugging, wrong runtime behavior:
  → `references/diagnostics/common-failures-index.md`
  → `references/diagnostics/brainstate-transformed-diagnostics.md` for transformed BrainState runtime diagnostics.
  → `references/braincell/probe-reference.md` or `references/braincell/topology-building-and-visualization.md` for BrainCell runtime/probe issues.

### Debugging and diagnostics

Use `references/diagnostics/common-failures-index.md` as the first router when the user asks why something fails, returns NaN, produces tracers, does not update state, records no trace, attaches probes incorrectly, or behaves unexpectedly under `jit`, `grad`, `vmap`, control flow, or BrainCell runtime.

Use `references/diagnostics/brainstate-transformed-diagnostics.md` for transformed-code debugging under `jit`, `grad`, `vmap`, callbacks, runtime checks, NaN/Inf checks, and traced-value diagnostics.

## Control-flow routing rule

Use BrainState control-flow APIs when the module/function needs looping or branching whose execution must remain valid under JAX/BrainState transformations, especially `jit`, `grad`, and `vmap`.

Ordinary module-internal data passing is just static dataflow. Control-flow APIs are for dynamic or repeated execution structure.

A fixed call graph is module structure. A runtime-dependent or repeated call graph is control flow.

## Debugging routing rule

Transformed-code debugging and BrainCell runtime debugging share the diagnostics references.

- Tracers, transformed prints, callbacks, runtime checks, NaNs, assertions, and breakpoints route to `references/diagnostics/brainstate-transformed-diagnostics.md`.
- Missing BrainCell traces, wrong probe targets, wrong runtime topology, wrong CVs, and wrong locsets route to `references/braincell/probe-reference.md`, `references/braincell/topology-building-and-visualization.md`, and `references/diagnostics/common-failures-index.md`.

## Randomness routing rule

Randomness is a concise skill, with advanced details in a reference.

Open `skills/brainstate-randomness-reproducibility/SKILL.md` when a task involves:

- Random initialization.
- Dropout.
- Stochastic layers.
- Random batches.
- Synthetic data.
- Noisy stimuli.
- Randomized simulations.
- `vmap` over stochastic functions.
- Stochastic control flow.
- Reproducibility or checkpointed RNG state.

Open `references/brainstate/advanced-randomness.md` only after the core randomness boundary is clear.

## Unsupported domain guard

This bundle covers BrainUnit, BrainState, and BrainCell. If the user asks for BrainMass-specific guidance and no BrainMass source is provided, state that BrainMass is outside this bundle and use only general BrainX/BrainState/BrainUnit principles.
