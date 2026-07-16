---
name: brainx-skill-bundle
description: Routes BrainX requests across BrainUnit, BrainState, BrainCell, BrainEvent, BrainMass, BrainPy, BrainTrace, diagnostics, acceleration, and installation. Use when a request involves BrainX generally, requires choosing the correct BrainX package or skill, or should use BrainX tools instead of ad hoc NumPy/JAX simulation code.
---

# BrainX Skill Bundle Router

## Objective

This bundle routes the full local BrainX skill set cleanly and correctly: **BrainUnit**, **BrainState**, **BrainCell**, **BrainEvent**, **BrainMass**, **BrainPy**, and **BrainTrace**.

The agent should use BrainX packages for simulations, stateful computation, physical quantities, morphology handling, and transformed execution instead of writing random NumPy/JAX code when BrainX tools are appropriate.

## Progressive-disclosure rule

Start with the smallest skill that answers the user’s intent.

Use this order:

1. Route to the right domain.
2. Open the smallest relevant skill.
3. Open references only when the skill needs details.
4. Use examples or validation checks only after the concept/API boundary is clear.

Do not over-explain. Assume the agent is smart. Give the next correct page, invariant, or workflow.

## Workspace reference copies

The workspace router keeps its own reference copies under `references/`. Every package skill separately owns every Markdown reference it opens under `skills/<skill>/references/`; if several consumers need the same material, retain a copy for each consumer.

- `references/brainstate/`, `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`, and `references/brainstate-dynamics/` contain the workspace's BrainState copies.
- `references/braincell/`, `references/libraries/`, and `references/diagnostics/` contain the workspace's BrainCell, reusable-library, and diagnostic copies.
- `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md` remains the first-hop multicompartment parent reference because it alone selects its nested BrainCell children.

Use `references/index.md` as the workspace map before opening a detailed reference.

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

Use `skills/brainunit/SKILL.md` when the user asks about:

- Physical units.
- Voltage, current, time, conductance, capacitance, concentration.
- Unit conversion.
- Dimensional mismatch.
- Bare numbers passed into BrainCell code.

### BrainState

Use `skills/brainstate/SKILL.md` for mutable values, `.value`, State subclasses, randomness, Modules, layers, composition, nested models, state traversal, size inference, and state-aware transformations.

Use `references/brainstate-dynamics/dynamics-and-integration.md` from `skills/brainstate/SKILL.md` for BrainState / brainpy.state-style neural dynamics: `Dynamics` modules, `update()`, LIF-like models, state evolution, spiking populations, synaptic delays, event-driven spike operators, or SNN simulation/training.

Use `references/brainstate/parameter-constraints-regularization.md` for learnable parameters, constrained parameters, parameter transforms, and regularization.

Use `skills/brainx-acceleration-audit/SKILL.md` from `skills/brainstate/SKILL.md` when auditing or refactoring BrainX/BrainState simulation code for performance, replacing slow Python/NumPy loops with state-aware BrainState transform patterns, or benchmarking `jit`, `scan`, `vmap`, `grad`, checkpointed scans, multi-device mapping, shape stability, RNG safety, and warm runtime.

Use `references/brainstate/brainstate-control-flow-patterns.md` when looping or branching must remain valid under JAX/BrainState transformations.

Use `skills/brainstate/references/deeplearning-training/supervised-training-workflows.md` from the main BrainState skill for losses, optimizers, metrics, training steps, evaluation, and complete training loops.

Use `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` for seeds, stochasticity, random initialization, dropout, random batches, random connectivity, random spike trains, independent RNG streams, reproducibility, or checkpointed RNG state.

BrainState routing shortcuts:

- LIF / `update()` / neural dynamics / spiking population / delays / event-driven SNN:
  → `skills/brainstate/SKILL.md`
  → `references/brainstate-dynamics/dynamics-and-integration.md`

- Looping or branching that must remain valid under `jit` / `grad` / `vmap`:
  → `skills/brainstate/SKILL.md`
  → `references/brainstate/brainstate-control-flow-patterns.md`

- Performance audit or acceleration of BrainX/BrainState simulation code:
  → `skills/brainstate/SKILL.md`
  → `skills/brainx-acceleration-audit/SKILL.md`

- Training feedforward model:
  → `skills/brainstate/SKILL.md`
  → `skills/brainstate/references/deeplearning-training/supervised-training-workflows.md`

- Training RNN / SNN / time-unrolled dynamics:
  → `skills/brainstate/SKILL.md`
  → `skills/brainstate/references/deeplearning-training/supervised-training-workflows.md`
  → `references/brainstate/brainstate-control-flow-patterns.md`
  → `references/brainstate-dynamics/dynamics-and-integration.md` when spiking/neural-dynamics-specific.

- Seeds / stochasticity / random initialization / dropout / random batches / random connectivity / random spike trains / reproducibility:
  → `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`

- Constrained or regularized learnable parameters:
  → `references/brainstate/parameter-constraints-regularization.md`

### BrainCell

- Single-compartment HH-style neuron, point cell, no morphology:
  → `skills/braincell/SKILL.md`

- Density-to-total capacitance, conductance, or current conversion using cell area:
  → `skills/braincell/SKILL.md`
  → `skills/braincell/references/area-scaled-hh-pattern.md`

- Calcium-dependent adaptation, AHP/KCa, rebound, dynamic calcium, or `MixIons(k, ca)`:
  → `skills/braincell/SKILL.md`
  → `skills/braincell/references/mixions-for-adaptation.md`

- Morphology, soma, dendrite, axon, branch, region, locset, CV, SWC, ASC, NeuroML2:
  → `skills/braincell/SKILL.md`
  → `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`

- Custom ion/channel mechanism:
  → `skills/braincell/SKILL.md`
  → `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`; custom authoring is intentionally available only as that parent's nested child.

- Morphology construction/loading/validation before full Cell creation:
  → `skills/braincell/SKILL.md`
  → `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`
  → let the multicompartment parent choose its nested manual-construction or morphology-IO leaf.

- Probes, traces, NodeTree, transformed debugging, wrong runtime behavior:
  → `references/diagnostics/brainstate-transformed-diagnostics.md` for transformed BrainState runtime diagnostics.
  → `skills/braincell/SKILL.md`
  → `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md` for BrainCell runtime/probe issues; let that parent choose the nested probe or topology child, which may then select the second-level common-failures child.

### Debugging and diagnostics

Use `references/diagnostics/brainstate-transformed-diagnostics.md` for transformed-code debugging under `jit`, `grad`, `vmap`, callbacks, runtime checks, NaN/Inf checks, and traced-value diagnostics.

For BrainCell runtime failures, route to the multicompartment parent. The workspace router does not select its second-level failure child directly.

## Control-flow routing rule

Use BrainState control-flow APIs when the module/function needs looping or branching whose execution must remain valid under JAX/BrainState transformations, especially `jit`, `grad`, and `vmap`.

Ordinary module-internal data passing is just static dataflow. Control-flow APIs are for dynamic or repeated execution structure.

A fixed call graph is module structure. A runtime-dependent or repeated call graph is control flow.

## Debugging routing rule

The workspace keeps diagnostic copies for transformed-code and BrainCell runtime debugging.

- Tracers, transformed prints, callbacks, runtime checks, NaNs, assertions, and breakpoints route to `references/diagnostics/brainstate-transformed-diagnostics.md`.
- Missing BrainCell traces, wrong probe targets, wrong runtime topology, wrong CVs, and wrong locsets route to `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`; only that parent chooses the probe, topology, filter, or CV child.

## Randomness routing rule

Randomness is a conditional reference, not a primary skill.

Open the workspace copy at `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md` when a task involves:

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

The randomness parent reference alone decides whether its advanced nested child is needed; this router never selects that child directly.

## Additional package routing

- BrainMass neural-mass simulation, forward models, fitting, catalogs, or whole-brain workflows route to `skills/brainmass/SKILL.md`.
- BrainEvent binary events, event-driven connectivity, sparse formats, or plasticity route to `skills/brainevent/SKILL.md`.
- Native BrainPy point-neuron, synapse, projection, plasticity, readout, or surrogate-gradient workflows route to `skills/brainpy/SKILL.md`.
- BrainTrace online learning, eligibility traces, ETP primitives, compiler graphs, or batching route to `skills/braintrace/SKILL.md`.

For a package outside the listed BrainX skills, report the unsupported boundary instead of inventing package-specific guidance.
