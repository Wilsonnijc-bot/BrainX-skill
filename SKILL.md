---
name: brainx-skill-bundle
description: Routes BrainX requests across BrainUnit, BrainState, BrainCell, BrainEvent, BrainMass, BrainPy, BrainTrace, acceleration, installation, and cross-cutting guards. Use when a request involves BrainX generally or requires selecting the owning BrainX package skill.
---

# BrainX Skill Bundle Router

## Purpose

Route a BrainX request to the smallest owning skill. This router does not teach
package APIs, select package reference children, or provide executable scripts.

## Routing Protocol

1. Classify the request by package boundary.
2. Open the owning `skills/<skill>/SKILL.md`.
3. Let that skill select its first-layer reference.
4. Let an exclusive parent reference select any nested child.
5. Open a script only when the owning skill or selected reference names it.

## Package Routing

### BrainUnit

Open `skills/brainunit/SKILL.md` for physical quantities, dimensions,
conversion, unit-aware arrays or math, constants, prefixes, temperatures, and
unit validation.

### BrainState

Open `skills/brainstate/SKILL.md` for mutable `State`, `.value`, State roles,
Module graphs, size inference, randomness, parameter constraints, lifecycle
operations, and state-aware `jit`, `grad`, `vmap`, or control flow.

Open `skills/brainx-acceleration-audit/SKILL.md` only after the request becomes
a performance, batching, vectorization, memory, compilation, or device-scaling
task.

### BrainCell

Open `skills/braincell/SKILL.md` for single-compartment HH-style cells, ions,
channels, current clamps, solver choice, FI curves, ablation, adaptation,
rebound, or morphology-based cell work. The BrainCell skill decides whether to
stay on the point-cell path or open its multicompartment parent.

### BrainEvent

Open `skills/brainevent/SKILL.md` for binary events, `BinaryArray`, explicit or
generated sparse connectivity, fixed fan-in/fan-out, event plasticity, and
custom event operators.

### BrainMass

Open `skills/brainmass/SKILL.md` for neural-mass models, stochastic ensembles,
coupled networks, whole-brain workflows, forward observation models, fitting,
task training, or regime sweeps.

### BrainPy

Open `skills/brainpy/SKILL.md` for native BrainPy-State neurons, synapses,
projections, plasticity, delays, readouts, SNN simulation/training, or
NEST-compatible porting. The BrainPy skill owns the native-versus-NEST split.

### BrainTrace

Open `skills/braintrace/SKILL.md` for online learning, eligibility traces,
D-RTRL, ES-D-RTRL, pp-prop, compiler graphs, ETP primitives, hidden State, or
online-learning batching.

## Infrastructure Routing

Open `skills/brainx-install/SKILL.md` for installation, compatibility tuples,
Python/JAX versions, CPU/CUDA/TPU selection, import failures, or environment
repair. That skill requires approval before environment mutation.

Open `skills/brainx-general-guard/SKILL.md` for broad or ambiguous BrainX work
that needs package selection, unit safety, State-transform safety, or common
cross-package guardrails.

## Ownership Rules

- The root router selects skills, not nested reference leaves.
- A skill may select another skill only where its supplied routing tree declares
  that semantic route.
- Parameter, randomness, diagnostics, multicompartment, training, fitting, and
  NEST children remain behind their declared parents.
- Every Markdown reference lives under each skill that consumes it. When two
  skills need the same material, each skill owns a local copy.
- Cross-skill paths point to the other skill's `SKILL.md`, never into another
  skill's `references/` tree.
- No unsupported package guidance or extra script is inferred from an existing
  draft, file, URL, or example.

## Script Rule

This router owns no executable workflow and selects no script. Script choice,
default status, optional status, and helper dependencies belong to the owning
package skill.
