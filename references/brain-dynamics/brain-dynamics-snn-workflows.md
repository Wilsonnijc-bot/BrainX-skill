# Brain Dynamics SNN Workflows Blueprint

## Purpose

Collect build, simulate, and train workflows for BrainState-style spiking neural networks.

## Used by

- `skills/brainstate-brain-dynamics/SKILL.md`
- `skills/brainstate-deeplearning-training/SKILL.md`

## Section 1: Build and simulate SNN

- Population of spiking neurons.
- Projection = communication operator + synapse + output + post population.
- E/I network.
- `init_all_states`.
- Trajectory via `brainstate.transform.for_loop`.
- Spike raster.

## Section 2: Train SNN

- SNN as recurrent network unrolled over time.
- Surrogate gradient.
- Loss over time.
- `for_loop` inside loss.
- `grad` over `ParamState`.
- `jit` train_step.
- Optimizer update.

## Section 3: Routes

- For generic optimizer/loss/metrics: `skills/brainstate-deeplearning-training/SKILL.md`.
- For time loops/branches: `skills/brainstate-transformations-core/SKILL.md` and `references/brainstate/brainstate-control-flow-patterns.md`.
- For stochastic data/dropout/random spikes: `skills/brainstate-randomness-reproducibility/SKILL.md`.
- For constrained trainable dynamics parameters: `references/brainstate/parameter-constraints-regularization.md`.
- For sparse spiking connectivity: `references/brain-dynamics/brain-dynamics-event-driven-operators.md`.

## Section 4: Control-flow boundary

Open `references/brainstate/brainstate-control-flow-patterns.md` when the SNN simulation or loss unrolls over time inside transformed execution, especially under `jit`, `grad`, or `vmap`.

Do not open control flow for the ordinary Python epoch loop outside transformed code.

Full runnable training scripts can live as examples or root scripts. Open them only after this workflow reference has selected the relevant build, simulation, training, control-flow, and randomness routes.

## Common mistakes to document

- Treating an SNN time unroll as an ordinary feedforward pass.
- Forgetting surrogate gradients for non-differentiable spikes.
- Putting the time loop outside the differentiable loss when BPTT is needed.
- Training stochastic SNNs without reproducible RNG handling.
- Treating projection/connectivity as local neuron Dynamics.

## Placeholder examples

- E/I SNN simulation route.
- `for_loop` trajectory route.
- Spike raster route.
- Surrogate-gradient training route.
