---
name: brainstate-state-management
description: Guides BrainState State usage, including State, ParamState, HiddenState, PyTrees, tracing, splitting, mutation discipline, state traversal, and value updates through .value. Use when the user asks about mutable values, trainable parameters, dynamical state, state containers, state updates, or safe state handling in BrainState programs.
---

### Title

Brain-state-management

### Concepts


State - the place where all mutations happens
- read it with .value
subclass
e.g ParamState for trainable parameters
HiddenState for dynamical state

### Mini titles

• Core feature of State + Important notes

• how to create
-> script

• datastructure protocols & update state
-> scripts

• Subclass states
each with scripts example

• State tracking script example

• Common failures

## References

Use these references by scope instead of keeping long examples inline.

### Full script references

- `references/scripts/lif_neuron_model.py`
  Use as the state-management LIF example: `HiddenState` for membrane potential, `ShortTermState` for last-spike time, `ParamState` for weights, and explicit `.value` reads/writes during update. Source section: Practical Example: LIF Neuron Model.
  Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html

### Related official docs

- State Management: https://brainx.chaobrain.com/brainstate/tutorials/core/01_state_and_pytrees.html
  Use for `State`, PyTree values, `.value` updates, `ParamState`, `HiddenState`, `ShortTermState`, `LongTermState`, and `StateTraceStack`.

- Dynamics and Integration: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html
  Use `references/brainstate-dynamics/` when the task moves from state containers into formal `brainstate.nn.Dynamics`, unit-aware LIF dynamics, simulation loops, delays, or update hooks.

- `references/brainstate-randomness-reproducibility/`
  Open only when state management intersects seed control, random trials, stochastic state, reproducibility, or randomness inside BrainState transforms.
