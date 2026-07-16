# BrainState Dynamics and Integration Parent Reference

## Purpose

BrainState dynamics is a specialized `Module` pattern for time-evolving systems. Use this parent reference when a primary skill has already established that the task is about time evolution rather than ordinary static module composition.

## Use when

Open this parent for `Dynamics`, `update()`, time-evolving state, LIF/SNN populations, delays, hooks, event-driven spike communication, trajectory simulation, or neural population workflows.

## Parent-reference boundary

This is the only first-layer BrainState dynamics reference. `skills/brainstate/SKILL.md`, the workspace router, and `references/index.md` route here, not to the dynamics children. This parent is the only first-hop selector for:

- `references/brainstate-dynamics/brain-dynamics-delay-protocol.md`
- `references/brainstate-dynamics/brain-dynamics-event-driven-operators.md`
- `references/brainstate-dynamics/brain-dynamics-snn-workflows.md`

After this parent establishes the dynamics task, nested children may cross-route to a sibling when needed.

## Usually reached from

- `skills/brainstate/SKILL.md`

## Also route to

- `references/brainstate/brainstate-control-flow-patterns.md` for compiled loops.
- `references/brainstate/transformation-grad-expansion.md` for differentiable simulation or fitting.
- `references/libraries/solver-library-with-effects.md` for solver/integration choices.
- BrainCell skills when morphology, mechanisms, ions/channels, probes, clamps, or cell-specific APIs become the actual task.

## Tiny RNG block

Use `brainstate.random.seed(seed)` once when dynamics examples, stochastic spike trains, random trials, noise, or randomized initial conditions must be reproducible. Route advanced stochastic dynamics, mapped random trials, or RNG under transforms to `references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`.

## Concepts

• what this skill is for
Use when the task is BrainState dynamics/integration: Dynamics modules, time stepping, update flow, before/after hooks, delays, and simulation loops. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• brain models as dynamical systems
Brain models are dynamical systems: state variables that evolve over time according to update rules. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• Dynamics
Dynamics is a module that defines how state variables evolve over time in neural or other dynamical systems. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• element-wise principle
State updates only affect local variables and do not include cross-unit interactions. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• what Dynamics is not
Dynamics does not include synaptic connections, network connectivity, or inter-neuron interactions. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• architecture flow
Input → connection modules → dynamics modules → output. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• update flow
Every Dynamics module must implement update(), which defines how state variables evolve in time. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• input / output / states / parameters
Input is external drive; output is observable quantity; states are dynamic variables; parameters are constants that don’t change during simulation. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• before / after update mechanism
Dynamics supports before-update and after-update hooks to insert custom logic at specific points in the update cycle. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• input/output size definition
Dynamics modules use in_size to define the geometry of the neuron population; out_size defaults to the same as in_size. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• delay support
Dynamics naturally supports temporal delays through the after-update mechanism. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

• BrainCell / BrainPy boundary note
Do not turn this into a BrainCell morphology or single-cell route; the official BrainState SNN workflow says concrete LIF, synapse, and projection models live in companion brainpy.state and wire together as ordinary BrainState modules. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html

### Mini title

Simulation time-step setup

#### Script

```python
brainstate.environ.set(dt=0.1 * u.ms)
```

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

#### Explanation text

Set the simulation time step before running the dynamics model. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

### Mini title

Minimal element-wise Dynamics module

#### Script

```python
class ExponentialDecay(brainstate.nn.Dynamics):
    """Simple exponential decay dynamics: τ dv/dt = -v"""
    def __init__(self, size, tau=10.0 * u.ms):
        super().__init__(in_size=size)
        self.v = brainstate.State(jnp.zeros(size))
        self.tau = tau
    def update(self, inp):
        dt = brainstate.environ.get_dt()
        alpha = jnp.exp(-dt / self.tau)
        self.v.value = self.v.value * alpha + inp
        return self.v.value
dynamics = ExponentialDecay(size=(5,), tau=5.0 * u.ms)
inp = jnp.array([1.0, 0.5, 0.0, -0.5, -1.0])
output = dynamics(inp)
```

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

#### Explanation text

This demonstrates the element-wise principle: each element evolves independently. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

### Mini title

Minimal LIF dynamics module

#### Script

```python
class SimpleLIF(brainstate.nn.Dynamics):
    """Leaky Integrate-and-Fire neuron dynamics.
    Equation: τ dV/dt = -(V - V_rest) + R*I
    Spike when V >= V_th, then reset to V_reset
    """
    def __init__(self, size, tau=10.0 * u.ms, V_rest=-65.0 * u.mV,
                 V_th=-50.0 * u.mV, V_reset=-65.0 * u.mV, R=1.0 * u.ohm):
        super().__init__(in_size=size)
        self.V = brainstate.State(jnp.ones(size) * V_rest)
        self.spike = brainstate.State(jnp.zeros(size, dtype=bool))
        self.tau = tau
        self.V_rest = V_rest
        self.V_th = V_th
        self.V_reset = V_reset
        self.R = R
    def update(self, I):
        dt = brainstate.environ.get_dt()
        alpha = jnp.exp(-dt / self.tau)
        V_inf = self.V_rest + self.R * I
        self.V.value = self.V.value * alpha + V_inf * (1 - alpha)
        self.spike.value = self.V.value >= self.V_th
        self.V.value = u.math.where(self.spike.value, self.V_reset, self.V.value)
        return self.spike.value
lif = SimpleLIF(size=(3,))
```

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

#### Explanation text

The LIF example updates membrane potential, detects spikes, resets spiking neurons, and returns spike events. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

### Mini title

Simulation loop with for_loop

#### Script

```python
lif.V.value = jnp.ones(3) * lif.V_rest
lif.spike.value = jnp.zeros(3, dtype=bool)
duration = 100 * u.ms
n_steps = int(duration / brainstate.environ.get_dt())
times = jnp.arange(n_steps) * brainstate.environ.get_dt()
I_inputs = jnp.array([20.0, 30.0, 40.0]) * u.mA
def step_run(t):
    with brainstate.environ.context(t=t):
        spikes = lif(I_inputs)
        return lif.V.value, spikes
V_history, spike_history = brainstate.transform.for_loop(step_run, times)
```

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

#### Explanation text

Use brainstate.transform.for_loop for trajectory simulation instead of a plain Python loop when compiling the run matters. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

### Mini title

Before / after update hooks

#### Script

```python
dynamics.add_before_update(key, function)
dynamics.add_after_update(key, function)
```

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

#### Explanation text

Default flow: before-update receives no update() input parameters; after-update receives the update() return value. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

### Mini title

Delay support minimal pattern

#### Script

```python
delayed_output = dynamics.output_delay(5.0 * u.ms)
value = delayed_output()
delayed_V = dynamics.prefetch_delay('V', delay_time=5.0 * u.ms)
v_delayed = delayed_V()
```

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

#### Explanation text

After each update(), the delay buffer is automatically updated through the after-update hook; no manual management required. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html

### Mini title

BrainPy-state SNN workflow note

#### Script

```python
neurons = brainpy.state.LIF(
    4,
    V_rest=-52. * u.mV,
    V_th=-50. * u.mV,
    V_reset=-60. * u.mV,
    tau=10. * u.ms,
    V_initializer=braintools.init.Constant(-60. * u.mV),
    spk_reset='soft',
)
brainstate.nn.init_all_states(neurons)
with brainstate.environ.context(t=0. * u.ms):
    spikes = neurons(8. * u.mA)
```

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html

#### Explanation text

brainpy.state.LIF is a leaky integrate-and-fire population — a BrainState Dynamics module. Source mirrored: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html

## Nested child references

references/brainstate-dynamics/brain-dynamics-delay-protocol.md

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html
**Purpose:** Explains BrainState Brain Dynamics delay APIs and delay-buffer behavior.

references/brainstate-dynamics/brain-dynamics-event-driven-operators.md

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/03_event_driven_operators.html
**Purpose:** Catalogs event-driven sparse spike operators and connectivity patterns for scalable SNNs.

references/brainstate-dynamics/brain-dynamics-snn-workflows.md

**Mirror Source URLs:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html, https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html
**Purpose:** Routes build, simulate, and train workflows for BrainState-style spiking neural networks.

references/brainstate/brainstate-control-flow-patterns.md

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html
**Purpose:** Collects loop and branch patterns that remain valid under BrainState and JAX transformations.

references/libraries/solver-library-with-effects.md

**Mirror Source URLs:** https://brainx.chaobrain.com/braincell/apis/integration.html, https://brainx.chaobrain.com/braincell/integration/solvers.html, https://brainx.chaobrain.com/braincell/integration/advanced.html
**Purpose:** Catalogs BrainCell and BrainState solver and integration choices with modeling consequences.

references/brainstate/transformation-grad-expansion.md

**Mirror Source URLs:** https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html, https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html, https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html
**Purpose:** Expands gradient and autodiff teaching for differentiable simulation and parameter fitting.

## Full bundled script references

references/brainstate-dynamics/scripts/hodgkin-huxley-neuron.py

**Source mirrored:** https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/hodgkin_huxley_neuron.html
**Purpose:** complete executable HH neuron example showing biophysical state variables and continuous-time dynamics.


references/brainstate-dynamics/scripts/building-ei-snn.py

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html
**Purpose:** complete E/I spiking-network simulation workflow with init_all_states and compiled trajectory.

references/brainstate-dynamics/scripts/training-snn.py

**Source mirrored:** https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html
**Purpose:** full representative SNN training workflow; use when the task crosses from simulation into optimization.

## Common mistakes -> Fix

• putting synaptic connectivity inside Dynamics.update() -> keep dynamics element-wise; use connection/projection modules for interactions.
• forgetting update() -> every Dynamics module must implement it.
• using unitless time/current/voltage in dynamics examples -> keep brainunit quantities from source scripts.
• running long simulations with plain Python loops -> use brainstate.transform.for_loop for compiled trajectories.
• forgetting brainstate.environ.context(t=t) -> set time context inside step functions when simulation time matters.
• manually managing delay buffers -> use output_delay / prefetch_delay; after-update synchronizes buffers.
• treating BrainState dynamics as a full BrainCell route -> branch to BrainCell when morphology, mechanisms, or cell-specific APIs are the actual task.

⸻
