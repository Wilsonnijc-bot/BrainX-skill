---
name: brainpy
description: Use when working with native BrainPy-style brainpy.state simulations and trainable spiking neural networks: point neurons, synapses, synaptic outputs, projections, plasticity, readouts, unitful simulation, or surrogate-gradient training.
---

# BrainPy State

## Purpose And Boundary

Use this skill for native BrainPy-style brainpy.state simulations and trainable spiking neural networks: point neurons, synapses, synaptic outputs, projections, plasticity, readouts, unitful simulation, and surrogate-gradient training.

Do not use NEST-compatible APIs in this skill body. Route those requests to the separate `NEST-compatible/nest-workflow.md` branch; keep Builder/Network style as later disclosure.

The skill body should stay compact: the official skill guidance says concise skills work better, and SKILL.md should act as an overview that points to additional files only when needed.

## P0 Concepts

- **Native BrainPy-style layer**
  Source: https://brainx.chaobrain.com/brainpy-state/brainpy-style/index.html
  The native BrainPy-style layer is composable spiking neurons, synapses, outputs, projections, plasticity, and readouts driven with brainstate.transform loops; the same building blocks serve simulation and training.

- **State**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/state-paradigm.html
  A State wraps a value and exposes it through .value; because the container is tracked, the framework can collect it, batch it, checkpoint it, and route gradients through it.

- **Units**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/physical-units.html
  A unitful value is a Quantity = magnitude × unit; arithmetic checks dimensions, conversions are explicit, and incompatible combinations raise before a single time step runs.

- **Model hierarchy / Dynamics contract**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/model-anatomy.html
  brainpy.state models form the hierarchy brainstate.nn.Module → Dynamics → Neuron / Synapse; every Dynamics declares state, allocates it with init_all_states, advances one step with update, and exposes outputs.

- **Neuron anatomy**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/model-anatomy.html
  A neuron population owns membrane potential V and unitful parameters; calling the neuron with input current advances V one dt, emits a spike when threshold is crossed, and resets.

- **Projection anatomy: comm, syn, out, post**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html
  Every projection is composed from four interchangeable roles: comm maps presynaptic signal to postsynaptic signal, syn filters spikes, out converts conductance to current, and post receives the current.

- **AlignPre / AlignPost**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html
  AlignPre and AlignPost reduce synaptic state from per-synapse to per-neuron exactly, not approximately; the choice is whether the surviving state lives on the presynaptic or postsynaptic dimension.

- **Projection choice**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html
  Reach for AlignPost when synapses are exponential-family and wiring fans in; reach for AlignPre when the synapse is nonlinear or one source fans out to many targets; use CurrentProj / DeltaProj for direct current or no synapse dynamics.

- **Differentiability**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/differentiability.html
  Spiking models are differentiable: a surrogate gradient replaces the non-differentiable threshold, and BPTT runs over the same transform loops used for simulation.

- **Online learning boundary**
  Source: https://brainx.chaobrain.com/brainpy-state/concepts/online-learning.html
  Online learning updates as the network runs without storing the whole trajectory; the executable engine lives in separate package braintrace, so keep this skill to BrainPy modeling and point to BrainTrace for online learning.

## Minimal Official Workflow: Single Neuron Rollout

Source: https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/01-first-neuron.html

```python
import brainpy
import brainstate
import braintools
import brainunit as u

with brainstate.environ.context(dt=0.1 * u.ms):
    neuron = brainpy.state.LIFRef(
        1,                     # one neuron
        R=1. * u.ohm,          # membrane resistance
        tau=20. * u.ms,        # membrane time constant
        V_rest=-60. * u.mV,    # resting potential
        V_th=-50. * u.mV,      # spike threshold
        V_reset=-60. * u.mV,   # reset after a spike
        tau_ref=5. * u.ms,     # refractory period
    )
    # Allocate the neuron's state variables (V, last_spike_time).
    brainstate.nn.init_all_states(neuron)

with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 200. * u.ms, brainstate.environ.get_dt())

    def step(t):
        with brainstate.environ.context(t=t):
            neuron(25. * u.mA)                 # inject a supra-threshold current
            return neuron.V.value, neuron.get_spike()

    vs, spikes = brainstate.transform.for_loop(step, times)

print('membrane trace shape:', vs.shape)
print('total spikes:', float(u.math.sum(spikes)))
```

Official output:

```text
membrane trace shape: (2000, 1)
total spikes: 17.0
```

A neuron is a state-based Module; advance it one step per call, and run many steps with brainstate.transform.for_loop, not a bare Python time loop.

## Representative General Flow: Synapse + Projection

Source: https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/02-synapse-and-projection.html

```python
class TwoPop(brainstate.nn.Module):
    def __init__(self, n_pre=20, n_post=10):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.pre = brainpy.state.LIFRef(
            n_pre, tau=20. * u.ms, tau_ref=5. * u.ms,
            V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
        )
        self.post = brainpy.state.LIFRef(
            n_post, tau=20. * u.ms, tau_ref=5. * u.ms,
            V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
        )
        # one projection: pre -> post
        self.proj = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(
                n_pre, n_post, conn_num=0.5, conn_weight=0.5 * u.mS),
            syn=brainpy.state.Expon.desc(n_post, tau=5. * u.ms),
            out=brainpy.state.COBA.desc(E=0. * u.mV),
            post=self.post,
        )

    def update(self, t, drive):
        with brainstate.environ.context(t=t):
            pre_spikes = self.pre.get_spike() != 0.
            self.proj(pre_spikes)                  # route spikes through the synapse
            self.pre(drive)                        # advance presynaptic neurons
            self.post(0. * u.mA)                   # post driven only by the synapse
            return (self.post.V.value,
                    self.proj.syn.g.value,         # postsynaptic conductance
                    pre_spikes)

with brainstate.environ.context(dt=0.1 * u.ms):
    net = TwoPop()
    brainstate.nn.init_all_states(net)
    times = u.math.arange(0. * u.ms, 200. * u.ms, brainstate.environ.get_dt())
    post_V, syn_g, pre_spk = brainstate.transform.for_loop(
        lambda t: net.update(t, 30. * u.mA), times)

print('postsynaptic V:', post_V.shape)
print('synaptic conductance g:', syn_g.shape, '(unit:', u.get_unit(syn_g), ')')
```

Official output:

```text
postsynaptic V: (2000, 10)
synaptic conductance g: (2000, 10) (unit: mS )
```

The projection bundles comm, syn, out, and post; AlignPostProj keeps synaptic state aligned to postsynaptic neurons so memory scales as O(N_post).

## Sparse Event Case

BrainPy defines neuron, synapse, projection, and network dynamics. When a projection carries sparse binary spikes through large or sparse connectivity, open the `brainevent` skill for event representations, connectivity formats, event matrix operations, plasticity operators, and custom kernels.

Do not treat BrainEvent as a separate simulator: the BrainPy simulation may still advance with fixed timesteps while BrainEvent accelerates the sparse communication performed at each step.

## Training Script: Surrogate-Gradient SNN

Source: https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html

Full local script: `references/brainstate-dynamics/scripts/training-snn.py`. Use it when the complete runnable form is more useful than the compact inline pattern below.

Open `references/braintools-optimizer-reference.md` when optimizer or learning-rate-scheduler selection goes beyond the canonical Adam pattern; keep BrainPy surrogate gradients, time unrolling, loss construction, and State initialization in this skill.

```python
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import brainstate
import brainpy
import braintools

brainstate.random.seed(0)
brainstate.environ.set(dt=1.0 * u.ms)
num_inputs, num_hidden, num_outputs = 100, 4, 2
num_steps, batch_size = 100, 128

class SNN(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        decay = 1 - u.math.exp(-brainstate.environ.get_dt() / (1 * u.ms))
        self.i2r = brainstate.nn.Sequential(
            brainstate.nn.Linear(
                n_in, n_rec,
                w_init=braintools.init.KaimingNormal(scale=7 * decay, unit=u.mA),
                b_init=braintools.init.ZeroInit(unit=u.mA),
            ),
            brainpy.state.Expon(n_rec, tau=10. * u.ms,
                                g_initializer=braintools.init.Constant(0. * u.mA)),
        )
        self.r = brainpy.state.LIF(
            n_rec, tau=20 * u.ms, V_rest=0 * u.mV, V_reset=0 * u.mV, V_th=1. * u.mV,
            spk_fun=braintools.surrogate.ReluGrad(),   # surrogate gradient for the spike
        )
        self.r2o = brainstate.nn.Linear(n_rec, n_out, w_init=braintools.init.KaimingNormal())
        self.o = brainpy.state.Expon(n_out, tau=10. * u.ms,
                                     g_initializer=braintools.init.Constant(0.))

    def update(self, spike):
        # one time step: input projection -> recurrent spikes -> readout
        return self.o(self.r2o(self.r(self.i2r(spike))))

net = SNN(num_inputs, num_hidden, num_outputs)
firing_rate = 5 * u.Hz
x_data = brainstate.random.rand(num_steps, batch_size, num_inputs) < firing_rate * brainstate.environ.get_dt()
y_data = u.math.asarray(brainstate.random.rand(batch_size) < 0.5, dtype=int)
optimizer = braintools.optim.Adam(lr=3e-3)
optimizer.register_trainable_weights(net.states(brainstate.ParamState))

def loss_fn():
    outs = brainstate.transform.for_loop(net.update, x_data)
    outs = u.math.mean(outs, axis=0)
    return braintools.metric.softmax_cross_entropy_with_integer_labels(outs, y_data).mean()

@brainstate.transform.jit
def train_step():
    brainstate.nn.init_all_states(net, batch_size=batch_size)
    grads, loss = brainstate.transform.grad(
        loss_fn, net.states(brainstate.ParamState), return_value=True)()
    optimizer.update(grads)
    return loss
```

A spiking network unrolled over time is a recurrent network; the spike threshold is handled by a surrogate gradient, and the loss runs the network over all time steps with for_loop, then differentiates with respect to ParamState.

## Alternative Pure BrainPy-Style Training Body

Source: https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/04-train-an-snn.html

Use this when the skill should point to the newer BrainPy-style tutorial instead of the broader BrainState brain-dynamics page.

```python
with brainstate.environ.context(dt=1.0 * u.ms):
    net = SNN(n_in, n_rec, n_out)
    optimizer = braintools.optim.Adam(lr=3e-3)
    optimizer.register_trainable_weights(net.states(brainstate.ParamState))

    def loss_fn():
        preds = brainstate.transform.for_loop(net.update, x_data)   # [T, B, C]
        preds = u.math.mean(preds, axis=0)                          # [B, C]
        return braintools.metric.softmax_cross_entropy_with_integer_labels(
            preds, y_data).mean()

    @brainstate.transform.jit
    def train_step():
        brainstate.nn.init_all_states(net, batch_size=num_sample)
        grads, loss = brainstate.transform.grad(
            loss_fn, net.states(brainstate.ParamState), return_value=True)()
        optimizer.update(grads)
        return loss

    losses = []
    for epoch in range(1, 201):           # outer optimization loop -- OK
        losses.append(float(train_step()))
```

The outer epoch loop is ordinary Python, but time-stepping the model goes through for_loop.

## Important Performance Concepts

### P0 execution and performance rules

- **Time-series execution**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  Never drive a repeatedly executed model with a bare Python `for` / `while` loop; `brainstate.transform` primitives lower the whole loop into one compiled XLA program and trace the body only once.

- **jit**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  Use `brainstate.transform.jit` for a single step or one-shot call: it compiles the step once, then subsequent calls reuse the compiled program. For many-step BrainPy rollouts, do not call a jitted step from a Python loop; use `for_loop` or `scan` so the whole rollout is lowered.

- **for_loop**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  Use `brainstate.transform.for_loop` for many steps when the model's `State` carries hidden variables automatically and the per-step outputs should be stacked.

```python
with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 200. * u.ms, brainstate.environ.get_dt())

    def step(t):
        with brainstate.environ.context(t=t):
            neuron(25. * u.mA)
            return neuron.V.value, neuron.get_spike()

    vs, spikes = brainstate.transform.for_loop(step, times)
```

Use this as the default BrainPy rollout pattern: define one simulation step, set `t` / `dt` through `brainstate.environ.context`, return monitored values, and let `for_loop` stack the time axis.

- **scan**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  Use `brainstate.transform.scan` when the rollout needs an explicit carry alongside model `State`, with body shape `f(carry, x) -> (carry, y)`.

```markdown
[NEEDS OFFICIAL BRAINPY-STATE SCRIPT SOURCE]
```

Use `scan` when the time-series loop needs a value carried outside the model's own `State`, for example a running statistic, task memory, curriculum variable, external drive state, or custom loss accumulator.

- **Checkpointed loop variants**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  For long rollouts under reverse-mode gradients, swap `for_loop` / `scan` for `checkpointed_for_loop` / `checkpointed_scan` only when BPTT memory would otherwise be exhausted.

- **Vmap batching**
  Source: https://brainx.chaobrain.com/braintrace/tutorials/batching.html
  Vmap-based batching is recommended: compile the computation graph for a single sample, then use `vmap` to automatically vectorize across the batch dimension.

## References

### brainpy-neuron-library.md

Source:

- https://brainx.chaobrain.com/brainpy-state/apis/brainpy-neurons.html
- https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-choose-neuron.html

Purpose: consolidated neuron library and selection guide; covers IF/LIF/LIFRef/ALIF/ExpIF/AdEx/HH/Izhikevich/etc. Open only when replacing the canonical first-neuron model.

### brainpy-synapse-library.md

Source:

- https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synapses.html

Purpose: synaptic dynamics catalog for `syn=...`: `Expon`, `DualExpon`, `Alpha`, `AMPA`, `GABAa`, `BioNMDA`. Keep separate from synaptic outputs because this is about temporal filtering / receptor dynamics.

### brainpy-synaptic-outputs.md

Source:

- https://brainx.chaobrain.com/brainpy-state/apis/brainpy-synouts.html
- https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-coba-cuba-synapses.html

Purpose: merged output catalog and COBA/CUBA decision guide; covers `COBA`, `CUBA`, `MgBlock`, reversal potential semantics, and current-based vs conductance-based projection output.

### brainpy-projection-library.md

Source:

- https://brainx.chaobrain.com/brainpy-state/apis/brainpy-projections.html
- https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html
- https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-delays.html

Purpose: projection choice and projection API reference; covers `AlignPostProj`, `align_pre_projection`, `align_post_projection`, `CurrentProj`, `DeltaProj`, gap junctions, `comm/syn/out/post`, and delayed prefetch idioms. Skill body keeps only the compact rule of thumb.

### brainpy-plasticity.md

Source:

- https://brainx.chaobrain.com/brainpy-state/apis/brainpy-plasticity.html
- https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-short-term-plasticity.html

Purpose: merged plasticity reference for `STP` and `STD`, including their internal state and projection integration through `stp=...`.

### brainpy-custom-models.md

Source:

- https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-reproduce-a-paper.html
- https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html

Purpose: custom `brainpy.state.Neuron` / `brainpy.state.Synapse` patterns, custom ODE steps, and paper-reproduction style. Open before adapting `106_COBA_HH_2007.py` or `107_gamma_oscillation_1996.py`.

### brainpy-training.md

Source:

- https://brainx.chaobrain.com/brainpy-state/concepts/differentiability.html
- https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/04-train-an-snn.html
- https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-surrogate-gradients.html
- https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-long-rollouts-checkpoint.html

Purpose: one training pillar covering differentiability, surrogate gradients, `ParamState`, `brainstate.transform.grad`, BPTT through `for_loop`, and checkpointed long rollouts. Replaces separate `differentiability.md`, `surrogate-gradients.md`, and `long-rollouts-checkpoint.md`.

### brainpy-readouts-and-inputs.md

Source:

- https://brainx.chaobrain.com/brainpy-state/apis/brainpy-readouts.html
- https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-readouts.html
- https://brainx.chaobrain.com/brainpy-state/apis/brainpy-inputs.html

Purpose: readout and input helpers for training / decoding tasks; covers `LeakyRateReadout`, spike generators, Poisson helpers, encoders, and when to add a trainable readout head.

### braintools.md

Source:

- https://brainx.chaobrain.com/braintools/

Purpose: helper package reference for initializers, surrogate functions, metrics, encoders, and visualization utilities used by BrainPy scripts. Route optimizer families, schedulers, and external optimizer wrappers to the local `references/braintools-optimizer-reference.md` copy.

## External Routing / Not Bundled Here

### online-learning.md

Decision: do not create this as a normal BrainPy reference by default.

Source:

- https://brainx.chaobrain.com/brainpy-state/concepts/online-learning.html

Purpose: route online-learning implementation questions to the BrainTrace skill/package. Keep only the conceptual boundary in this BrainPy skill body.

### NEST-compatible/nest-workflow.md

Decision: keep this as the existing separate NEST-compatible branch.

Purpose: route NEST-compatible API requests to `NEST-compatible/nest-workflow.md` without merging that material into the native BrainPy body.

## Full Bundled Script References

Include only these genuinely useful scripts.

### 103_COBA_2005.py

Source:

- https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html
- https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/103_COBA_2005.py

Purpose: representative full E/I COBA network; shared E/I population, excitatory and inhibitory `AlignPostProj`, `Expon`, and `COBA`. Use as the default full simulation reference.

### 104_CUBA_2005_version2.py

Source:

- https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html
- https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/104_CUBA_2005_version2.py

Purpose: separate E and I populations with multiple explicit projections:

```text
E -> E
E -> I
I -> E
I -> I
```

Use when the user needs decomposition into named pathways instead of one shared population.

### brunel.py

Source:

- https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html
- https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/brunel.py

Purpose: Builder / Network API style; keep as a script-only branch, not the canonical skill-body path.

### 106_COBA_HH_2007.py

Source:

- https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html
- https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/106_COBA_HH_2007.py

Purpose: custom Hodgkin-Huxley neuron model inside a COBA E/I network; open when the user needs custom neuron model anatomy.

### 107_gamma_oscillation_1996.py

Source:

- https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html
- https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/107_gamma_oscillation_1996.py

Purpose: custom neuron + custom synapse + `CurrentProj`; representative paper-reproduction script for gamma oscillation.

### 109_fast_global_oscillation.py

Canonical filename: `109_fast_global_oscillation.py`. Do not create a numeric alias.

Source:

- https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html
- https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/109_fast_global_oscillation.py

Purpose: `DeltaProj` plus delayed spike feedback using `brainstate.nn.Delay`; use for direct delta-synapse and delayed-projection patterns.

### 201_surrogate_grad_lif_fashion_mnist.py

Source:

- https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html
- https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/201_surrogate_grad_lif_fashion_mnist.py

Purpose: real-dataset SNN training; Fashion-MNIST, latency/input encoding, LIF hidden layer, synaptic filtering, optimizer, loss, train/test loop.


## Common Mistakes -> Fix

- Time-step a model with a bare Python `for` / `while` loop -> write a per-step function and pass it to `brainstate.transform.for_loop` or `brainstate.transform.scan`.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Wrap only the per-step function in `brainstate.transform.jit` and keep the Python loop -> `jit` compiles the step once, but Python still dispatches every iteration; use `for_loop` / `scan` for rollouts.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Use `scan` for every rollout -> use `for_loop` by default; use `scan` only when an explicit carry is needed outside the model's own `State`.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Manually append outputs inside the step function -> return monitored values from the step; `for_loop` / `scan` stack per-step outputs.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Run long BPTT and hit memory pressure -> replace `for_loop` / `scan` in the loss with `checkpointed_for_loop` / `checkpointed_scan`.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Construct a model and immediately call it without state allocation -> call brainstate.nn.init_all_states(model) before rollout; call it again to reset dynamical state.

- Use unitless numbers for membrane and synaptic quantities -> use brainunit quantities like 20. * u.ms, -60. * u.mV, 0.5 * u.mS, and unit-aware initializers.

- Put projection call after the postsynaptic neuron update -> call projection first so synaptic current is accumulated before post(...) integrates the step.

- Treat projection as one monolithic object -> remember the four roles: comm, syn, out, post.

- Use AlignPost for nonlinear synapse kinetics -> use align_pre_projection when synapse is nonlinear or one source fans out to many targets.

- Look for an AlignPre class -> use the function brainpy.state.align_pre_projection; AlignPost has AlignPostProj.

- Differentiate all states during training -> select trainable parameters with net.states(brainstate.ParamState); leave dynamical state as rollout state.

- Expect online learning code in BrainPy core -> keep online learning as a concept pointer and open the BrainTrace skill/package for actual APIs.
