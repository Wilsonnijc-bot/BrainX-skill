---
name: braincell-singlecell
description: Use when working with BrainCell single-compartment HH-style point neurons, ion/channel choices, current-clamp simulations, solver selection, FI curves, channel ablation, adaptation/rebound mechanisms, or vectorized point-neuron populations where explicit morphology, regions, locsets, paint/place, CV policies, network wiring, or cable geometry are out of scope.
---

# BrainCell Single-Compartment Modeling

## Purpose And Boundary

Use this skill for BrainCell point-neuron work: Hodgkin-Huxley style `SingleCompartment` cells, built-in ion/channel combinations, direct current injection through `cell.update(I)`, solver choice, channel ablation, FI curves, adaptation/rebound mechanisms, and vectorized populations of independent point neurons.

If the task mentions soma/dendrite/axon structure, branches, morphology files, `Cell`, `Morphology`, regions, locsets, `paint`, `place`, CV policies, cable equations, distributed mechanisms, point clamps/probes on a morphology, SWC/ASC/NeuroML2, manual branch construction, or custom channel authoring, route only to the parent reference `references/multicompartment/multicompartment-cell-workflow.md`. That parent alone selects its nested child references.


## P0 Concepts

- `SingleCompartment` is the BrainCell point-neuron front end. It has no morphology layer, represents one isopotential compartment, and is built by subclassing `braincell.SingleCompartment` and attaching ions/channels in `__init__`.
- `SingleCompartment` and multicompartment `Cell` both inherit from `HHTypedNeuron`; the integrator, ion abstractions, and channel abstractions are shared. Do not invent a different channel API for point neurons.
- `size` is a batch dimension. `HH(100)` means 100 independent point neurons, not 100 compartments.
- BrainCell requires BrainUnit quantities for physical values. Import `brainunit as u`; pass quantities for voltage, current, conductance, capacitance, time, area, concentration, and temperature. Bare floats for physical parameters commonly become `TypeError: ... expected a quantity`.
- The single-compartment membrane equation is the HH current balance: capacitance times `dV/dt` equals channel currents plus external current. In practice, channels contribute currents of the form `g(V, state) * (E - V)`.
- Rule: for single-compartment HH models, keep `C`, `g_max`, and injected `I` all density-based by default; only multiply by area when converting the entire model consistently to total capacitance, total conductance, and total current. Open the first-layer `references/area-scaled-hh-pattern.md` when that conversion is required.
- `V_th` is a spike-event threshold used for spike recording/output. It is not a reversal potential, conductance, or biological voltage clamp.
- An ion container owns reversal potential/concentration behavior; a channel owns gating/current behavior. Add channels to the ion whose `E` drives that current:

```python
na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
na.add(INa=braincell.channel.Na_HH1952(size))
```

- Match each channel's `root_type`. Sodium channels go on sodium ions, potassium channels on potassium ions, calcium channels on calcium ions. `IL` is a leak channel whose root is the cell, so assign it directly on `self`, not inside an ion.
- Use `braincell.MixIons(k, ca)` only when a channel depends on multiple ions, such as calcium-activated potassium currents needing both K reversal potential and intracellular Ca concentration. Keep ion order consistent with the channel's `root_type`. Open the first-layer `references/mixions-for-adaptation.md` for adaptation, AHP/KCa, rebound, or dynamic-calcium composition.
- Fixed ions (`SodiumFixed`, `PotassiumFixed`, `CalciumFixed`) are the simplest choice when reversal potential is constant. `InitNernst` ions derive `E` from fixed concentrations at initialization/reset. Dynamic ions such as `CalciumDetailed` make intracellular concentration a differential-equation state and compute `E` from current concentration.
- Integration is selected by solver name when constructing the cell. `braincell.quad` registers solver names; for single-compartment HH-style models, prefer `exp_euler` for accurate traces and `ind_exp_euler` for faster large sweeps/networks unless the user asks for another solver.
- Execution uses BrainState state and transforms: call `init_state()`, set `dt` and current time through `brainstate.environ.context`, advance with `cell.update(I)`, and use `brainstate.transform.for_loop` for simulations.

## Canonical Workflow

1. Confirm the task is point-neuron modeling. Route away if explicit morphology or spatial targeting appears.
2. Import `braincell`, `brainstate`, and `brainunit as u`; import `braintools` only when using initializers such as `braintools.init.Constant`.
3. Subclass `braincell.SingleCompartment`; pass `size`, `solver`, optional `C`, optional `V_initializer`, and optional `V_th` to `super().__init__`.
4. Add ion containers to `self`, then attach channels with `ion.add(NAME=braincell.channel.ChannelClass(size, ...))`.
5. Attach leak and other root-cell channels directly on `self`.
6. Initialize with `cell.init_state()` before simulation.
7. In each step, set time with `brainstate.environ.context(t=t)`, call `cell.update(input_current)`, and return `cell.V.value`, `cell.spike.value`, or other states/currents needed by the task.
8. For current sweeps or independent populations, make `size=N` and pass an `N`-vector current quantity.
9. For plots or scalar metrics, convert units explicitly, for example `times / u.ms`, `V / u.mV`, or `value.to_decimal(target_unit)`.

## Minimal HH Cell

Use this as the default classical Hodgkin-Huxley single-compartment starting point.

```python
import braincell
import brainunit as u


class HH(braincell.SingleCompartment):
    def __init__(self, size, solver="exp_euler"):
        super().__init__(size, V_th=20. * u.mV, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        self.IL = braincell.channel.IL(
            size,
            E=-54.387 * u.mV,
            g_max=0.03 * (u.mS / u.cm ** 2),
        )
```

Code decisions:

- Use `V_th` when spike events must be recorded.
- Keep Na/K as ion containers because their channels use those ion reversal potentials.
- Keep `IL` directly on the cell because its root is `HHTypedNeuron`, not a sodium/potassium/calcium ion.
- Do not add morphology, `Cell`, `paint`, `place`, regions, locsets, probes, or CV policies to this path.

## Run A Current-Clamp Simulation

For single-compartment cells, inject current by passing a BrainUnit current quantity to `update`. Do not use multicompartment `CurrentClamp` unless the task has explicit morphology/place semantics.

```python
import brainstate
import brainunit as u

neuron = HH(1)
neuron.init_state()

I = 5. * u.uA / u.cm ** 2


def step(t):
    with brainstate.environ.context(t=t):
        neuron.update(I)
    return neuron.V.value, neuron.spike.value


with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(0. * u.ms, 100. * u.ms, brainstate.environ.get_dt())
    vs, spikes = brainstate.transform.for_loop(step, times)
```

Operational notes:

- Set `dt` outside the loop and `t` inside the loop.
- Return state values from the loop body; do not mutate Python lists under transformed loops.
- Use `neuron.spike.value` only when the cell was constructed with an appropriate threshold.

## Existing Ions And Channels

Open `references/libraries/ion-library.md` and `references/libraries/channel-library.md` to find built-ins covering common sodium, potassium, calcium, K-Ca, HCN, and leak currents.

Open `references/mixions-for-adaptation.md` directly from this skill when a point-neuron task needs calcium-dependent adaptation, AHP/KCa, rebound, dynamic calcium, or `MixIons(k, ca)`. This is a first-layer BrainCell reference, not a morphology-only leaf.

When selecting a channel:

```python
braincell.channel.Na_HH1952.root_type      # braincell.ion.Sodium
braincell.channel.KDR_Ba2002.root_type     # braincell.ion.Potassium
braincell.channel.CaT_HM1992.root_type     # braincell.ion.Calcium
braincell.channel.AHP_De1994.root_type     # joint K/Ca dependency
braincell.channel.IL.root_type             # braincell.HHTypedNeuron
```

Then attach it at the matching level:

```python
self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
self.na.add(INa=braincell.channel.Na_HH1952(size))

self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
self.k.add(IK=braincell.channel.K_HH1952(size))

self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))
```


## Vectorized FI Curve Pattern

Use `size=N` for current sweeps over independent point neurons. Drive the cell with an `N`-vector current quantity, discard warm-up, and count spikes after transients.

```python
import numpy as np

n_levels = 11
amplitudes = np.linspace(0., 20., n_levels)
I = amplitudes * (u.uA / u.cm ** 2)

net = HH(n_levels)
net.init_state()

warmup = 100. * u.ms
total = 600. * u.ms


def step(t):
    with brainstate.environ.context(t=t):
        net.update(I)
    return t, net.spike.value


with brainstate.environ.context(dt=0.01 * u.ms):
    times = u.math.arange(0. * u.ms, total, brainstate.environ.get_dt())
    ts, spikes = brainstate.transform.for_loop(step, times)

mask = ts >= warmup
counts = np.asarray(u.math.sum(spikes[mask], axis=0))
rate_hz = counts / float((total - warmup) / u.second)
```

Use the same vectorized shape for ablations or parameter sweeps when each neuron is independent.

## Channel Ablation Pattern

Expose the target conductance as a constructor argument, instantiate an intact cell and an ablated cell, and compare traces under the same input.

```python
class HHWithGK(braincell.SingleCompartment):
    def __init__(self, size, gK=36. * (u.mS / u.cm ** 2), solver="exp_euler"):
        super().__init__(size, V_th=20. * u.mV, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))
        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size, g_max=gK))
        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))


intact = HHWithGK(1)
ablated = HHWithGK(1, gK=0. * (u.mS / u.cm ** 2))
```

Use zero conductance for ablation, not deletion of the ion container, unless the task explicitly asks to remove all state and current machinery.

## Channel Gating Diagnostic

Use `references/scripts/calcium_channel_gating.py` when the task asks for voltage-dependent gating curves, steady-state activation/inactivation, low-threshold vs high-threshold calcium channel comparison, or direct inspection of channel methods such as `f_p_inf` and `f_q_inf`.

This is a channel-level diagnostic pattern, not a normal current-clamp simulation. It directly instantiates channel classes, sweeps voltage, supplies ion information, computes steady-state gates, and plots activation/inactivation curves. Do not use it as the default pattern for simulating membrane-potential traces.

## Solver Guidance

- Default to `exp_euler` for high-precision single-cell HH traces.
- Use `ind_exp_euler` for large vectorized sweeps or network simulations when speed matters and small waveform deviations are acceptable.
- Use `rk4` when the user explicitly asks to compare explicit Runge-Kutta behavior or reproduce a source that chose it.
- Treat solver-dependent trace differences as numerical effects first, not biological conclusions.
- Open `references/libraries/solver-library-with-effects.md` for the current integrator families, registry APIs, and solver-effect examples.

## Network-Embedded Cells

For isolated-cell or intracellular biophysics tasks, remain within BrainCell.

When BrainCell neurons are connected into an SNN, use BrainCell for cellular and compartmental dynamics, BrainPy for synapses and projections, and open the `brainevent` skill when spike communication is sparse, large-scale, plastic, or requires specialized connectivity representations.

For general E-I network construction, leave this skill and route to `skills/brainpy/SKILL.md`. BrainCell owns no SNN-workflow reference.

## Custom Channel Or Ion Route

Do not teach or route directly to custom HH/Markov ion or channel authoring from this skill. Custom authoring is an exclusive nested child of `references/multicompartment/multicompartment-cell-workflow.md`; only that parent may select it after establishing a multicompartment task.

## Common Mistakes And Fixes

- User asks for dendrites, soma, geometry, morphology import, regions, locsets, CVs, or `paint/place` -> open `references/multicompartment/multicompartment-cell-workflow.md`; do not jump directly to a morphology leaf.
- Bare floats for physical parameters -> attach BrainUnit units.
- Channel added to wrong ion -> check `channel.root_type`; move it to the matching ion or `MixIons`.
- Leak placed inside sodium/potassium/calcium -> assign `self.IL = braincell.channel.IL(...)` directly on the cell.
- `init_state()` forgotten -> initialize before the first `update`.
- Python lists mutated inside transformed loops -> return values from `brainstate.transform.for_loop`.
- `V_th` tuned as if it changes the membrane equation -> treat it as spike-event threshold only.
- Mixing density-based and total quantities -> keep `C`, `g_max`, and injected `I` all density-based by default, or convert the entire model consistently with area.
- Custom authoring started too early -> open ion/channel libraries first.

## Full Script References

Core single-cell scripts:

- `references/scripts/hh_neuron_basics.py`
  Use as the default end-to-end HH point-neuron script: subclass `SingleCompartment`, attach Na/K/leak, initialize state, inject current through `update(I)`, run `for_loop`, and plot voltage/spikes.
  Source mirrored: https://brainx.chaobrain.com/braincell/examples/hh_neuron_basics.html

- `references/scripts/fi_curve.py`
  Use for FI curves, current sweeps, vectorized independent point neurons, warm-up discard, spike counting, and firing-rate extraction.
  Source mirrored: https://brainx.chaobrain.com/braincell/examples/fi_curve.html

- `references/scripts/channel_ablation.py`
  Use for intact-vs-ablated comparisons, especially setting a channel conductance to zero while preserving the ion/channel structure.
  Source mirrored: https://brainx.chaobrain.com/braincell/examples/channel_ablation.html

Advanced single-cell mechanism scripts:

- `references/scripts/spike_frequency_adaptation.py`
  Use for calcium-dependent spike-frequency adaptation, dynamic calcium, `MixIons(k, ca)`, and AHP/KCa mechanisms.
  Source mirrored: https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html

- `references/scripts/t_current_rebound.py`
  Use for post-inhibitory rebound, T-type calcium current, thalamic-style rebound bursting, and time-dependent hyperpolarizing current protocols.
  Source mirrored: https://brainx.chaobrain.com/braincell/examples/t_current_rebound.html

- `references/scripts/thalamic_neurons.py`
  Use for advanced thalamic point-neuron variants, richer channel compositions, calcium dynamics, HCN/AHP/T-type mechanisms, and phenotype comparison.
  Source mirrored: https://brainx.chaobrain.com/braincell/examples/thalamic_neurons.html

Channel diagnostic script:

- `references/scripts/calcium_channel_gating.py`
  Use for voltage-dependent gating curves, steady-state activation/inactivation, low-threshold vs high-threshold calcium channel comparison, and direct channel-method inspection. This is a channel-level diagnostic script, not the default current-clamp simulation pattern.
  Source mirrored: https://brainx.chaobrain.com/braincell/examples/calcium_channel_gating.html

## First-Layer Reference Routing

Local:

- `references/area-scaled-hh-pattern.md` - density-to-total capacitance, conductance, and current conversion.
- `references/mixions-for-adaptation.md` - direct adaptation/AHP/KCa/rebound route for point-neuron use and reuse by the nested multicompartment parent.
- `references/multicompartment/multicompartment-cell-workflow.md` - the only direct multicompartment route; it exclusively owns custom channel authoring, manual construction, morphology IO, filters, CV policies, probes, and topology children.
- `references/libraries/ion-library.md` - built-in ions, fixed/init-Nernst/dynamic ion choices, concentration dynamics, `MixIons`.
- `references/libraries/channel-library.md` - built-in channel families, channel dependencies, HH/Markov/custom decision rules.
- `references/libraries/solver-library-with-effects.md` - integrator names, solver families, speed/accuracy guidance.

Do not list or open custom-channel-authoring, manual-construction, morphology-IO, morphology-filter, CV-policy, probe, or topology children directly from this skill. They are nested exclusively beneath `references/multicompartment/multicompartment-cell-workflow.md`.

Secondary route references:

- Solver comparison only: https://brainx.chaobrain.com/braincell/examples/integration_methods.html
- Point-neuron E-I network route only: https://brainx.chaobrain.com/braincell/examples/ei_network.html
