---
name: braincell-singlecell
description: Guides BrainCell single-compartment HH-style neuron modeling from concepts to API workflows, including ions, channels, synapses, membrane dynamics, solvers, integration, probes, current clamps, FI curves, ablations, and vectorized point-neuron simulations. Use when geometry does not matter and the user wants conceptual or code guidance for one or many non-morphological BrainCell neurons.
---

# BrainCell Single-Compartment Modeling

## Task Boundary

Use this skill for point-neuron BrainCell work where morphology, cable geometry, branch topology, paint/place regions, CV policies, SWC/ASC/NeuroML2 loading, dendrites, soma/axon targeting, or manual morphology construction are not part of the task.

If the user mentions geometry, soma, dendrite, axon, morphology files, branches, regions, locsets, CVs, cable equations, or manual morphology construction, route to `braincell-multicompartment` and open `references/braincell/braincell-manual-morphology-construction.md`.

Start with this clarification when the boundary is ambiguous:

```text
It seems like you want a single-compartment BrainCell model: ion, channel, synapse, current injection, or vectorized point-neuron dynamics without explicit morphology. I will keep geometry out unless you need dendrites, soma/axon targeting, or morphology files.
```

## Core Concepts

* `SingleCompartment` represents a neuron model without explicit spatial structure. Its membrane potential is governed by membrane capacitance, channel conductances, reversal potentials, and external injected current.
  Source: https://brainx.chaobrain.com/braincell/tutorials/single_compartment.html
* BrainCell builds neuron models on the HH framework. `SingleCompartment` and `MultiCompartment` inherit from `HHTypedNeuron`, sharing the same ion-channel mechanisms and dynamical modeling interface.
  Source: https://brainx.chaobrain.com/braincell/tutorials/single_compartment.html
* An `Ion` manages one ion species and passes `Ci/Co/E/valence` to channels as `IonInfo`; fixed ions are for fixed reversal potential or concentration, while dynamic ions such as `CalciumDetailed` make concentration an ODE state.
  Source: https://brainx.chaobrain.com/braincell/tutorials/ion.html
* A channel computes ionic current and owns channel state such as gates. Practical channel models should use existing `braincell.channel` classes first; custom channel authoring belongs in `references/braincell/braincell-custom-ion-channel-authoring.md`.
  Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html
* `MixIons` combines multiple ion containers for channels whose `root_type` depends on multiple ions, such as calcium-activated potassium currents. Ion order must match the channel `root_type`.
  Source: https://brainx.chaobrain.com/braincell/tutorials/ion.html
* `braincell.synapse` provides Markov synapse models including single-exponential `AMPA`, single-exponential `GABAa`, and double-exponential `NMDA`.
  Source: https://brainx.chaobrain.com/braincell/apis/braincell.synapse.html
* Integration is selected by solver name on cell construction. `braincell.quad` holds registered solvers; for HH point-neurons, exponential-Euler variants are the usual default because HH gating can be stiff.
  Source: https://brainx.chaobrain.com/braincell/concepts/integration.html

## Canonical Workflow

1. Confirm this is point-neuron modeling. If geometry matters, route to `braincell-multicompartment`.
2. Import `braincell`, `brainstate`, and `brainunit as u`. Use BrainUnit quantities for voltage, current, conductance, capacitance, time, area, and concentration.
3. Subclass `braincell.SingleCompartment`; call `super().__init__` with `size`, solver, and any needed initializer, threshold, or capacitance arguments.
4. Create ion containers on `self`: usually `SodiumFixed`, `PotassiumFixed`, and optionally `CalciumDetailed` / `CalciumFixed`.
5. Add channels to their matching ion container with `ion.add(INa=braincell.channel.Na_HH1952(size))` or the selected channel class. Use `MixIons(self.k, self.ca)` when a channel depends on both K and Ca.
6. Add leak as `self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))`.
7. Initialize state with `cell.init_state()`, set `brainstate.environ.context(dt=0.01 * u.ms, t=t)`, then advance with `cell.update(input_current)`.
8. Use `brainstate.transform.for_loop` for time loops and vectorized `size` for many independent neurons or current sweeps.
9. Record `cell.V.value`, `cell.spike.value`, channel states, or currents. For morphology-free single cells, do not introduce paint/place mechanisms or CV probes.

## Minimal HH Cell

Use this as the default starting point for a classical Hodgkin-Huxley single cell.

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

Explanation: sodium, potassium, and leak currents determine the membrane dynamics; `V_th` only controls spike-event emission for recording and plotting, not the membrane equation itself.

## Run A Current-Clamp Simulation

Use direct `update(I)` for point-neuron current injection. Keep time and step size in `brainstate.environ.context`.

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

## Add Existing Channels And Ions

Use built-in ions and channels before custom authoring. Put reversal potential and concentration behavior on the ion; put gating/current behavior on the channel.

```python
na = braincell.ion.SodiumFixed(1, E=50. * u.mV)
na.add(INa=braincell.channel.Na_HH1952(1))

k = braincell.ion.PotassiumFixed(1, E=-77. * u.mV)
k.add(IK=braincell.channel.K_HH1952(1))
```

For Traub-Miles style HH cells with area-scaled conductance:

```python
class AreaHH(braincell.SingleCompartment):
    def __init__(self, size, area, Cm, solver="ind_exp_euler"):
        super().__init__(size, C=Cm, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(
            INa=braincell.channel.Na_TM1991(
                size,
                g_max=(100. * u.mS * u.cm ** -2) * area,
                V_sh=-63. * u.mV,
            )
        )
        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(
            IK=braincell.channel.K_TM1991(
                size,
                g_max=(30. * u.mS * u.cm ** -2) * area,
                V_sh=-63. * u.mV,
            )
        )
        self.IL = braincell.channel.IL(
            size,
            E=-60. * u.mV,
            g_max=(5. * u.nS * u.cm ** -2) * area,
        )
```

Open `references/libraries/ion-library.md` and `references/libraries/channel-library.md` when choosing among built-in ions/channels.

## MixIons For Calcium-Activated Potassium Current

Use `MixIons` only after the channel requires more than one ion dependency. Match ion order to `channel.root_type`.

```python
import braintools


class AdaptingCell(braincell.SingleCompartment):
    def __init__(self, size, solver="ind_exp_euler"):
        super().__init__(
            size,
            V_initializer=braintools.init.Constant(-65. * u.mV),
            V_th=20. * u.mV,
            solver=solver,
        )

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-30. * u.mV))

        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))
        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV))

        self.ca = braincell.ion.CalciumDetailed(
            size,
            C_rest=5e-5 * u.mM,
            tau=10. * u.ms,
            d=0.5 * u.um,
        )
        self.ca.add(ICaL=braincell.channel.CaL_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaN=braincell.channel.CaN_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=0.3 * (u.mS / u.cm ** 2)))

        self.IL = braincell.channel.IL(size, E=-70. * u.mV, g_max=0.0075 * (u.mS / u.cm ** 2))
```

## Vectorized FI Curve Pattern

BrainCell cells are vectorized: `HH(N)` creates `N` independent neurons. Drive them with an `N`-vector of currents and count spikes after a warm-up period.

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

## Network Route

For E-I network tasks, keep the cell as `SingleCompartment` and build network wiring in BrainState / brainpy.state projection layers. The official E-I example uses an HH point-neuron population, then excitatory and inhibitory `AlignPostProj` projections with `EventFixedProb` connectivity, exponential synaptic dynamics, and conductance-based postsynaptic effects.

```python
class EINet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.num = self.n_exc + self.n_inh
        self.N = HH(self.num)
        # Add E/I projections following the official EI network example.
```

Open the official E-I network example before writing full projection code: https://brainx.chaobrain.com/braincell/examples/ei_network.html

For a request specifically about BrainCell synapse classes rather than projection wiring, start from the API-listed Markov models:

```python
ampa = braincell.synapse.AMPA
gabaa = braincell.synapse.GABAa
nmda = braincell.synapse.NMDA
```

Then open the API page for constructor signatures before writing code: https://brainx.chaobrain.com/braincell/apis/braincell.synapse.html

## Solver Guidance

* Default to `exp_euler` for high-precision single-cell HH traces unless the task already specifies another solver.
* Use `ind_exp_euler` for large vectorized or network simulations when speed matters and small waveform deviations are acceptable.
* Use `rk4` mainly when the user asks to compare explicit Runge-Kutta behavior, not as the default for stiff HH gating.
* Treat solver-dependent trace differences as numerical-method effects, not biological effects.
* Open `references/libraries/solver-library-with-effects.md` for the current integrator families, registry APIs, and solver-effect examples.

## Common Mistakes -> Fix

* Asking for dendrites/soma/geometry but using this skill -> route to `braincell-multicompartment`.
* Treating sodium ion and sodium channel as the same object -> create `SodiumFixed(size, E=50. * u.mV)`, then add `Na_*` channels to it.
* Adding a channel to the wrong ion container -> match `root_type`; sodium channels on sodium, potassium channels on potassium, calcium channels on calcium.
* Passing bare floats for voltage/current/time/conductance -> attach BrainUnit units.
* Using `add_elem` from old snippets when examples use `add` -> prefer `ion.add(INa=braincell.channel.Na_HH1952(size))` unless a local API version requires otherwise.
* Putting leak current inside an ion container -> assign `self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))` as the examples do.
* Forgetting `init_state()` before simulation -> initialize cell state before `update`.
* Running raw Python/JAX loops when compiling or recording traces -> use `brainstate.transform.for_loop`.
* Interpreting `V_th` as a conductance parameter -> it is a spike-event threshold for recording.
* Treating synapse wiring as cell morphology -> use point-neuron network projections or `braincell.synapse` classes; do not introduce regions, locsets, or CVs.
* Authoring a custom ion/channel before checking the library -> open the ion/channel references first.

## References

* `references/libraries/ion-library.md` - built-in ion choices, fixed vs dynamic ions, Nernst/concentration patterns, `MixIons`.
* `references/libraries/channel-library.md` - built-in channel families, HH/Markov/custom decision rules, ablation/adaptation examples.
* `references/libraries/solver-library-with-effects.md` - integrator names, solver family effects, speed/accuracy guidance.
* `references/braincell/braincell-custom-ion-channel-authoring.md` - use only when a requested channel is not covered by built-ins.
* `references/braincell/braincell-manual-morphology-construction.md` - route here when geometry or manual morphology construction appears.
