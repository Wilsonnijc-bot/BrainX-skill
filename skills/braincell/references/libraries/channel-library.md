# BrainCell Channel Library

## Purpose

Catalog BrainCell channel families and channel-modeling patterns so agents reuse built-in components before writing custom channels.

## Source Pages

* https://brainx.chaobrain.com/braincell/concepts/ions_channels.html
* https://brainx.chaobrain.com/braincell/tutorials/channel.html
* https://brainx.chaobrain.com/braincell/apis/braincell.channel.html
* https://brainx.chaobrain.com/braincell/examples/channel_ablation.html
* https://brainx.chaobrain.com/braincell/examples/spike_frequency_adaptation.html

## Core Model

* A channel computes ionic current and usually owns voltage-gated or concentration-dependent state variables.
* `IonChannel` provides `current`, `init_state`, `reset_state`, `compute_derivative`, `pre_integral`, and `post_integral`.
* The channel tutorial recommends `braincell.Channel` as the practical base class for custom channel models.
* Built-in channels are organized by ion family and literature/source-year names such as `Na_HH1952`, `K_TM1991`, or `CaT_HM1992`.
* Always check this library and the official API page before custom authoring.

## Built-In Families

| Family | Examples | Typical use |
| --- | --- | --- |
| Sodium | `Na_HH1952`, `Na_TM1991`, `Na_Ba2002`, `NaF_SU2015_DCN`, `NaP_SU2015_DCN`, `Nav*` templates | Spike initiation and depolarizing inward current. |
| Potassium | `K_HH1952`, `K_TM1991`, `K_Leak`, `KDR_Ba2002`, `KA1_HM1992`, `KA2_HM1992`, `KM_*`, `Kv*`, `Kir*` templates | Repolarization, leak, delayed rectifier, A-current, M-current, inward rectifier. |
| Calcium | `CaL_IS2008`, `CaN_IS2008`, `CaT_HM1992`, `CaT_HP1992`, `CaHT_HM1992`, `CaHVA_*`, `CaLVA_*`, `Cav*` templates | Calcium entry, low/high-threshold currents, rebound, calcium-dependent mechanisms. |
| Potassium-calcium | `AHP_De1994`, `SK_SU2015_DCN`, `Kca*` templates | Spike-frequency adaptation and calcium-activated outward current. |
| HCN | `HCN_HM1992`, `HCN_SU2015_DCN`, `HCN_ZH2019_IO`, `HCN1_*`, `HCN2_*` | Hyperpolarization-activated current and rebound/excitability control. |
| Leak | `IL`, `LeakageChannel`, `K_Leak` | Passive leak current and resting conductance. |

The API page lists many template-based imports beyond these examples. Prefer exact literature/channel matches when the user names a model or paper.

## Minimal Built-In Channel Pattern

```python
self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
self.na.add(INa=braincell.channel.Na_HH1952(size))

self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
self.k.add(IK=braincell.channel.K_HH1952(size))

self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))
```

## Channel Selection Patterns

* Classical HH point neuron -> `Na_HH1952`, `K_HH1952`, `IL`.
* Area-scaled E-I network HH neuron -> `Na_TM1991`, `K_TM1991`, `IL` with conductance multiplied by membrane area.
* Potassium ablation study -> parameterize `g_max` on `K_HH1952` or the target potassium channel; set it to zero for the ablated condition.
* Spike-frequency adaptation -> combine calcium channels (`CaL_IS2008`, `CaN_IS2008`, `CaT_HM1992`, `CaHT_HM1992`) with `AHP_De1994` on `MixIons(k, ca)`.
* T-current rebound -> use low-threshold calcium channels such as `CaT_HM1992` or `CaT_HP1992` with appropriate calcium ion dynamics.
* HCN/rebound or sag behavior -> check `HCN_*` channels before custom work.

## Ablation Pattern

Expose the maximal conductance of the channel under test. The official potassium-ablation example compares intact HH to `gK = 0`.

```python
class HH(braincell.SingleCompartment):
    def __init__(self, size, gK=36. * (u.mS / u.cm ** 2), solver="exp_euler"):
        super().__init__(size, V_th=20. * u.mV, solver=solver)
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))
        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size, g_max=gK))
        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))

intact = HH(1)
ablated = HH(1, gK=0. * (u.mS / u.cm ** 2))
```

## Custom Channel Route

If no built-in channel fits a multicompartment task, return to `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`; only that parent may select its custom-authoring child. This library must not route to the child directly. The custom route must include:

* `root_type` aligned with the ion dependency.
* state initialization/reset.
* derivative calculation for channel state.
* `current(V, IonInfo)` with BrainUnit quantities preserved.
* HH or Markov template use when possible.

## Common Mistakes -> Fix

* Writing a custom channel first -> inspect `braincell.channel` API families first.
* Adding a channel under the wrong ion -> match the channel `root_type`.
* Forgetting `g_max` units -> use conductance density or total conductance units expected by the example.
* Confusing leak current with an ion container -> assign `self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))`.
* Comparing ablated and intact cells with different current input or solver -> keep all non-ablated parameters identical.
* Treating solver artifacts as channel effects -> compare solver settings separately with the integration reference.
