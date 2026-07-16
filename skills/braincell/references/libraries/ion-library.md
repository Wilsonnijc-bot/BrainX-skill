# BrainCell Ion Library

## Purpose

Catalog BrainCell ion species and ion-related modeling patterns for single-compartment and multicompartment cells.

## Source Pages

* https://brainx.chaobrain.com/braincell/concepts/ions_channels.html
* https://brainx.chaobrain.com/braincell/tutorials/ion.html
* https://brainx.chaobrain.com/braincell/apis/braincell.ion.html

## Core Model

* `Ion` manages one ion species and passes `Ci/Co/E/valence` to channels as an `IonInfo`.
* `FixedIon` has no concentration ODE; use it for fixed reversal potential or fixed concentration.
* `InitNernstIon` uses fixed `Ci/Co` and initializes `E` from the Nernst equation at init/reset.
* `DynamicNernstIon` turns intracellular concentration into an integrable state; concrete ions provide the concentration derivative.
* `CalciumDetailed` is the direct dynamic-ion example: `Ci` is an ODE state, `E` is computed live from Nernst, and `tau` controls return toward rest.

## Built-In Ion Choices

| Need | Use | Notes |
| --- | --- | --- |
| Fixed sodium reversal | `braincell.ion.SodiumFixed` | Standard HH sodium container; add `Na_*` channels. |
| Sodium Nernst from concentrations | `braincell.ion.SodiumInitNernst` | Use when fixed `Ci/Co` should initialize `E`. |
| Fixed potassium reversal | `braincell.ion.PotassiumFixed` | Standard HH potassium container; add `K_*` channels. |
| Potassium Nernst from concentrations | `braincell.ion.PotassiumInitNernst` | Use when fixed `Ci/Co` should initialize `E`. |
| Fixed calcium | `braincell.ion.CalciumFixed` | Fixed calcium concentration and reversal potential. |
| Dynamic calcium concentration | `braincell.ion.CalciumDetailed` | Tracks calcium dynamics and live Nernst reversal potential. |
| Simple first-order calcium | `braincell.ion.CalciumFirstOrder` | Use when simplified calcium concentration dynamics are enough. |

The API also exposes literature/template calcium dynamics such as `CdpHVA_SU2015_DCN`, `CdpLVA_SU2015_DCN`, `CdpStC_MA2020_GoC`, and related template imports. Check the API page before writing custom calcium dynamics.

## Basic Usage

```python
import braincell
import brainunit as u

na = braincell.ion.SodiumFixed(1, E=50. * u.mV)
na.add(INa=braincell.channel.Na_HH1952(1))

k = braincell.ion.PotassiumFixed(1, E=-77. * u.mV)
k.add(IK=braincell.channel.K_HH1952(1))
```

Use `E` for reversal potential and concentration parameters (`C`, `Ci`, `Co`, `C_rest`, `C0`) on the ion object, not on the channel unless a channel-specific API asks for it.

## CalciumDetailed Pattern

Use `CalciumDetailed` when calcium concentration should change with calcium currents and feed back into calcium-dependent channels.

```python
ca = braincell.ion.CalciumDetailed(
    size=1,
    C_rest=5e-5 * u.mM,
    tau=10. * u.ms,
    d=0.5 * u.um,
)
ca.add(ICaT=braincell.channel.CaT_HM1992(1, g_max=2.1 * (u.mS / u.cm ** 2)))
```

The tutorial describes calcium dynamics as driven by calcium current, Faraday constant, shell depth, and return toward resting concentration.

## MixIons

Use `braincell.MixIons` when a channel depends on multiple ion species. The order must match the channel `root_type`.

```python
kca = braincell.MixIons(k, ca)
kca.add(IAHP=braincell.channel.AHP_De1994(1, g_max=0.3 * (u.mS / u.cm ** 2)))
```

For `AHP_De1994`, the tutorial shows `root_type` as potassium then calcium, so pass `(self.k, self.ca)`, not `(self.ca, self.k)`.

## Decision Rules

* Use `SodiumFixed` + sodium channels for classical action-potential depolarization.
* Use `PotassiumFixed` + potassium channels for repolarization, leak, delayed rectifier, A-current, or M-current families.
* Use `CalciumDetailed` when calcium concentration participates in adaptation, rebound, or calcium-activated potassium currents.
* Use `MixIons` only when the channel requires multiple ion dependencies.
* If built-ins do not cover a multicompartment mechanism, return to `skills/braincell/references/multicompartment/multicompartment-cell-workflow.md`; only that parent may select custom authoring.

## Common Mistakes -> Fix

* Treating ion and channel as the same object -> create the ion container, then add channel modules to it.
* Hard-coding reversal potential inside every channel -> put shared `E` on the ion unless the channel API explicitly differs.
* Using fixed calcium when the task depends on adaptation or calcium-dependent current -> use dynamic calcium (`CalciumDetailed` or a documented calcium dynamics template).
* Passing `MixIons` in the wrong order -> inspect `channel.root_type` and match exactly.
* Adding a potassium channel to sodium or calcium -> keep channel root type and ion container aligned.
* Passing bare concentrations, time constants, voltage, or depth values -> attach `brainunit` units.
