---
name: braincell-multicompartment
description: Guides BrainCell morphology-based multicompartment modeling from concepts to API workflows, including branches, regions, locsets, mechanisms, paint/place semantics, CV policies, runtime topology, solvers, probes, clamps, morphology loading, and simulation execution. Use when soma, dendrites, axons, cables, spatial targeting, SWC/ASC/NeuroML2 inputs, or geometry-dependent electrophysiology matters.
---

## Slide 17

### Title

Braincell - multi-compartment

### Activation

activation: for building braincell that geometry matters (soma, dendrite), do not when only...

### Scope and concepts

Scope
Concept
Mechanism systems  summarize sharply from https://brainx.chaobrain.com/braincell/concepts/mechanisms.html
Density side:
CableProperty -> passive default property of a branch
Ion
Channel
Points side:
ClampProbe, state Probe, Mechanism Probe, current Probe

Concept continued
morphology, Branch https://brainx.chaobrain.com/braincell/concepts/morphology.html


paint the Region, place the locset https://brainx.chaobrain.com/braincell/concepts/regions_locsets.html


What is CV -> different CV Policies https://brainx.chaobrain.com/braincell/concepts/discretization.html


What is solver https://brainx.chaobrain.com/braincell/concepts/integration.html

### Minimal scripts example

1° minimal scripts example
paint/place mechanisms in Braincell

#### Painting cable properties

Painting cable properties
Every Cell starts with one default global cable-property rule. A later paint(...) call can locally override those midpoint properties on the CVs that fall inside the painted region.

##### Script

```python
cable_cell = Cell(morpho, cv_policy=CVPerBranch(cv_per_branch=2))
cable_cell.paint(
    BranchSlice(branch_index=0, prox=0.0, dist=1.0),
    CableProperty(
        resting_potential=-70.0 * u.mV,
        membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
        axial_resistivity=200.0 * (u.ohm * u.cm),
        temperature=u.celsius2kelvin(20.0),
    ),
)
```

#### Painting a density mechanism

Painting a density mechanism
A density mechanism is distributed over a region rather than attached to a single point.

In the example below, Channel("IL", ...) is a declarative channel specification. The cell runtime lowers it into a dense layout over the active CV midpoints.

##### Script

```python
channel_cell = Cell(morpho)
channel_cell.paint(
    BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
    Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
)
```

#### place(...): declarative point mechanisms

place(...): declarative point mechanisms
place(...) is the companion API for mechanisms that should live at a single location instead of over a region.

Its general shape is:

##### Script

```python
cell.place(locset, *mechanisms)
```

##### Explanation text

Here:

locset is a location expression such as RootLocation(...)

the remaining arguments are point mechanisms such as CurrentClamp

This is the right API for clamps, probes, and other point-like declarations.

##### Script

```python
place_cell = Cell(morpho, cv_policy=CVPerBranch(2))
place_cell.place(
    RootLocation(x=0.5),
    CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA),
)

place_cell.init_state()
layout = place_cell.layouts[0]
```

#### Build a Cell directly from existing morphology

Build a Cell directly from existing morphology
load a morphology

pass it into Cell(...)

inspect the resulting CV structure

##### Script

```python
morpho = Morphology.from_swc("../../data/morphology/example_tree.swc")

print(morpho.topo())

cell = Cell(morpho)
print(cell)
```

##### Script result

```text
soma
├── axon_0
│   └── axon_1
└── basal_dendrite_0
    ├── basal_dendrite_1
    │   ├── basal_dendrite_2
    │   └── basal_dendrite_3
    ├── basal_dendrite_4
    └── basal_dendrite_5
Cell(root='soma', n_branches=9, n_paint_rules=1, n_place_rules=0, initialized=False)
```

##### Explanation text

->use the script from "Cell in Braincell" tutorial

### Workflow clarification

Workflow clarification
load or build a morphology
choose CV Policy + solver
paint & place regions and points
cell.init_state()
StateProbe()

### Add-ons Mini example

Add-ons Mini example: each within 5 lines of code

#### 1° Import Morphology from files SWC, ASC examples

1° Import Morphology from files SWC, ASC examples

##### Script

```python
Morphology.from_swc(path)
Morphology.from_asc(path)
swc_tree = Morphology.from_swc("../../data/morphology/example_tree.swc")
```

#### 2° Region and Locset Filter

2° Region and Locset Filter  parse and pick smartly from https://brainx.chaobrain.com/braincell/tutorials/filter.html
script in "Region and
Locset Filter"

#### 3° Use existing Ion/Channel + Mixions through painting

3° Use existing Ion/Channel + Mixions through painting
1° add channels
code example

##### Script

```python
na = braincell.ion.SodiumFixed(1)
na.add(ina=braincell.channel.Na_HH1952(1))
```

#### 2° Use existing ions + Using mixions

2° Use existing ions + Using mixions

##### Script

```python
import braincell

class HH(braincell.SingleCompartment):
    def __init__(self, in_size):
        super().__init__(in_size, C=Cm, solver='ind_exp_euler')
        self.na = braincell.ion.SodiumFixed(in_size, E=50. * u.mV)
        self.na.add_elem(
            INa=braincell.channel.Na_TM1991(in_size, g_max=(100. * u.mS * u.cm ** -2) * area, V_sh=-63. * u.mV)
        )

        self.k = braincell.ion.PotassiumFixed(in_size, E=-90 * u.mV)
        self.k.add_elem(
            IK=braincell.channel.K_TM1991(in_size, g_max=(30. * u.mS * u.cm ** -2) * area, V_sh=-63. * u.mV)
        )

        self.IL = braincell.channel.IL(in_size, E=-60. * u.mV, g_max=(5. * u.nS * u.cm ** -2) * area)
```

##### Explanation text

Let’s look at an example of using MixIons in practical modeling:

##### Script

```python
class HTC(braincell.SingleCompartment):
    def __init__(
        self,
        size,
        gKL=0.01 * (u.mS / u.cm ** 2),
        V_initializer=braintools.init.Constant(-65. * u.mV),
        solver: str = 'ind_exp_euler'
    ):
        super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV, solver=solver)

        self.area = 1e-3 / (2.9e-4 * u.cm ** 2)

        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=gKL))
        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV, q10=2.0, temp=u.celsius2kelvin(16.)))

        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
        self.ca.add(ICaL=braincell.channel.CaL_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaN=braincell.channel.CaN_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=0.3 * (u.mS / u.cm ** 2))
```

#### 3.5 CV Policy

3.5 CV Policy

##### Script

```python
# the d-lambda rule at 100 Hz
policy = braincell.DLambda(frequency=100. * u.Hz)

# or: no CV longer than 20 microns
policy = braincell.MaxCVLen(20. * u.um)
```

#### 4° Add cableProperty

4° Add cableProperty

##### Script

```python
cable_cell = Cell(morpho, cv_policy=CVPerBranch(cv_per_branch=2))
cable_cell.paint(
    BranchSlice(branch_index=0, prox=0.0, dist=1.0),
    CableProperty(
        resting_potential=-70.0 * u.mV,
        membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
        axial_resistivity=200.0 * (u.ohm * u.cm),
        temperature=u.celsius2kelvin(20.0),
    ),
)
```

#### 5° Add Clamp

5° Add Clamp

##### Script

```python
from braincell.filter import RootLocation
import braincell.mech as mech
import brainunit as u

# inject a step current at the soma
cell.place(RootLocation(0.5),
           mech.CurrentClamp(delay=10 * u.ms, durations=50 * u.ms, amplitudes=0.2 * u.nA))

# record membrane voltage there
cell.place(RootLocation(0.5), mech.StateProbe("V"))
```

#### 6° Add Probe

6° Add Probe

##### Explanation text

Probes: the main observer surface
Probes are sparse point declarations. They do not allocate their own evolving state; they sample state or current that already exists elsewhere in the initialized runtime.

The three public probe types are:

StateProbe: reads cell-owned state. In the current multi-compartment implementation, that means membrane voltage v.

MechanismProbe: reads a runtime state field from a named mechanism or ion.

CurrentProbe: reads the current of a named mechanism, or the total current of a named ion owner.

##### Script

```python
morpho_multi = build_demo_morphology()
locset = RootLocation(x=0.5) | Terminals()

cell_multi = Cell(morpho_multi)
cell_multi.place(locset, StateProbe())
cell_multi.init_state()

print("locset display names:", morpho_multi.select(locset).display_names)
print("resolved probe keys:", sorted(cell_multi.sample_probes()))
print()
for layout in cell_multi.layouts:
    declaration = cell_multi.runtime.get_layout_mechanism(layout.id)
    print(layout.kind, layout.point_index.tolist(), declaration.name)
```

##### Script result

```text
locset display names: ('soma(0.5)', 'dend(1)', 'axon(1)')
resolved probe keys: ['axon(1)_v', 'dend(1)_v', 'soma(0.5)_v']

state_probe:v:soma(0.5)_v [1] soma(0.5)_v
state_probe:v:dend(1)_v [3] dend(1)_v
state_probe:v:axon(1)_v [5] axon(1)_v
```

#### 7° Visualization example

7° Visualization example
-> then refer to Cell visualization skill

-> script and explanation
in "Region and
Locset Filter"

### Common failure

Common failure -> Fix

## References

Use `references/cell_multicompartment_reference.py` when the user asks for multicompartment BrainCell modeling, morphology-based cells, CV discretization, region-based channel assignment, point clamps/probes, or simulation of a full `Cell`.

- `references/cell_multicompartment_reference.py`
  Source mirrored: https://brainx.chaobrain.com/braincell/tutorials/cell.html
  Purpose: primary reference for turning morphology into a simulation-ready multicompartment `Cell`.
  Covers: `Morphology.from_swc`, `Cell`, CVs, CV policies, `init_state`, `node_tree`, `paint`, `place`, density mechanisms, point mechanisms, `CurrentClamp`, `StateProbe`, and minimal simulation with `run`.

Do not confuse with: `SingleCompartment` examples are useful for point-cell HH/channel workflows, but multicompartment work should start from `Cell(morpho)` and CV-based discretization.

Supporting references:

- `references/libraries/ion-library.md` - built-in ion choices.
- `references/libraries/channel-library.md` - built-in channel and mechanism choices.
- `references/braincell/morphology-io-loading-validation.md` - morphology loading paths.
- `references/libraries/filter-function-library.md` - region and locset filters.
- `references/libraries/solver-library-with-effects.md` - integrator methods and solver effects.
- `references/braincell/braincell-manual-morphology-construction.md` - manual morphology construction.
- `references/braincell/braincell-custom-ion-channel-authoring.md` - custom ion/channel authoring route.
- `references/brainstate-randomness-reproducibility/` - conditional route only for random trials, noise, parameter sweeps, or reproducibility.
