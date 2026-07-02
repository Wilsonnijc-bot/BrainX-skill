---
name: braincell-singlecell
description: Guides BrainCell single-compartment HH-style neuron modeling from concepts to API workflows, including ions, channels, synapses, membrane dynamics, solvers, integration, probes, current clamps, FI curves, ablations, and vectorized point-neuron simulations. Use when geometry does not matter and the user wants conceptual or code guidance for one or many non-morphological BrainCell neurons.
---

## Slide 15

### Title

Braincell - single cell

### Scope

Scope: ion, channel, synapse only  
Doesn't include geometry

### Clarification script

send Clarification sentence to user: it seems like you want a ... model

### Conceptual explanation

Conceptual explanation

what is ion, channel, synapse in Braincell  
what is solver, what is integration  
In Braincell, singlecompartment usually inherit directly from HH  
sodium ion <-> sodium channel

### Common mistakes

Common mistakes

-> how to call braincell api, with scripts

### Mini title

Minimal Single Cell modeling example (directly with code)

### Script

```python
class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='rk4'):
        super().__init__(size, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        self.IL = braincell.channel.IL(
            size,
            E=-54.387 * u.mV,
            g_max=0.03 * (u.mS / u.cm **2)
        )
```

### Explanation text

In this example, we use SingleCompartment to construct a classical HH neuron with sodium (INa), potassium (IK), and leak (IL) currents, which together determine the electrophysiological properties of the neuron.

As this example shows, SingleCompartment serves as a convenient modeling interface in braincell, greatly simplifying the construction of neurons and improving modeling efficiency.

declaration -> integration  
and advancing
summarize key from https://brainx.chaobrain.com/braincell/concepts/integration.html

### Add-ons minimal example

Add-ons minimal example

#### 1° add channels

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

#### 3° Neural network with Brainstate direct code example

3° Neural network with Brainstate direct code example
directly use https://brainx.chaobrain.com/braincell/examples/ei_network.html

### Reference

Reference
weblink: concepts ions & channels&Integrations
ion library.md  summarize from https://brainx.chaobrain.com/braincell/apis/braincell.ion.html
channel library.md  summarize from https://brainx.chaobrain.com/braincell/apis/braincell.channel.html
integration method.md + effect of different solvers.
summarize from https://brainx.chaobrain.com/braincell/apis/integration.html

example script of more complex single cell
choose from html

customize ion channel skill

