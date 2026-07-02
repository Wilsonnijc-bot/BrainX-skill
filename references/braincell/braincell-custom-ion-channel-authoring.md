# BrainCell Custom Ion Channel Authoring

Concepts

• Ion species + channels
The electrical behavior of a neuron comes from ion channels; braincell models ion species as owning reversal potential/concentration dynamics and channels as carrying current driven by that reversal potential.
Source: https://brainx.chaobrain.com/braincell/concepts/ions_channels.html

• Channel
A channel produces a current as a function of voltage and possibly ion concentration; braincell.channel ships a large literature-derived library named by <current>_<source><year>.
Source: https://brainx.chaobrain.com/braincell/concepts/ions_channels.html

• Existing vs custom channels
For Channel, braincell supports two ways of use: defining new ion channels, or using existing ones.
Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html

• Custom channel base class
IonChannel provides current, init_state, reset_state, compute_derivative, pre_integral, and post_integral; for practical channel models the docs recommend using Channel.
Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html

• root_type
root_type tells braincell which ion the channel acts on, and ensures the right IonInfo is passed into current / compute_derivative.
Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html

• Registry
Channels and ions self-register via @register_channel / @register_ion; the registry lets string declarations resolve names like mech.Channel("Na_Ba2002", ...).
Source: https://brainx.chaobrain.com/braincell/concepts/ions_channels.html

Evoke case / task boundary

• use this reference when the requested biological current is not covered by a pre-built braincell.channel model.
• use this reference when the task asks for a reusable custom ion-channel component: gates, parameters, state variables, derivatives, current.
• do not use this reference when a named pre-built channel already matches the task; use the library first.
• do not turn channel authoring into generic ODE code; keep it inside braincell.Channel, HH, or Markov patterns.
• do not ignore root_type; sodium channels go on sodium ions, potassium channels go on potassium ions, etc.

Minimal custom HH channel pattern

Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html

Script

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import brainunit as u
import jax.numpy as jnp
import braincell
from braincell import IonInfo
from braincell.channel import NaF_SU2015_DCN, Na_HH1952
from braincell.channel._base import Gate, HH, Markov, Transition
print("braincell version:", braincell.__version__)
class DemoKChannel(HH):
    root_type = braincell.ion.Potassium
    gates = (Gate("n", power=4, phi=2.0),)
    def __init__(self, size, g_max=1.0 * (u.mS / u.cm**2)):
        super().__init__(size=size)
        self.g_max = g_max
    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)
    def f_n_inf(self, V, K: IonInfo):
        V_mV = V.to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V_mV + 35.0) / 10.0))
    def f_n_tau(self, V, K: IonInfo):
        return 5.0
V = jnp.array([-55.0]) * u.mV
k = IonInfo(
    Ci=jnp.array([54.4]) * u.mM,
    Co=jnp.array([2.5]) * u.mM,
    E=jnp.array([-90.0]) * u.mV,
    valence=1,
)
demo = DemoKChannel(size=1)
demo.init_state(V, k)
demo.n.value = jnp.array([0.2])
demo.compute_derivative(V, k)
print("n_inf:", demo.f_n_inf(V, k))
print("dn/dt:", demo.n.derivative.to_decimal(u.Hz), "Hz")
print("current:", demo.current(V, k).to_decimal(u.mA / u.cm**2), "mA/cm^2")

Explanation text

If a new channel can be written in HH-gate form, the docs say to inherit from HH, declare root_type, declare gates, implement each gate’s inf/tau or alpha/beta, then implement current().
Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html

Lower-level channel interface skeleton

Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html

Script

class CalciumChannel(braincell.Channel):
    # Specify the ion type this channel acts on
    root_type = braincell.ion.Calcium
    def pre_integral(self, V, Ca: braincell.IonInfo):
        pass
    def post_integral(self, V, Ca: braincell.IonInfo):
        pass
    def compute_derivative(self, V, Ca: braincell.IonInfo):
        pass
    def current(self, V, Ca: braincell.IonInfo):
        raise NotImplementedError
    def init_state(self, V, Ca: braincell.IonInfo, batch_size=None):
        pass
    def reset_state(self, V, Ca: braincell.IonInfo, batch_size=None):
        pass

Explanation text

Use this only when the HH / Markov templates are not enough; Channel defines the minimal interfaces every channel must implement.
Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html

Registering a custom channel name

Source: https://brainx.chaobrain.com/braincell/developer/extending.html

Script

import braincell
from braincell.mech import register_channel
@register_channel("MyNa")
class MyNa(braincell.Channel):
    # define states, derivatives, and current here
    ...
import braincell.mech as mech
mech.Channel("MyNa", g_max=0.1 * u.S / u.cm**2)

Explanation text

Concrete channels self-register at import time with @register_channel; once imported, the registered name is what string-based declarations resolve to.
Source: https://brainx.chaobrain.com/braincell/developer/extending.html

Full bundled script references

custom-hh-channel-demo.py

Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html
Purpose: complete runnable example of a minimal custom HH potassium channel.

manual-icat-hp1992-authoring.py

Source: https://brainx.chaobrain.com/braincell/tutorials/channel.html
Purpose: full direct Channel subclass example with parameters, gates, DiffEqState, derivatives, and current.

registered-channel-extension.py

Source: https://brainx.chaobrain.com/braincell/developer/extending.html
Purpose: complete example of making a custom channel available by string name.

Common mistakes -> Fix

• authoring new channel before checking library -> check braincell.channel / Ion Channel Collection first.
• missing root_type -> declare the ion class the channel binds to.
• adding a potassium channel to a sodium ion -> add channels only to matching ion roots; the docs show this raises a type error.
• writing pure ODE functions with raw floats -> implement compute_derivative, init_state, reset_state, and current on Channel / HH.
• stripping units early -> keep brainunit quantities; convert only at formula boundaries with to_decimal(...).
