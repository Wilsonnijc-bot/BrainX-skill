"""BrainState Hodgkin-Huxley neuron reference.

Purpose: complete executable HH neuron example showing biophysical state
variables and continuous-time dynamics.

Source mirrored:
https://brainx.chaobrain.com/brainstate/examples/brain_dynamics/hodgkin_huxley_neuron.html
"""

import brainunit as u
import brainstate
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


brainstate.random.seed(42)


class HH(brainstate.nn.Dynamics):
    """Hodgkin-Huxley neuron model."""

    def __init__(
        self,
        in_size,
        ENa=50.0 * u.mV,
        gNa=120.0 * u.mS / u.cm**2,
        EK=-77.0 * u.mV,
        gK=36.0 * u.mS / u.cm**2,
        EL=-54.387 * u.mV,
        gL=0.03 * u.mS / u.cm**2,
        V_th=20.0 * u.mV,
        C=1.0 * u.uF / u.cm**2,
    ):
        super().__init__(in_size)
        self.ENa = ENa
        self.EK = EK
        self.EL = EL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.C = C
        self.V_th = V_th

    def m_alpha(self, V):
        return 1.0 / u.math.exprel(-(V / u.mV + 40.0) / 10.0)

    def m_beta(self, V):
        return 4.0 * jnp.exp(-(V / u.mV + 65.0) / 18.0)

    def m_inf(self, V):
        return self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))

    def dm(self, m, t, V):
        return (self.m_alpha(V) * (1.0 - m) - self.m_beta(V) * m) / u.ms

    def h_alpha(self, V):
        return 0.07 * jnp.exp(-(V / u.mV + 65.0) / 20.0)

    def h_beta(self, V):
        return 1.0 / (1.0 + jnp.exp(-(V / u.mV + 35.0) / 10.0))

    def h_inf(self, V):
        return self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))

    def dh(self, h, t, V):
        return (self.h_alpha(V) * (1.0 - h) - self.h_beta(V) * h) / u.ms

    def n_alpha(self, V):
        return 0.1 / u.math.exprel(-(V / u.mV + 55.0) / 10.0)

    def n_beta(self, V):
        return 0.125 * jnp.exp(-(V / u.mV + 65.0) / 80.0)

    def n_inf(self, V):
        return self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))

    def dn(self, n, t, V):
        return (self.n_alpha(V) * (1.0 - n) - self.n_beta(V) * n) / u.ms

    def init_state(self, batch_size=None):
        self.V = brainstate.HiddenState(
            jnp.ones(self.varshape, brainstate.environ.dftype()) * -65.0 * u.mV
        )
        self.m = brainstate.HiddenState(self.m_inf(self.V.value))
        self.h = brainstate.HiddenState(self.h_inf(self.V.value))
        self.n = brainstate.HiddenState(self.n_inf(self.V.value))

    def dV(self, V, t, m, h, n, I):
        I_Na = (self.gNa * m * m * m * h) * (V - self.ENa)
        I_K = (self.gK * n * n * n * n) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        return (-I_Na - I_K - I_leak + I) / self.C

    def update(self, x=0.0 * u.mA / u.cm**2):
        t = brainstate.environ.get("t")
        V = brainstate.nn.exp_euler_step(
            self.dV, self.V.value, t, self.m.value, self.h.value, self.n.value, x
        )
        m = brainstate.nn.exp_euler_step(self.dm, self.m.value, t, self.V.value)
        h = brainstate.nn.exp_euler_step(self.dh, self.h.value, t, self.V.value)
        n = brainstate.nn.exp_euler_step(self.dn, self.n.value, t, self.V.value)
        spike = jnp.logical_and(self.V.value < self.V_th, V >= self.V_th)
        self.V.value = V
        self.m.value = m
        self.h.value = h
        self.n.value = n
        return spike


def main():
    hh = HH(10)
    brainstate.nn.init_all_states(hh)

    dt = 0.01 * u.ms
    brainstate.environ.set(dt=dt)

    def run(t, inp):
        with brainstate.environ.context(t=t, dt=dt):
            hh(inp)
        return hh.V.value

    duration = 100.0 * u.ms
    times = u.math.arange(0.0 * u.ms, duration, dt)
    inputs = brainstate.random.uniform(1.0, 10.0, times.shape) * u.uA / u.cm**2
    vs = brainstate.transform.for_loop(run, times, inputs)

    times_ms = times.to_decimal(u.ms)
    vs_mv = vs.to_decimal(u.mV)

    threshold = 20.0
    spike_counts = []
    spike_times = []
    for i in range(hh.varshape[0]):
        above = vs_mv[:, i] > threshold
        indices = np.where(np.diff(above.astype(int)) > 0)[0]
        spike_times.append(times_ms[indices])
        spike_counts.append(len(indices))

    print("Spike counts per neuron:", spike_counts)
    print(
        "Average firing rate:",
        np.mean(spike_counts) / (duration.to_decimal(u.ms) / 1000.0),
        "Hz",
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i in range(3):
        axes[i].plot(times_ms, vs_mv[:, i], linewidth=1.5)
        axes[i].set_ylabel(f"V{i} (mV)")
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=threshold, color="r", linestyle="--", alpha=0.5)
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
