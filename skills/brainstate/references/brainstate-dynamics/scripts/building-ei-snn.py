"""BrainState E/I spiking-network simulation reference.

Purpose: complete E/I spiking-network simulation workflow with
`init_all_states` and compiled trajectory.

Source mirrored:
https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html
"""

import brainunit as u
import brainpy
import brainstate
import braintools
import jax.numpy as jnp
import matplotlib.pyplot as plt


brainstate.random.seed(0)
brainstate.environ.set(dt=0.1 * u.ms)


class EINet(brainstate.nn.Module):
    """Balanced excitatory/inhibitory LIF network."""

    def __init__(self, n_exc, n_inh, prob, JE, JI):
        super().__init__()
        self.n_exc = n_exc
        self.num = n_exc + n_inh
        self.N = brainpy.state.LIF(
            self.num,
            V_rest=-52.0 * u.mV,
            V_th=-50.0 * u.mV,
            V_reset=-60.0 * u.mV,
            tau=10.0 * u.ms,
            V_initializer=braintools.init.Normal(-60.0, 10.0, unit=u.mV),
            spk_reset="soft",
        )
        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(n_exc, self.num, prob, JE),
            syn=brainpy.state.Expon.desc(self.num, tau=2.0 * u.ms),
            out=brainpy.state.CUBA.desc(),
            post=self.N,
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(n_inh, self.num, prob, JI),
            syn=brainpy.state.Expon.desc(self.num, tau=2.0 * u.ms),
            out=brainpy.state.CUBA.desc(),
            post=self.N,
        )

    def update(self, inp):
        spikes = self.N.get_spike() != 0.0
        self.E(spikes[: self.n_exc])
        self.I(spikes[self.n_exc :])
        self.N(inp)
        return self.N.get_spike()


def main():
    n_exc, n_inh, prob = 400, 100, 0.1
    JE = 1.0 / u.math.sqrt(prob * n_exc) * u.mS
    JI = -1.0 / u.math.sqrt(prob * n_inh) * u.mS

    net = EINet(n_exc, n_inh, prob, JE, JI)
    brainstate.nn.init_all_states(net)

    times = u.math.arange(0.0 * u.ms, 200.0 * u.ms, brainstate.environ.get_dt())
    spikes = brainstate.transform.for_loop(lambda t: net.update(3.0 * u.mA), times)

    print("spike array shape (time, neurons):", spikes.shape)
    print("total spikes:", int(jnp.sum(spikes != 0)))

    t_ms = times.to_decimal(u.ms)
    t_idx, n_idx = u.math.where(spikes != 0)
    plt.figure(figsize=(8, 4))
    plt.plot(t_ms[t_idx], n_idx, "k.", markersize=1)
    plt.xlabel("time (ms)")
    plt.ylabel("neuron index")
    plt.title("E/I balanced network raster")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
