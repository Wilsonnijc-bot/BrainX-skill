---
name: brainx-general-guard
description: Prevents cross-cutting BrainX misuse and infrastructure mistakes, including wrong package selection, bypassed units, raw JAX transforms on stateful code, and unnecessary custom NumPy/JAX simulations. Use when a request needs BrainX domain routing, safety guardrails, common mistake correction, or confirmation that BrainX APIs should be used.
---

brainx-general-guard/

Concepts

• BrainState mental model
Three nouns carry the whole framework: State, Module, and Transform.
Source: https://brainx.chaobrain.com/brainstate/getting_started/thinking_in_brainstate.html

• State-aware transforms
BrainState transformations mirror JAX’s, but understand State; reads and writes are threaded through compiled functions automatically.
Source: https://brainx.chaobrain.com/brainstate/getting_started/thinking_in_brainstate.html

• BrainUnit / BrainCell unit guard
In braincell, every physical quantity carries an explicit unit; passing a bare number where a quantity is expected raises TypeError.
Source: https://brainx.chaobrain.com/braincell/concepts/units.html

• BrainCell modeling guard
A mechanism is anything installed on a cell that affects dynamics; in braincell, mechanisms are declarative and describe what to install without touching JAX, time, or runtime state.
Source: https://brainx.chaobrain.com/braincell/concepts/mechanisms.html

Evoke case / task boundary

• use this skill before writing BrainX / BrainState / BrainCell / BrainUnit code.
• use this skill when choosing package boundaries, transforms, units, biological components, or install path.
• do not use this as a tutorial; it is a guardrail checklist.
• do not invent BrainX APIs if official examples exist.
• do not write raw NumPy/JAX simulation code when BrainState / BrainCell / BrainUnit provides the intended abstraction.
• do not author custom biological components before checking pre-built components.

Domain routing guard

Script

[NEEDS OFFICIAL SCRIPT SOURCE]

Explanation text

Route before coding: BrainState for State / Module / Transform, BrainCell for cells / mechanisms / channels / ions, BrainUnit for physical quantities and dimensional safety, BrainMass only when the official task is population / whole-brain mass modeling.

Use BrainState JIT for stateful modules

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html

Script

import jax
import jax.numpy as jnp
import brainstate
class RunningMean(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.sum = brainstate.HiddenState(jnp.array(0.0))
        self.count = brainstate.HiddenState(jnp.array(0))
    def __call__(self, batch: jax.Array) -> jax.Array:
        self.sum.value += jnp.sum(batch)
        self.count.value += batch.size
        return self.sum.value / self.count.value
tracker = RunningMean()
@brainstate.transform.jit
def update_running_mean(batch: jax.Array) -> jax.Array:
    return tracker(batch)
for step in range(3):
    data = jnp.arange(4.0) + step
    print(f'step {step}: mean={float(update_running_mean(data)):.2f}')
float(tracker.sum.value), int(tracker.count.value)

Explanation text

brainstate.transform.jit understands State objects and automatically wires read/write traces into the compiled function; raw jax.jit requires explicitly splitting and merging module state.
Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html

Use BrainState loop transforms for stateful loops

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html

Script

# Example 2: Stateful for_loop
class Accumulator(brainstate.nn.Module):
    """Simple accumulator that tracks total and count."""
    def __init__(self):
        super().__init__()
        self.total = brainstate.ShortTermState(jnp.array(0.0))
        self.count = brainstate.ShortTermState(jnp.array(0))
    def process(self, x):
        self.total.value = self.total.value + x
        self.count.value = self.count.value + 1
        return self.total.value / self.count.value  # running average
acc = Accumulator()
data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
running_averages = for_loop(acc.process, data)
print("Data:", data)
print("Running averages:", running_averages)
print(f"\n Final state: total={acc.total.value}, count={acc.count.value}")
print(f"Final average: {acc.total.value / acc.count.value}")

Explanation text

Control-flow APIs in brainstate.transform provide JAX-compatible loops and conditionals while safely handling State objects; loop transformations compile to a single JAX primitive, reducing compilation overhead.
Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html

Use units before biological simulation

Source: https://brainx.chaobrain.com/braincell/concepts/units.html

Script

import numpy as np
import jax.numpy as jnp
import brainunit as u
v_rest = -65.0 * u.mV                 # scalar
dt     = 0.1 * u.ms
lengths = np.array([10.5, 20.0]) * u.um   # numpy array
coords  = jnp.zeros((10, 3)) * u.um       # JAX array
radii   = [2.0, 3.0, 4.0] * u.um          # list

Explanation text

Silent unit mismatches are one of the most common and hardest-to-find bugs in neural modeling; BrainCell requires units everywhere and catches dimensional errors immediately.
Source: https://brainx.chaobrain.com/braincell/concepts/units.html

Full bundled script references

stateful-jit-guard-example.py

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html
Purpose: canonical example of stateful BrainState module compiled with brainstate.transform.jit.

stateful-for-loop-guard-example.py

Source: https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html
Purpose: canonical state-aware for_loop pattern.

unit-required-braincell-example.py

Source: https://brainx.chaobrain.com/braincell/concepts/units.html
Purpose: minimal official quantity creation pattern for BrainCell model parameters.

Common mistakes -> Fix

• writing random NumPy simulation code -> route first to BrainState / BrainCell / BrainUnit.
• using raw jax.jit on stateful BrainState modules -> use brainstate.transform.jit; raw jax.jit requires manual state threading.
• Python for-loop around state updates when compiling/scanning -> use BrainState scan, for_loop, or control-flow transform.
• passing bare floats into BrainCell mechanisms -> attach brainunit units.
• treating BrainCell as generic ODE code -> use cells, ions, channels, mechanisms, and integrators.
• writing custom ion channel first -> check built-in braincell.channel list first.
• inventing APIs -> open official example/source first; leave [NEEDS OFFICIAL SCRIPT SOURCE] if no official pattern exists.

