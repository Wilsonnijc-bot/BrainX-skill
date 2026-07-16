---
name: brainmass
description: "Use when working with BrainMass neural-mass workflows: one-model simulation, stochastic runs, batched ensembles, delay-coupled networks, BOLD/EEG/MEG forward models, parameter fitting, model catalogs, or whole-brain neural-mass examples. Routes non-differentiable fitting tasks to a standalone gradient-free reference script."
---

# BrainMass Core Workflows

## Purpose And Boundary

Use this skill when the user asks for BrainMass neural-mass workflows: one-model simulation, stochastic runs, batched ensembles, delay-coupled networks, forward models to observable BOLD / EEG / MEG signals, or parameter fitting. The body teaches the canonical differentiable fitting path; gradient-free fitting is a routed reference workflow, not a second inline fitting tutorial.

Do not use this as a BrainState module-building skill. BrainMass "ships only models and delegates everything else to the rest of the BrainX ecosystem."
Source: https://brainx.chaobrain.com/brainmass/concepts/architecture_overview.html

## P0 Concepts

- **models-only, delegate the rest**
  "The single sentence to remember: brainmass ships only models and delegates everything else to the rest of the BrainX ecosystem."
  Source: https://brainx.chaobrain.com/brainmass/concepts/architecture_overview.html

- **core workflow chain**
  "A model describes one region's dynamics, a Simulator runs it, a Network couples many regions, and a Fitter tunes parameters to data. Units thread through all of them."
  Source: https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html

- **Step model contract**
  "Every neural-mass model is a *Step class implementing one update step of its differential equations."
  Source: https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html

- **model discovery**
  "brainmass.list_models() enumerates every model with its category and number of state variables."
  Source: https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html

- **noise belongs to the model**
  "Noise is a property of the model, not the simulator."
  Source: https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html

- **seed before noisy reports**
  "Always seed before a noisy run you intend to report."
  Source: https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html

- **batched ensembles**
  "batch_size=N runs an ensemble of independent trials in parallel; the output gains a batch axis."
  Source: https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html

- **Network**
  "A Network turns a single *Step node sized for N regions into a coupled whole-brain model."
  Source: https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html

- **forward / observation layer**
  "A forward model (or observation model) is the biophysical map from that hidden activity to a measurable neuroimaging signal."
  Source: https://brainx.chaobrain.com/brainmass/tutorials/05_forward_models.html

- **Fitter + objectives**
  "The Fitter tunes a model's trainable parameters to data behind one .fit() call."
  Source: https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html

- **backend switch**
  "One backend= switch chooses the optimiser." Keep `grad` as the canonical body path. For `nevergrad` or derivative-free `scipy`, open `references/scripts/gradient-free-fitting.py` instead of reconstructing a backend swap from memory.
  Source: https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html

- **gradient-free boundary**
  "Not every fitting problem has a usable gradient." Use `nevergrad` / derivative-free `scipy` only when the gradient is unavailable or unreliable, and follow the standalone reference script because gradient-free parameters need a bounded search space.
  Source: https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html

## Canonical Setup

Source: https://brainx.chaobrain.com/brainmass/getting_started/quickstart.html

```python
import brainmass
import braintools
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.nn import Param

# `dt` is a global, set once through the environment. The Simulator also takes an
# explicit dt= below; setting it here lets delay-buffer sizing work when we build a
# Network (its conduction delays are measured in dt-steps at construction time).
brainstate.environ.set(dt=0.1 * u.ms)
```

## Model Discovery

Keep this brief in the body. Use it when the user asks "what model should I use?" or when the skill needs to inspect model categories before choosing a gallery / API reference.

Source: https://brainx.chaobrain.com/brainmass/reference/utilities.html

```pycon
>>> import brainmass
>>> models = brainmass.list_models()
>>> {m.name for m in models} >= {'HopfStep', 'JansenRitStep'}
True
>>> next(m for m in models if m.name == 'JansenRitStep').category
'physiological'
>>> print(brainmass.list_models.to_table())
name...category...#states...use_case...
```

## One Model Simulation

A brainmass model is a *Step object. One .run(duration, monitors=...) call sets dt, initialises state, steps the model, and returns trajectories as a dict plus 'ts'.

Source: https://brainx.chaobrain.com/brainmass/getting_started/quickstart.html

```python
node = brainmass.HopfStep(in_size=1, a=0.25, w=0.3)
sim = brainmass.Simulator(node, dt=0.1 * u.ms)
res = sim.run(200.0 * u.ms, monitors=["x", "y"], transient=20.0 * u.ms)
print("recorded keys:", list(res))
print("x trajectory shape (steps, regions):", res["x"].shape)
print("time axis:", res["ts"][:3], "...", res["ts"][-1])
```

Official output:

```text
recorded keys: ['x', 'y', 'ts']
x trajectory shape (steps, regions): (1800, 1)
time axis: [20.1 20.2 20.3] ms ... 200. ms
```

## Noise + Random Seed Basics

Attach noise when constructing the model. Seed before a noisy run that should be reproducible. Same seed gives the same trajectory; different seed gives a different one.

Source: https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html

```python
def noisy_run(seed):
    brainstate.random.seed(seed)
    node = brainmass.HopfStep(
        in_size=1, a=-0.05, w=0.3,
        noise_x=brainmass.OUProcess(in_size=1, sigma=0.1, tau=20.0 * u.ms),
    )
    return brainmass.Simulator(node, dt=0.1 * u.ms).run(100.0 * u.ms, monitors=["x"])["x"]

a1 = noisy_run(7)
a2 = noisy_run(7)   # same seed
b = noisy_run(8)    # different seed
print("same seed -> identical trajectory: ", bool(jnp.allclose(a1, a2)))
print("different seed -> different trajectory:", not bool(jnp.allclose(a1, b)))
```

Official output:

```text
same seed -> identical trajectory:  True
different seed -> different trajectory: True
```

## Batching / Transform Basics

Do not teach BrainState transforms here. Just remember the primitive map and show the Simulator.run(..., batch_size=...) route first.

Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

Shape of work -> primitive:

- single step / one-shot call -> brainstate.transform.jit
- many steps, collect outputs -> brainstate.transform.for_loop
- many steps with explicit carry -> brainstate.transform.scan
- batch over inputs / parameters -> brainstate.transform.vmap
- long rollout under autograd -> checkpointed_for_loop / checkpointed_scan

```python
node = brainmass.HopfStep(
    in_size=4, a=0.1, w=0.3,
    noise_x=brainmass.OUProcess(4, sigma=0.1, tau=10 * u.ms),
)
brainstate.random.seed(0)
res = brainmass.Simulator(node, dt=0.1 * u.ms).run(
    200 * u.ms, monitors=['x'], batch_size=16,
)
print("batched trajectory:", res['x'].shape, "(time, batch, regions)")
```

Official output:

```text
batched trajectory: (2000, 16, 4) (time, batch, regions)
```

## Small Network

The network zeros self-connections, turns distance / speed into conduction delays, and feeds a diffusive coupling current back into the node. The Simulator drives the whole network like the single node.

Source: https://brainx.chaobrain.com/brainmass/getting_started/quickstart.html

```python
conn = brainmass.datasets.load_dataset("example_connectome")
print("connectome:", conn.weights.shape, "regions:", list(conn.labels))
# Use the first two regions as a minimal 2-node network.
W = conn.weights[:2, :2]
D = conn.distances[:2, :2]
two_nodes = brainmass.HopfStep(in_size=2, a=0.2, w=0.3)
net = brainmass.Network(
    two_nodes,
    conn=W,
    distance=D,
    speed=10.0 * u.mm / u.ms,   # delays = distance / speed
    coupling="diffusive",
    coupled_var="x",
    k=0.5,                       # global coupling strength
)
res_net = brainmass.Simulator(net, dt=0.1 * u.ms).run(
    200.0 * u.ms,
    monitors=lambda m: m.node.x.value,   # read the coupled node state each step
    transient=20.0 * u.ms,
)
print("network output shape (steps, regions):", res_net["output"].shape)
```

Official output excerpt:

```text
network output shape (steps, regions): (1800, 2)
```

## Forward Models

Hidden dynamics are not the experimental measurement. Body should show one canonical BOLD path only: source network -> HRFBold. Put BOLDSignal, EEG, MEG, spectra, and validation variants in reference.

Source: https://brainx.chaobrain.com/brainmass/tutorials/05_forward_models.html

```python
# A delay-coupled Network sizes its delay buffer from the global dt at construction,
# so set dt once here, before building any Network.
brainstate.environ.set(dt=1.0 * u.ms)
brainstate.random.seed(0)
conn = brainmass.datasets.load_dataset('example_connectome')
N = conn.weights.shape[0]
labels = list(conn.labels)
print(f"connectome: {N} regions, labels = {labels}")
node = brainmass.WilsonCowanStep(
    in_size=N,
    noise_E=brainmass.OUProcess(N, sigma=0.4, tau=20.0 * u.ms),
)
net = brainmass.Network(
    node,
    conn=conn.weights,
    distance=conn.distances,
    speed=10.0 * u.mm / u.ms,
    coupling='diffusive',
    coupled_var='rE',
    k=0.5,
)
# Run the source network; monitor each region's excitatory rate rE.
src = brainmass.Simulator(net, dt=1.0 * u.ms).run(
    6000.0 * u.ms,
    monitors=lambda m: m.node.rE.value,
    transient=500.0 * u.ms,
)
neural = src['output']          # (T, N) excitatory rate (dimensionless WilsonCowan rate)
ts = src['ts']
print("neural source:", neural.shape, "| unit:", u.get_unit(neural))
hrf = brainmass.HRFBold(
    period=720.0 * u.ms,            # output TR (~0.72 s, a fast multiband protocol)
    downsample_period=20.0 * u.ms,  # internal convolution step
    kernel=brainmass.GammaHRFKernel(),
)
bold_conv = hrf(u.get_magnitude(neural), dt=1.0 * u.ms)   # (T_bold, N), dimensionless
print("HRFBold BOLD:", bold_conv.shape)
```

Official output excerpts:

```text
neural source: (5500, 8) | unit: 1
HRFBold BOLD: (7, 8)
```

## Fitting With Gradients

Use Quickstart fitting as the body path. It loads example_signal, extracts a target RMS amplitude, marks one Hopf parameter trainable with Param(..., fit=True), and fits with Fitter.

Open `references/braintools-optimizer.md` through the fitting parent when choosing an optimizer, learning-rate scheduler, or SciPy/Nevergrad wrapper beyond this canonical `Fitter` example; keep BrainMass-specific objectives and backend selection in this skill.

Source: https://brainx.chaobrain.com/brainmass/getting_started/quickstart.html

```python
# Target: the RMS amplitude of region 0 of the bundled example signal.
sig = brainmass.datasets.load_dataset("example_signal")
target0 = sig.signal[:, 0]
target_amp = jnp.asarray(float(np.sqrt(np.mean((target0 - target0.mean()) ** 2))))
print("target amplitude from example_signal:", round(float(target_amp), 4))

# Model with ONE trainable parameter: the Hopf bifurcation parameter `a`.
model = brainmass.HopfStep(
    in_size=1,
    a=Param(0.05, fit=True),   # <- the single knob the Fitter will tune
    w=0.3, beta=1.0,
    init_x=braintools.init.Constant(0.5),
)

def loss_fn(m):
    """Run the model and compare its settled amplitude to the target."""
    x = brainmass.Simulator(m, dt=0.1 * u.ms).run(
        300.0 * u.ms, monitors=["x"], transient=150.0 * u.ms
    )["x"][:, 0]
    amp = jnp.sqrt(jnp.mean((x - jnp.mean(x)) ** 2))
    return (amp - target_amp) ** 2, amp

fitter = brainmass.Fitter(model, braintools.optim.Adam(lr=0.05), loss_fn=loss_fn)
result = fitter.fit(n_steps=50)
print(result)
print(f"a:        0.05  ->  {float(result.best_params['a']):.4f}")
print(f"loss:  {result.history[0]:.4f}  ->  {result.best_loss:.5f}")
print(f"amplitude matched: {float(result.prediction):.4f}  (target {float(target_amp):.4f})")
```

Official output:

```text
target amplitude from example_signal: 0.7152
FitResult(backend='grad', best_loss=3.10454e-07, n_steps=50, params=[a])
a:        0.05  ->  1.0262
loss:  0.3042  ->  0.00000
amplitude matched: 0.7157  (target 0.7152)
```

### Boundary: When Gradient Fitting Ends

Stop using the inline workflow when the objective is non-differentiable, discrete, jagged, black-box, or produces an unavailable or unreliable gradient. Do not merely replace `backend='grad'`: gradient-free fitting also needs a bounded search space and a different optimizer-argument form.

Open and adapt `references/scripts/gradient-free-fitting.py` for the complete, runnable version of the same Hopf amplitude-recovery problem with both `backend='nevergrad'` and derivative-free `backend='scipy'`. Preserve its common objective, `SigmoidT`-derived bounds, and backend-specific step semantics. Then open the BrainMass-local `references/braintools-optimizer.md` through `references/fitting-with-objectives-api.md` only if the task requires choosing a different SciPy / Nevergrad method or configuring its wrapper.

Decision rule: "if you can differentiate the objective, use `grad`." Reach for gradient-free search only when the gradient is unavailable or unreliable.
Source: https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html

## Important Performance Concepts

### P0 execution and performance rules

- **Time-series execution**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  Never drive a repeatedly executed model with a bare Python `for` / `while` loop; `brainstate.transform` primitives lower the whole loop into one compiled XLA program and trace the body only once.

- **jit**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  Use `brainstate.transform.jit` for a single step or one-shot call: it compiles the step once, then subsequent calls reuse the compiled program. For many-step rollouts, do not call a jitted step from a Python loop; use `for_loop` or `scan` so the whole rollout is lowered.

- **for_loop**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  Use `brainstate.transform.for_loop` for many steps when the model's `State` carries hidden variables automatically and the per-step outputs should be stacked.

```python
node = brainmass.HopfStep(in_size=64, a=0.25, w=0.3)
node.init_all_states()

def step(i):
    with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
        node.update()
    return node.x.value

run = brainstate.transform.jit(lambda: brainstate.transform.for_loop(step, jnp.arange(n)))
xs = run(); jax.block_until_ready(xs)  # warm up

t0 = time.perf_counter()
xs = run(); jax.block_until_ready(xs)
t_forloop = time.perf_counter() - t0
```

Use this as the default BrainMass rollout pattern: define one simulation step, set `t` / `dt` through `brainstate.environ.context`, return monitored values, and let `for_loop` stack the time axis.

- **scan**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  Use `brainstate.transform.scan` when the rollout needs an explicit carry alongside model `State`, with body shape `f(carry, x) -> (carry, y)`.

```python
node = brainmass.HopfStep(in_size=8, a=0.25, w=0.3)
node.init_all_states()

drive = 0.05 * jnp.sin(2 * jnp.pi * jnp.arange(n) / n)[:, None]  # (time, 1)

def body(carry, inp):
    running_sum = carry
    with brainstate.environ.context(t=0. * u.ms):
        node.update(inp)          # external drive into x
    x = node.x.value
    return running_sum + x, x     # (new carry, per-step output)

total, xs = brainstate.transform.scan(body, jnp.zeros(8), drive)
```

Use `scan` when the time-series loop needs a value carried outside the model's own `State`, for example a running statistic, task memory, curriculum variable, external drive state, or custom loss accumulator.

- **Checkpointed loop variants**
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html
  For long rollouts under reverse-mode gradients, swap `for_loop` / `scan` for `checkpointed_for_loop` / `checkpointed_scan` only when BPTT memory would otherwise be exhausted.

- **Vmap batching**
  Vmap-based batching is recommended: compile the computation graph for a single sample, then use `vmap` to automatically vectorize across the batch dimension.
  Source: https://brainx.chaobrain.com/braintrace/tutorials/batching.html

## References

### Reference Selection Rule

Do not keep key-concepts.md as a separate reference by default. The skill body already absorbs its P0 concepts. Keep references for API inventory and workflow variants only.

The canonical architecture is ten BrainMass package references plus four reusable Braintools references:

```text
references/
├── modellibrary.md
├── noiseprocesses.md
├── coupling-network-api.md
├── forward-observation-api.md
├── fitting-with-objectives-api.md
│   ├── braintools-metrics.md
│   ├── braintools-optimizer.md
│   └── braintools-surrogate-gradient.md
├── datasets-api.md
├── visualization-analysis-api.md
├── batch-transform-acceleration.md
├── horn-task-training.md
│   ├── braintools-cognitive-tasks.md
│   ├── braintools-metrics.md
│   └── braintools-optimizer.md
└── parameter-sweeps-and-regime-analysis.md
```

Treat the indented Braintools entries as nested routes, not additional top-level BrainMass branches. Enter `braintools-surrogate-gradient.md` only through `fitting-with-objectives-api.md`; enter `braintools-cognitive-tasks.md` only through `horn-task-training.md`. Select `braintools-metrics.md` or `braintools-optimizer.md` only after choosing the fitting or HORN parent that owns the workflow.

Only `braintools-optimizer.md` currently exists in this draft. The other canonical reference targets remain planned; do not infer that a routed file is already bundled.

### modellibrary.md

Sources:

- https://brainx.chaobrain.com/brainmass/reference/models.html
- https://brainx.chaobrain.com/brainmass/reference/utilities.html

Purpose: exact model inventory, categories, state-variable counts, typical use cases, and brainmass.list_models() / ModelInfo; body only shows the smallest discovery snippet. The models API divides models into phenomenological and physiological categories and includes the model comparison table.

### noiseprocesses.md

Sources:

- https://brainx.chaobrain.com/brainmass/reference/noise.html
- https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html

Purpose: exact noise process inventory: Gaussian, white, OU, Brownian, colored, pink, blue, violet; body only shows OU plus seeding and batch basics. The noise tutorial is also the reference for reproducible stochastic runs and batch_size ensembles.

### coupling-network-api.md

Sources:

- https://brainx.chaobrain.com/brainmass/reference/coupling.html
- https://brainx.chaobrain.com/brainmass/tutorials/04_building_a_network.html
- https://brainx.chaobrain.com/brainmass/concepts/coupling_and_delays.html

Purpose: exact coupling mechanisms and delay-coupled network variants; body only shows built-in diffusive coupling from Quickstart.

### forward-observation-api.md

Sources:

- https://brainx.chaobrain.com/brainmass/reference/forward.html
- https://brainx.chaobrain.com/brainmass/reference/observation.html
- https://brainx.chaobrain.com/brainmass/tutorials/05_forward_models.html

Purpose: one merged signal-mapping reference for HRFBold, HRF kernels, TemporalAverage, BOLDSignal, EEG / MEG lead fields, and lead-field readouts; body only keeps the fast differentiable HRFBold path. Observation docs explicitly separate convolution BOLD from temporal averaging and point to BOLDSignal / EEG / MEG on the forward-model side.

### fitting-with-objectives-api.md

Sources:

- https://brainx.chaobrain.com/brainmass/reference/orchestration.html
- https://brainx.chaobrain.com/brainmass/tutorials/06_fitting_with_gradients.html
- https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html
- https://brainx.chaobrain.com/brainmass/howto/custom_objective.html

Purpose: exact Simulator, Network, Fitter, FitResult, objective-function, backend, and custom-loss details; body shows only Quickstart gradient fitting and routes the complete gradient-free workflow to `references/scripts/gradient-free-fitting.py`.

### datasets-api.md

Source: https://brainx.chaobrain.com/brainmass/reference/datasets.html

Purpose: exact dataset registration / loading API, Connectome, Signal, and delayed-match task containers; body only uses example_connectome and example_signal.

### visualization-analysis-api.md

Sources:

- https://brainx.chaobrain.com/brainmass/reference/viz.html
- https://brainx.chaobrain.com/brainmass/howto/analyze_results.html

Purpose: plotting helpers plus FC / FCD / spectra analysis; keep body mostly script-oriented and avoid plotting except where official script outputs require it. The visualization API is explicitly thin wrappers around matplotlib and braintools.metric outputs.

### batch-transform-acceleration.md

Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

Purpose: raw transform usage: jit, for_loop, scan, vmap, checkpointed loops, batched initial conditions, and parameter sweeps; body only shows the batch_size route.

### horn-task-training.md

Sources:

- https://brainx.chaobrain.com/brainmass/reference/horn.html
- https://brainx.chaobrain.com/brainmass/tutorials/08_training_on_tasks.html
- https://brainx.chaobrain.com/brainmass/gallery/case_studies/horn_cognitive_task.html

Purpose: HORN components and task-training loop. Training on tasks should not be a body section; open this when the user asks for sequence tasks, delayed match-to-sample, minibatches, direct optimizer loops, or HORN networks. The HORN case study notes that task training is a different loop from Fitter parameter fitting.

### parameter-sweeps-and-regime-analysis.md

Source: https://brainx.chaobrain.com/brainmass/howto/parameter_sweeps.html

Purpose: regime exploration and sensitivity analysis; open when the user asks for sweeps rather than fitting.

### braintools-cognitive-tasks.md

Parent route: `horn-task-training.md` only.

Source: https://brainx.chaobrain.com/braintools/apis/cogtask.html

Purpose: build and generate cognitive-task trials for HORN task-training workflows.

### braintools-metrics.md

Parent routes: `fitting-with-objectives-api.md` and `horn-task-training.md`.

Source: https://brainx.chaobrain.com/braintools/apis/metric.html

Purpose: select losses and metrics for objective fitting or HORN task training after the BrainMass parent workflow has been chosen.

### braintools-optimizer.md

Parent routes: `fitting-with-objectives-api.md` and `horn-task-training.md`.

Sources:

- https://brainx.chaobrain.com/braintools/apis/optim.html
- https://brainx.chaobrain.com/braintools/optim/index.html

Purpose: select optimizers, schedulers, Optax bridges, or SciPy/Nevergrad wrappers after the BrainMass parent workflow has been chosen.

### braintools-surrogate-gradient.md

Parent route: `fitting-with-objectives-api.md` only.

Source: https://brainx.chaobrain.com/braintools/apis/surrogate.html

Purpose: select a surrogate gradient only for a differentiable fitting workflow that contains a non-differentiable spike function.

## Script Reference Inventory

### Selection Rule

Use the selected gallery workflows as full-script targets. The gallery is a "curated, runnable showcase"; model-zoo pages are one self-contained demo per model family, while case studies are end-to-end applications.

Prefer complete workflows over redundant model demos. Do not include every model zoo page by default.

The canonical inventory is eleven selected scripts plus one optional script. Only `references/scripts/gradient-free-fitting.py` currently exists; every other target below remains planned and must not be treated as bundled.

| Target | Selection | Parent route | Status |
|---|---|---|---|
| `references/scripts/gradient-free-fitting.py` | Selected | `fitting-with-objectives-api.md` | Exists |
| `references/scripts/resting-state-meg-whole-brain-pipeline.py` | Selected | `coupling-network-api.md` -> `forward-observation-api.md` | Planned |
| `references/scripts/eeg-fitting-with-gradients.py` | Selected | `fitting-with-objectives-api.md` | Planned |
| `references/scripts/seizure-epileptor-case-study.py` | Selected | `modellibrary.md` | Planned |
| `references/scripts/wong-wang-decision-making.py` | Selected | `modellibrary.md` | Planned |
| `references/scripts/horn-cognitive-task-training.py` | Selected | `horn-task-training.md` | Planned |
| `references/scripts/hopf-bifurcation-single-node.py` | Selected | `modellibrary.md` | Planned |
| `references/scripts/wilson-cowan-ei-dynamics.py` | Selected | `modellibrary.md` | Planned |
| `references/scripts/jansen-rit-eeg-proxy.py` | Selected | `modellibrary.md` -> `forward-observation-api.md` | Planned |
| `references/scripts/kuramoto-synchronization.py` | Selected | `modellibrary.md` | Planned |
| `references/scripts/wong-wang-dmf-resting-state.py` | Selected | `modellibrary.md` | Planned |
| `references/scripts/linear-baseline-node.py` | Optional | `modellibrary.md` | Planned |

### gradient-free-fitting.py

Source: https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html

Purpose: standalone counterpart to the inline gradient-fitting workflow. It preserves the same Hopf amplitude-recovery objective while demonstrating bounded `SigmoidT` parameters, Nevergrad differential evolution, and SciPy Nelder-Mead. Open this script when the objective is non-differentiable, discrete, jagged, black-box, or has unavailable / unreliable gradients.

### resting-state-meg-whole-brain-pipeline.py

Source: https://brainx.chaobrain.com/brainmass/gallery/case_studies/resting_state_meg.html

Purpose: complete whole-brain pipeline: structural connectome -> delay-coupled Network -> Simulator -> MEG forward model -> FC validation. This is the most representative full workflow script.

### eeg-fitting-with-gradients.py

Source: https://brainx.chaobrain.com/brainmass/gallery/case_studies/eeg_fitting.html

Purpose: complete fitting workflow with Jansen-Rit EEG, scalar objective, Fitter, gradient backend, convergence, recovered EEG, and backend swap. Good full script for data-facing fitting.

### seizure-epileptor-case-study.py

Source: https://brainx.chaobrain.com/brainmass/gallery/case_studies/seizure_epileptor.html

Purpose: complete disease-dynamics workflow: EpileptorStep, LFP proxy, hidden slow driver z, seizure cycle, and epileptogenicity parameter x0.

### wong-wang-decision-making.py

Source: https://brainx.chaobrain.com/brainmass/gallery/case_studies/decision_making.html

Purpose: complete stochastic decision workflow: noisy trials, seeding, bistability under zero evidence, evidence sweep, and psychometric-style curve.

### horn-cognitive-task-training.py

Source: https://brainx.chaobrain.com/brainmass/gallery/case_studies/horn_cognitive_task.html

Purpose: complete task-training workflow: bundled delayed match-to-sample task, HORNSeqNetwork, direct optimizer loop, and held-out metric. Keep as bundled script, not body script.

### hopf-bifurcation-single-node.py

Source: https://brainx.chaobrain.com/brainmass/gallery/model_zoo/hopf.html

Purpose: minimal representative single-node oscillator workflow: HopfStep, Simulator, phase portrait, and bifurcation parameter a; useful as the smallest full script around the Quickstart body.

### wilson-cowan-ei-dynamics.py

Source: https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wilson_cowan.html

Purpose: representative excitatory-inhibitory population-rate workflow: WilsonCowanStep, E/I oscillations, and excitatory-drive variation. Useful because Wilson-Cowan underlies the body forward-model source network.

### jansen-rit-eeg-proxy.py

Source: https://brainx.chaobrain.com/brainmass/gallery/model_zoo/jansen_rit.html

Purpose: representative cortical-column EEG generator workflow: JansenRitStep, external input rate, EEG proxy M, and power spectrum. Useful before opening the full EEG fitting case study.

### kuramoto-synchronization.py

Source: https://brainx.chaobrain.com/brainmass/gallery/model_zoo/kuramoto.html

Purpose: representative coupled phase-oscillator workflow: heterogeneous oscillators, synchronization, coupling-strength variation, and Kuramoto order parameter.

### wong-wang-dmf-resting-state.py

Source: https://brainx.chaobrain.com/brainmass/gallery/model_zoo/wong_wang_exc_inh.html

Purpose: representative dynamic mean-field resting-state node workflow: excitatory / inhibitory Wong-Wang dynamics and local E/I balance through J_i. Distinct from decision-making WongWangStep.

### linear-baseline-node.py

Source: https://brainx.chaobrain.com/brainmass/gallery/model_zoo/linear.html

Purpose: optional sanity-check script: analytically tractable damped linear node for coupling and integration tests. Keep this optional, not a default scientific workflow.

### Not Selected By Default

- HORN model-zoo page -> covered better by horn-cognitive-task-training.py.
- every oscillator variant -> use hopf-bifurcation-single-node.py as canonical oscillator; open `references/modellibrary.md` for the exact class inventory.
- every physiological model page -> use Wilson-Cowan, Jansen-Rit, Epileptor, Wong-Wang DMF as representative anchors.

## Common Mistakes -> Fix

- Time-step a model with a bare Python `for` / `while` loop -> write a per-step function and pass it to `brainstate.transform.for_loop` or `brainstate.transform.scan`.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Wrap only the per-step function in `brainstate.transform.jit` and keep the Python loop -> `jit` compiles the step once, but Python still dispatches every iteration; use `for_loop` / `scan` for rollouts or `brainmass.Simulator` when the standard run path fits.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Use `scan` for every rollout -> use `for_loop` by default; use `scan` only when an explicit carry is needed outside the model's own `State`.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Manually append outputs inside the step function -> return monitored values from the step; `for_loop` / `scan` stack per-step outputs.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- Run long BPTT and hit memory pressure -> replace `for_loop` / `scan` in the loss with `checkpointed_for_loop` / `checkpointed_scan`.
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- treating BrainMass as a full framework -> remember BrainMass ships neural-mass-specific pieces and delegates state, units, optimizers, transforms, and backend work to BrainState / BrainUnit / BrainTools / JAX.
  Source: https://brainx.chaobrain.com/brainmass/concepts/architecture_overview.html

- adding noise in Simulator.run() -> attach OUProcess / noise process when constructing the *Step; the Simulator call does not change.
  Source: https://brainx.chaobrain.com/brainmass/getting_started/key_concepts.html

- forgetting stochastic reproducibility -> call brainstate.random.seed(s) before a noisy run intended for reporting.
  Source: https://brainx.chaobrain.com/brainmass/tutorials/03_noise.html

- misunderstanding batch_size output shape -> remember outputs gain a batch axis: (time, batch, regions).
  Source: https://brainx.chaobrain.com/brainmass/howto/batch_and_accelerate.html

- building a delay-coupled Network before setting global dt -> call brainstate.environ.set(dt=...) first so delay buffers can be sized at construction.
  Source: https://brainx.chaobrain.com/brainmass/getting_started/quickstart.html

- recording network state with the wrong monitor path -> use a callable like lambda m: m.node.x.value for coupled node state.
  Source: https://brainx.chaobrain.com/brainmass/getting_started/quickstart.html

- fitting raw oscillatory time series with pointwise RMSE -> fit a scalar summary such as amplitude, spectral feature, FC, or FCD.
  Source: https://brainx.chaobrain.com/brainmass/gallery/case_studies/eeg_fitting.html

- using gradient-free fitting as the default -> use grad when the objective is differentiable; use nevergrad / scipy when the gradient is unavailable or unreliable.
  Source: https://brainx.chaobrain.com/brainmass/tutorials/07_gradient_free_fitting.html

- using BOLDSignal as the default differentiable fitting path -> use HRFBold for fast differentiable BOLD fitting; use BOLDSignal when Balloon-Windkessel biophysics matters.
  Source: https://brainx.chaobrain.com/brainmass/tutorials/05_forward_models.html

- silently dropping units -> use explicit unit-aware quantities in the workflow and u.get_magnitude(...) only at raw-array boundaries.

- putting task training in the body -> task training is a HORN / data-driven reference path, not the core Simulator / Fitter body path.
