# Reference Markdown Plan

This plan inventories the files under `references/` plus selected skill-local reference markdown, and points each reference to the closest relevant HTML documentation source when one exists.

## Progressive-Disclosure Architecture

Primary skills are core agent task boundaries:

- `skills/braincell-multicompartment/`
- `skills/braincell-singlecell/`
- `skills/brainstate-deeplearning-training/`
- `skills/brainstate-module-building/`
- `skills/brainstate-state-management/`
- `skills/brainstate-transformations-core/`
- `skills/brainunit-quantity-safety/`
- `skills/brainx-general-guard/`

Primary references hold specialized, long-tail, environment, optimization, and advanced examples:

- `references/brainx-install/`
- `references/brainstate-randomness-reproducibility/`
- `references/brainstate-dynamics/`
- `references/brainx-acceleration-audit/`

Routing:

- `brainx-general-guard` refers to `references/brainx-install/` only for install/setup/backend/import/package issues.
- `brainstate-transformations-core` refers to `references/brainx-acceleration-audit/` for performance and acceleration work.
- `brainstate-module-building` refers to `references/brainstate-dynamics/` when module work becomes time-evolving dynamics, SNNs, delays, hooks, or event-driven spike communication.
- `brainstate-deeplearning-training` refers to `references/brainstate-randomness-reproducibility/` only when stochastic training/dropout/fit-mode/random data is involved.
- BrainCell skills mention randomness only as a conditional reference for random trials, noise, parameter sweeps, or reproducibility.

## BrainState Dynamics

### `references/brainstate-dynamics/README.md`
- **Description:** Primary reference hub for BrainState dynamics as a specialized Module pattern for time-evolving systems.
- **Mirror Source URLs:** [Dynamics and Integration](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/01_dynamics_and_integration.html)

### `references/brainstate-dynamics/brain-dynamics-delay-protocol.md`
- **Description:** Explains BrainState Brain Dynamics delay APIs and delay-buffer behavior.
- **Mirror Source URLs:** [Delay Protocol](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html)

### `references/brainstate-dynamics/brain-dynamics-event-driven-operators.md`
- **Description:** Catalogs event-driven sparse spike operators and connectivity patterns for scalable SNNs.
- **Mirror Source URLs:** [Event-Driven Operators](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/03_event_driven_operators.html)

### `references/brainstate-dynamics/brain-dynamics-snn-workflows.md`
- **Description:** Routes build, simulate, and train workflows for BrainState-style spiking neural networks.
- **Mirror Source URLs:** [Building a Spiking Neural Network](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/04_building_an_snn.html), [Training a Spiking Neural Network](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/05_training_an_snn.html)

## BrainX Infrastructure

### `references/brainx-install/README.md`
- **Description:** Environment setup, installation, import-error, backend selection, CUDA/GPU/TPU, JAX device validation, version pinning, and package mismatch reference.
- **Mirror Source URLs:** [Installing the Ecosystem](https://brainx.chaobrain.com/summ/install.html)

### `references/brainx-acceleration-audit/README.md`
- **Description:** Performance audit and acceleration reference for state-aware BrainState transform rewrites, shape stability, memory, throughput, multi-device execution, and RNG-safe benchmarking.
- **Mirror Source URLs:** [Transformations, the Essentials](https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html), [Control Flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html), [Vectorization](https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html)

## BrainCell

### `references/braincell/braincell-custom-ion-channel-authoring.md`
- **Description:** Custom BrainCell ion/channel authoring notes moved out of the skill list into shared reference markdown.
- **Mirror Source URLs:** [Ions and Channels](https://brainx.chaobrain.com/braincell/concepts/ions_channels.html), [Channels](https://brainx.chaobrain.com/braincell/tutorials/channel.html), [Extending BrainCell](https://brainx.chaobrain.com/braincell/developer/extending.html)

### `references/braincell/braincell-manual-morphology-construction.md`
- **Description:** Manual morphology construction and morphology-preparation notes moved out of the skill list into shared reference markdown.
- **Mirror Source URLs:** [Morphology concept](https://brainx.chaobrain.com/braincell/concepts/morphology.html)

### `references/braincell/morphology-io-loading-validation.md`
- **Description:** Unified reference for morphology import paths, validation reports/options, NeuroMorpho caching, and checkpoints.
- **Mirror Source URLs:** [IO Overview](https://brainx.chaobrain.com/braincell/file_formats/overview.html), [SWC](https://brainx.chaobrain.com/braincell/file_formats/swc.html), [Neurolucida ASC](https://brainx.chaobrain.com/braincell/file_formats/asc.html), [NeuroML2](https://brainx.chaobrain.com/braincell/file_formats/neuroml2.html), [NeuroMorpho.Org](https://brainx.chaobrain.com/braincell/file_formats/neuromorpho.html), [Checkpointing](https://brainx.chaobrain.com/braincell/file_formats/checkpointing.html), [Morphology concept](https://brainx.chaobrain.com/braincell/concepts/morphology.html)

### `references/braincell/probe-reference.md`
- **Description:** Documents BrainCell probes for recording runtime state, mechanism state, currents, and traces.
- **Mirror Source URLs:** [Mechanisms in BrainCell](https://brainx.chaobrain.com/braincell/tutorials/mech.html)

### `references/braincell/topology-building-and-visualization.md`
- **Description:** Guides runtime topology inspection and visualization for multicompartment BrainCell cells.
- **Mirror Source URLs:** [Point Tree Visualization](https://brainx.chaobrain.com/braincell/tutorials/vis.html), [Region and Locset Filters](https://brainx.chaobrain.com/braincell/tutorials/filter.html)

## BrainState

### Skill-local deeplearning-training references

#### `skills/brainstate-deeplearning-training/references/braintools-optimizer-reference.md`
- **Description:** Routes Braintools optimizer, external optimizer, and learning-rate-scheduler choices for BrainState training loops.
- **Mirror Source URLs:** [Optimization](https://brainx.chaobrain.com/braintools/optim/index.html)

### Skill-local module-building references

#### `skills/brainstate-module-building/references/size-inference-with-convolution.md`
- **Description:** Small module-building reference for `Conv2d.in_size`, `Conv2d.out_size`, kernel size, padding, and stride shape inference.
- **Mirror Source URLs:** [Basic Neural Network Layers](https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html)

#### `skills/brainstate-module-building/references/size-inference-with-pooling-flatten.md`
- **Description:** Small module-building reference for pooling dimension reduction and `Flatten` shape conversion before dense classifiers.
- **Mirror Source URLs:** [Basic Neural Network Layers](https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html)

### `references/brainstate-randomness-reproducibility/README.md`
- **Description:** Primary reference for BrainState seed control, random trials, stochastic modules, dropout/noise, RNG under transforms, and reproducibility.
- **Mirror Source URLs:** [Random Number Generation](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html)

### `references/brainstate-randomness-reproducibility/advanced-randomness.md`
- **Description:** Catalogs advanced BrainState RNG streams, stochastic layers, and transformed-randomness patterns.
- **Mirror Source URLs:** [Random Number Generation](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html)

### `references/brainstate/brainstate-control-flow-patterns.md`
- **Description:** Collects loop and branch patterns that remain valid under BrainState and JAX transformations.
- **Mirror Source URLs:** [Control Flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html)

### `references/brainstate/parameter-constraints-regularization.md`
- **Description:** Conceptual workflow for `ParamState` vs `nn.Param`, constrained values, regularization penalties, `Const`, and training-loss integration.
- **Mirror Source URLs:** [Parameters, Transforms, and Regularization](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html)

### `references/brainstate/parameter-containers-transforms-catalog.md`
- **Description:** Catalog for `Param`, `Const`, and built-in parameter transforms; maps constraints to transform choices.
- **Mirror Source URLs:** [Parameter Containers API](https://brainx.chaobrain.com/brainstate/apis/nn/parameters.html)

### `references/brainstate/regularization-catalog-priors.md`
- **Description:** Catalog for classical, structural, chained, and prior-distribution regularizers; maps modeling goals to `reg=` choices.
- **Mirror Source URLs:** [Standard Regularizations API](https://brainx.chaobrain.com/brainstate/apis/nn/regularization.html)

### `references/brainstate/transformation-grad-expansion.md`
- **Description:** Expands gradient and autodiff teaching for differentiable simulation and parameter fitting.
- **Mirror Source URLs:** [Automatic Differentiation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/02_autodiff.html), [Training and Metrics](https://brainx.chaobrain.com/brainstate/tutorials/core/07_training_and_metrics.html), [Parameters, Transforms, and Regularization](https://brainx.chaobrain.com/brainstate/tutorials/core/05_parameters_transforms_regularization.html)

### `references/brainstate/transformation-jit-expansion.md`
- **Description:** Expands BrainState-aware JIT compilation, state write-back, cache, and static-argument guidance.
- **Mirror Source URLs:** [JIT and Compilation](https://brainx.chaobrain.com/brainstate/tutorials/transformations/01_jit_and_compilation.html), [Transformations, the Essentials](https://brainx.chaobrain.com/brainstate/tutorials/core/06_transformations_essentials.html)

### `references/brainstate/transformation-vmap-expansion.md`
- **Description:** Expands BrainState vectorization, batching, state axes, sweeps, and stochastic vmap patterns.
- **Mirror Source URLs:** [Vectorization](https://brainx.chaobrain.com/brainstate/tutorials/transformations/03_vectorization.html), [Random Number Generation](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html), [pmap and parallelism](https://brainx.chaobrain.com/brainstate/tutorials/transformations/04_advanced_batching.html)

## Diagnostics

### `references/diagnostics/brainstate-transformed-diagnostics.md`
- **Description:** Collects runtime debugging, checking, and error-handling patterns for transformed BrainState code.
- **Mirror Source URLs:** [Error Handling and Runtime Checks](https://brainx.chaobrain.com/brainstate/tutorials/transformations/06_error_handling_and_checks.html), [Debugging Transformed Code](https://brainx.chaobrain.com/brainstate/tutorials/transformations/07_debugging.html)

### `references/diagnostics/common-failures-index.md`
- **Description:** Maps recurring BrainX failure patterns to the skill or reference that should handle them.
- **Mirror Source URLs:** None - local cross-reference router.

## Libraries

### `references/libraries/channel-library.md`
- **Description:** Catalogs BrainCell channels and channel-modeling patterns.
- **Mirror Source URLs:** [BrainCell channel API](https://brainx.chaobrain.com/braincell/apis/braincell.channel.html), [Channels](https://brainx.chaobrain.com/braincell/tutorials/channel.html)

### `references/libraries/cv-policy-reference.md`
- **Description:** Catalogs BrainCell control-volume policies and discretization tradeoffs.
- **Mirror Source URLs:** [Discretization](https://brainx.chaobrain.com/braincell/concepts/discretization.html)

### `references/libraries/filter-function-library.md`
- **Description:** Catalogs BrainCell region and locset filters for targeting mechanisms, probes, and clamps.
- **Mirror Source URLs:** [BrainCell filter API](https://brainx.chaobrain.com/braincell/apis/filter.html)

### `references/libraries/ion-library.md`
- **Description:** Catalogs BrainCell ion species and ion-related modeling patterns.
- **Mirror Source URLs:** [BrainCell ion API](https://brainx.chaobrain.com/braincell/apis/braincell.ion.html)

### `references/libraries/prebuilt-activation-library.md`
- **Description:** Catalogs BrainState activation and normalization components for module-building tasks.
- **Mirror Source URLs:** [BrainState activation API](https://brainx.chaobrain.com/brainstate/apis/nn/activation.html)

### `references/libraries/prebuilt-layer-library.md`
- **Description:** Catalogs BrainState prebuilt layers so agents reuse existing components before writing custom modules.
- **Mirror Source URLs:** [Linear layers](https://brainx.chaobrain.com/brainstate/apis/nn/linear.html), [Convolutional layers](https://brainx.chaobrain.com/brainstate/apis/nn/conv.html), [Normalization layers](https://brainx.chaobrain.com/brainstate/apis/nn/normalization.html), [Pooling layers](https://brainx.chaobrain.com/brainstate/apis/nn/pooling.html), [Padding layers](https://brainx.chaobrain.com/brainstate/apis/nn/padding.html), [Dropout layers](https://brainx.chaobrain.com/brainstate/apis/nn/dropout.html)

### `references/libraries/solver-library-with-effects.md`
- **Description:** Catalogs BrainCell and BrainState solver and integration choices with modeling consequences.
- **Mirror Source URLs:** [BrainCell integration API](https://brainx.chaobrain.com/braincell/apis/integration.html), [Choosing and Using Solvers](https://brainx.chaobrain.com/braincell/integration/solvers.html), [Advanced Integration](https://brainx.chaobrain.com/braincell/integration/advanced.html)

## Local Index and Policy

### `references/index.md`
- **Description:** Top-level router summarizing available references and when to open each one.
- **Mirror Source URLs:** None - local reference index.

### `references/policy/.gitkeep`
- **Description:** Keeps the reserved `references/policy/` directory present until policy references are added.
- **Mirror Source URLs:** None - placeholder file.
